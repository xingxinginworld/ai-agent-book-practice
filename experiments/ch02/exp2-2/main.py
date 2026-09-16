# -*- coding: utf-8 -*-
"""
实验 2-2 精简版：用本地 Qwen3-0.6B 可视化注意力权重热力图
============================================================================
前置：
    1) 安装依赖  : pip install torch transformers matplotlib numpy
    2) 联网首次运行会自动下载 Qwen3-0.6B 权重（约 1~2 GB）
运行示例：
    python main.py                                                        # 默认提示词热力图（最后一层、头平均）
    python main.py --prompt "北京 的 天气 怎么样" --layer 0 --head 3 --output layer0_head3.png
    python main.py --prompt "用一句话解释注意力机制。" --max-new-tokens 40
    python main.py --compare-layers 0 13 -1 --output layer_compare.png
说明：
    这是「教学用最小可运行版」，专注演示"加载模型 → 取注意力 → 画热力图 → 看 sink"的核心链路；
    官方完整项目见 bojieli/ai-agent-book/chapter2/attention_visualization（含前端 React、
    带工具调用的 ReAct Agent、多层对比与统计分析等）。若要交互式前端，请走官方项目。
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无界面环境也能保存图片
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---- 默认配置 ----
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_PROMPT = "北京 的 天气 怎么样"


def pick_device():
    """自动选择可用设备：CUDA > MPS > CPU。"""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def load_model(model_name: str, device: str):
    """加载因果语言模型，并强制 eager 注意力实现（flash attention 不返回注意力权重）。"""
    print(f"⏬ 加载模型 {model_name}（首次会从 Hugging Face 下载权重，约 1~2 GB）...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        attn_implementation="eager",   # 关键：确保 output_attentions 能拿到权重
    ).to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def get_attention(prompt: str, model, tokenizer, device: str,
                  max_new_tokens: int = 0, use_chat_template: bool = True):
    """
    返回：(attentions 元组, token列表, 层数)
    - 可选先生成 max_new_tokens 个续写 token，再对完整序列做一次前向取注意力
    - 默认只可视化输入提示词本身（max_new_tokens=0）
    """
    # 1. 组装文本：是否套用 Qwen3 的 chat 模板
    if use_chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
    else:
        text = prompt

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    # 2. 可选：先让模型续写一段，得到"输入 + 输出"的完整序列
    if max_new_tokens > 0:
        generated = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        input_ids = generated

    # 3. 单次前向，开启 output_attentions
    outputs = model(input_ids, output_attentions=True, return_dict=True)
    attentions = outputs.attentions          # tuple[层数]，每层 [B, H, S, S]
    n_layers = len(attentions)

    # 4. 解码 token（保留特殊符号便于对照注意力位置）
    token_ids = input_ids[0].tolist()
    tokens = [tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]
    return attentions, tokens, n_layers


def pick_matrix(attentions, n_layers, layer: int, head: int):
    """按层/头选取注意力矩阵，并跨 head 平均 → [S, S]，同时返回真实层号。"""
    layer_idx = n_layers - 1 if layer == -1 else layer
    attn = attentions[layer_idx][0]          # [H, S, S]
    if head != -1:
        attn = attn[head:head + 1]          # 只取某一头
    matrix = attn.mean(dim=0).cpu().numpy()  # 跨 head 平均
    return matrix, layer_idx


def save_heatmap(matrix, tokens, layer_idx, out_path: str, cmap: str = "viridis"):
    """画因果注意力热力图并保存。"""
    S = len(tokens)
    # 因果掩码：每个 token 只能看到自身及之前的 token（上三角应被遮挡）
    mask = np.triu(np.ones((S, S), dtype=bool), k=1)
    disp = np.ma.masked_array(matrix, mask)

    fig, ax = plt.subplots(figsize=(max(6, S * 0.5), max(5, S * 0.45)))
    im = ax.imshow(disp, cmap=cmap, vmin=0, vmax=float(matrix.max()))
    ax.set_xticks(range(S))
    ax.set_yticks(range(S))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("Key 位置（被关注的 token）")
    ax.set_ylabel("Query 位置（正在计算的 token）")
    ax.set_title(f"注意力热力图 · 第 {layer_idx} 层（heads 平均）")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"🖼 已保存热力图：{out_path}")


def save_compare(attentions, tokens, n_layers, layer_ids, out_path, cmap="viridis"):
    """把若干层并排画在一起，观察 attention sink 如何随层数演变。"""
    figs = len(layer_ids)
    fig, axes = plt.subplots(1, figs, figsize=(6 * figs, 5))
    if figs == 1:
        axes = [axes]
    S = len(tokens)
    mask = np.triu(np.ones((S, S), dtype=bool), k=1)
    for ax, lid in zip(axes, layer_ids):
        idx = n_layers - 1 if lid == -1 else lid
        m = attentions[idx][0].mean(dim=0).cpu().numpy()
        disp = np.ma.masked_array(m, mask)
        im = ax.imshow(disp, cmap=cmap, vmin=0, vmax=float(m.max()))
        ax.set_title(f"第 {idx} 层")
        ax.set_xticks(range(S))
        ax.set_yticks(range(S))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"🖼 已保存多层对比图：{out_path}")


def report_sink(matrix, tokens):
    """打印 attention sink：每行对第 1 个 token 的注意力占比。"""
    sink = matrix[:, 0]   # 每个 query 对第 0 个 key 的注意力
    print("\n📌 Attention Sink（每行对第 1 个 token 的注意力占比）：")
    for t, v in zip(tokens, sink):
        print(f"   {t!r:>14}: {v:.3f}")
    print(f"   平均 sink 占比 ≈ {sink.mean():.3f}")


def parse_args():
    ap = argparse.ArgumentParser(description="实验 2-2：本地模型注意力可视化（精简版）")
    ap.add_argument("-p", "--prompt", default=DEFAULT_PROMPT, help="待可视化的文本")
    ap.add_argument("-o", "--output", default="attention_heatmap.png", help="输出 PNG 路径")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL, help="HF 模型名或本地路径")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu，默认自动检测")
    ap.add_argument("-l", "--layer", type=int, default=-1, help="层索引，-1=最后一层")
    ap.add_argument("--head", type=int, default=-1, help="头索引，-1=跨头平均")
    ap.add_argument("--max-new-tokens", type=int, default=0, help="先生成 N 个 token 再可视化整段")
    ap.add_argument("--no-chat-template", action="store_true", help="不套用 chat 模板，直接喂原始文本")
    ap.add_argument("--cmap", default="viridis", help="matplotlib 色图")
    ap.add_argument("--compare-layers", nargs="+", type=int, default=None,
                    help="并排画多层，例如 --compare-layers 0 13 -1")
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device or pick_device()
    print(f"🖥 设备：{device}")

    model, tokenizer = load_model(args.model, device)
    attentions, tokens, n_layers = get_attention(
        args.prompt, model, tokenizer, device,
        max_new_tokens=args.max_new_tokens,
        use_chat_template=not args.no_chat_template,
    )
    print(f"🔢 序列长度 {len(tokens)} tokens，模型共 {n_layers} 层")

    if args.compare_layers:
        save_compare(attentions, tokens, n_layers, args.compare_layers, args.output, args.cmap)
        # 对比模式下也顺带报最后一层 sink
        matrix, layer_idx = pick_matrix(attentions, n_layers, -1, -1)
        report_sink(matrix, tokens)
    else:
        matrix, layer_idx = pick_matrix(attentions, n_layers, args.layer, args.head)
        save_heatmap(matrix, tokens, layer_idx, args.output, args.cmap)
        report_sink(matrix, tokens)
        print("\n💡 观察要点：上三角被遮挡（因果掩码）；最后一层常出现 attention sink"
              "——大量注意力压在第 1 个 token 上；第 0 层更接近局部对角。")


if __name__ == "__main__":
    main()
