# -*- coding: utf-8 -*-
"""
实验 2-2 精简版：用本地 Qwen 模型可视化注意力权重热力图
============================================================================
前置：
    1) 安装依赖  : pip install torch transformers matplotlib numpy
    2) 联网首次运行会自动下载模型权重（Qwen3-0.6B 约 1~2 GB；其他模型另计）
运行示例：
    python main.py                                                        # 默认提示词热力图（最后一层、头平均）
    python main.py --prompt "北京 的 天气 怎么样" --layer 0 --head 3 --output layer0_head3.png
    python main.py --prompt "用一句话解释注意力机制。" --max-new-tokens 40
    python main.py --compare-layers 0 13 -1 --output layer_compare.png
    python main.py --compare-models Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B -o compare.png   # 双模型并排对比
    python main.py --all-layers                                          # 单模型全部 28 层网格图
    python main.py --compare-models Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B --all-layers -o compare_all.png   # 两模型各出一张层网格图
说明：
    这是「教学用最小可运行版」，专注演示"加载模型 → 取注意力 → 画热力图 → 看 sink"的核心链路；
    官方完整项目见 bojieli/ai-agent-book/chapter2/attention_visualization（含前端 React、
    带工具调用的 ReAct Agent、多层对比与统计分析等）。若要交互式前端，请走官方项目。
"""

import argparse
import gc
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无界面环境也能保存图片
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

# 让热力图里的汉字（标题 / 坐标轴）正常显示，避免变成方框
def _pick_cjk_font():
    for name in ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
                 "WenQuanYi Micro Hei", "Arial Unicode MS", "PingFang SC"]:
        try:
            if _fm.findfont(_fm.FontProperties(family=name)):
                return name
        except Exception:
            continue
    return None

_cn = _pick_cjk_font()
if _cn:
    plt.rcParams["font.family"] = _cn
plt.rcParams["axes.unicode_minus"] = False

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


def _model_is_cached(model_name: str) -> bool:
    """粗略判断 HF 缓存里是否已有该模型权重（snapshots 存在即认为已下载）。"""
    if os.path.exists(model_name):   # 本地目录路径，直接认为可用
        return True
    if "/" not in model_name:
        return False
    org, name = model_name.split("/", 1)
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    cand = os.path.join(cache_dir, f"models--{org}--{name}")
    snap = os.path.join(cand, "snapshots")
    return os.path.isdir(cand) and os.path.isdir(snap)


@torch.no_grad()
def load_model(model_name: str, device: str, local_files_only=None):
    """加载因果语言模型，并强制 eager 注意力实现（flash attention 不返回注意力权重）。

    local_files_only 为 None 时自动探测：若本地缓存已存在该模型权重，则离线加载，
    避免对已下载好的模型做无谓联网（否则在无法直连 Hugging Face 的网络下会卡在超时）。
    未缓存的模型仍走联网下载。
    """
    if local_files_only is None:
        local_files_only = _model_is_cached(model_name)
    mode = "离线（使用本地缓存）" if local_files_only else "联网（首次会从 Hugging Face 下载权重）"
    print(f"⏬ 加载模型 {model_name}（{mode}）...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        attn_implementation="eager",   # 关键：确保 output_attentions 能拿到权重
        local_files_only=local_files_only,
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
    # 1. 组装文本：是否套用模型的 chat 模板（不同模型自带不同模板，无需手动指定）
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
    attn = attentions[layer_idx]
    # 兼容不同 transformers 版本：注意力可能是 torch.Tensor，也可能是 numpy 数组
    if not isinstance(attn, torch.Tensor):
        attn = torch.from_numpy(np.asarray(attn))
    if attn.dim() == 4:
        attn = attn[0]                       # 去掉 batch 维 → [H, S, S]
    if head != -1:
        attn = attn[head:head + 1]          # 只取某一头
    matrix = attn.float().mean(dim=0).detach().cpu().numpy()  # 跨 head 平均
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
        m = attentions[idx]
        # 兼容不同 transformers 版本：注意力可能是 torch.Tensor，也可能是 numpy 数组
        if not isinstance(m, torch.Tensor):
            m = torch.from_numpy(np.asarray(m))
        if m.dim() == 4:
            m = m[0]                         # 去掉 batch 维 → [H, S, S]
        m = m.float().mean(dim=0).detach().cpu().numpy()
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


def save_compare_models(matrices, tokens, model_names, layer_idx, out_path, cmap="viridis"):
    """两个模型并排画热力图，便于直观对比 attention sink（统一 vmax 才能比颜色深浅）。"""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]
    S = len(tokens)
    mask = np.triu(np.ones((S, S), dtype=bool), k=1)
    vmax = max(float(m.max()) for m in matrices)
    for ax, mat, name in zip(axes, matrices, model_names):
        disp = np.ma.masked_array(mat, mask)
        im = ax.imshow(disp, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(f"{name}\n第 {layer_idx} 层（heads 平均）", fontsize=10)
        ax.set_xticks(range(S))
        ax.set_yticks(range(S))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)
        ax.set_xlabel("Key 位置（被关注的 token）")
        ax.set_ylabel("Query 位置（正在计算的 token）")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"🖼 已保存双模型对比图：{out_path}")


def report_sink(matrix, tokens):
    """打印 attention sink：每行对第 1 个 token 的注意力占比。"""
    sink = matrix[:, 0]   # 每个 query 对第 0 个 key 的注意力
    print("\n📌 Attention Sink（每行对第 1 个 token 的注意力占比）：")
    for t, v in zip(tokens, sink):
        print(f"   {t!r:>14}: {v:.3f}")
    print(f"   平均 sink 占比 ≈ {sink.mean():.3f}")


def all_layer_matrices(attentions, n_layers, head):
    """取所有层的跨头平均注意力矩阵，返回 [S,S] 列表（第 L 个对应第 L 层）。"""
    return [pick_matrix(attentions, n_layers, L, head)[0] for L in range(n_layers)]


def short_name(model_name: str) -> str:
    """从模型名取短名用于文件名/标题，如 Qwen/Qwen3-0.6B -> Qwen3-0.6B。"""
    return model_name.split("/", 1)[-1]


def save_all_layers_grid(matrices, tokens, title, out_path, cmap="viridis"):
    """把全部层画成网格图，统一色标便于跨层比较 attention sink 强弱。"""
    n = len(matrices)
    cols = min(7, n)
    rows = (n + cols - 1) // cols
    S = len(tokens)
    mask = np.triu(np.ones((S, S), dtype=bool), k=1)
    vmax = max(float(m.max()) for m in matrices)   # 共享色标：颜色越亮 sink 越强
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
    axes = np.array(axes).flatten()
    for L in range(n):
        m = matrices[L]
        disp = np.ma.masked_array(m, mask)
        ax = axes[L]
        im = ax.imshow(disp, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(f"L{L}\n{m[:, 0].mean():.2f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for L in range(n, len(axes)):
        axes[L].axis("off")
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
    fig.colorbar(im, cax=cbar_ax)
    fig.suptitle(f"{title}\n序列：{''.join(tokens)}", fontsize=10, y=0.99)
    plt.subplots_adjust(right=0.9, top=0.9)
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"🖼 已保存全部层热力图：{out_path}（共享色标 vmax={vmax:.2f}，标题数字=该层平均 sink）")


def report_sink_per_layer(matrices, tokens, model_name):
    """逐层打印平均 sink 占比，方便对照 sink 随层数演变。"""
    print(f"\n📊 {model_name} 各层平均 sink 占比（每行对第 1 个 token）：")
    vals = []
    for L, m in enumerate(matrices):
        v = m[:, 0].mean()
        vals.append(v)
        print(f"   L{L:>2}: {v:.3f}")
    print(f"   各层 sink 范围 {min(vals):.3f} ~ {max(vals):.3f}，均值 ≈ {sum(vals) / len(vals):.3f}")


def run_compare(args, device):
    """双模型对比：依次加载→取全部层注意力→释放，控制峰值内存。

    --all-layers 时：为两个模型各生成一张「全部层网格图」；
    否则：对指定层（默认最后一层）做并排双图。
    采用「逐模型加载-释放」而非同时加载，对纯 CPU、内存偏紧的机器更友好。
    """
    m1, m2 = args.compare_models
    print(f"\n🔬 双模型对比模式：{m1}  vs  {m2}")
    results = []
    for mname in (m1, m2):
        model, tokenizer = load_model(mname, device)
        attentions, tokens_i, n_layers = get_attention(
            args.prompt, model, tokenizer, device,
            max_new_tokens=args.max_new_tokens,
            use_chat_template=not args.no_chat_template,
        )
        print(f"🔢 [{mname}] 序列长度 {len(tokens_i)} tokens，模型共 {n_layers} 层")
        all_mats = all_layer_matrices(attentions, n_layers, args.head)
        results.append((all_mats, tokens_i, n_layers, mname))
        # 立即释放当前模型，避免两个模型同时占用内存
        del model, tokenizer, attentions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tokens_a, tokens_b = results[0][1], results[1][1]
    if tokens_a != tokens_b:
        print("⚠️ 两模型分词结果不一致，并排热力图的行列可能不对齐；已用第一个模型的 token 作坐标轴。")
    tokens = tokens_a

    if args.all_layers:
        base = os.path.splitext(args.output)[0]
        for all_mats, _, n_layers, mname in results:
            out = f"{base}_{short_name(mname)}_all_layers.png"
            save_all_layers_grid(all_mats, tokens,
                                 f"{mname} 全部 {n_layers} 层（heads 平均）", out, args.cmap)
            report_sink_per_layer(all_mats, tokens, mname)
        return

    layer_idx = (results[0][2] - 1) if args.compare_layer == -1 else args.compare_layer
    matrices = [r[0][layer_idx] for r in results]
    save_compare_models(matrices, tokens, [r[3] for r in results], layer_idx, args.output, args.cmap)

    print("\n📊 双模型 Attention Sink 对比（每行对第 1 个 token 的注意力占比）：")
    for all_m, _, n_layers, mname in results:
        print(f"   {mname:>20}: 平均 sink 占比 ≈ {all_m[layer_idx][:, 0].mean():.3f}   （共 {n_layers} 层）")


def parse_args():
    ap = argparse.ArgumentParser(description="实验 2-2：本地模型注意力可视化（精简版）")
    ap.add_argument("-p", "--prompt", default=DEFAULT_PROMPT, help="待可视化的文本")
    ap.add_argument("-o", "--output", default="attention_heatmap.png", help="输出 PNG 路径")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL, help="HF 模型名或本地路径（单模型模式）")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu，默认自动检测")
    ap.add_argument("-l", "--layer", type=int, default=-1, help="层索引，-1=最后一层（单模型模式）")
    ap.add_argument("--head", type=int, default=-1, help="头索引，-1=跨头平均")
    ap.add_argument("--max-new-tokens", type=int, default=0, help="先生成 N 个 token 再可视化整段")
    ap.add_argument("--no-chat-template", action="store_true", help="不套用 chat 模板，直接喂原始文本")
    ap.add_argument("--cmap", default="viridis", help="matplotlib 色图")
    ap.add_argument("--compare-layers", nargs="+", type=int, default=None,
                    help="并排画多层，例如 --compare-layers 0 13 -1")
    ap.add_argument("--compare-models", nargs=2, default=None, metavar=("M1", "M2"),
                    help="双模型并排对比，例如 --compare-models Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B")
    ap.add_argument("--compare-layer", type=int, default=-1,
                    help="对比模式下可视化哪一层（默认最后一层 -1）")
    ap.add_argument("--all-layers", action="store_true",
                    help="把每一层都画成网格图（单模型，或与 --compare-models 配合各出一张）")
    return ap.parse_args()


def run_all_layers_single(args, device):
    """单模型：把全部层画成网格图。"""
    model, tokenizer = load_model(args.model, device)
    attentions, tokens, n_layers = get_attention(
        args.prompt, model, tokenizer, device,
        max_new_tokens=args.max_new_tokens,
        use_chat_template=not args.no_chat_template,
    )
    print(f"🔢 序列长度 {len(tokens)} tokens，模型共 {n_layers} 层")
    all_mats = all_layer_matrices(attentions, n_layers, args.head)
    out = args.output
    if out == "attention_heatmap.png":   # 未显式指定输出名时给个更贴切的默认名
        out = "attention_all_layers.png"
    save_all_layers_grid(all_mats, tokens, f"{args.model} 全部 {n_layers} 层（heads 平均）", out, args.cmap)
    report_sink_per_layer(all_mats, tokens, args.model)
    del model, tokenizer, attentions
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    device = args.device or pick_device()
    print(f"🖥 设备：{device}")

    # 双模型对比模式优先
    if args.compare_models:
        run_compare(args, device)
        return

    # 单模型：全部层网格图
    if args.all_layers:
        run_all_layers_single(args, device)
        return

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
