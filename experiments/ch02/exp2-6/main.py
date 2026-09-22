# -*- coding: utf-8 -*-
"""
实验 2-6：使用 Agent Skills 从论文生成演示文稿（轻量本地可跑版）
================================================================================
对应《深入理解 AI Agent》第 2 章 · 上下文工程 · 实验 2-6（原书目录名 agent-skills-ppt）。

【这个实验在测什么】
第 2 章讲"上下文工程"——如何把对模型有用的信息组织好、按需喂给模型。
本书用"Agent Skills"这一机制做了一个很典型的演示：把"论文 → 演示文稿"这件事
封装成一个**可复用的能力包（Skill）**，Agent 在需要时按 description 自动加载它。

本实验把这套能力落成可运行代码：
  论文文本 ──①抽取──▶ 结构化要素(JSON) ──②大纲──▶ 幻灯片大纲(JSON) ──③渲染──▶ 离线HTML演示文稿
其中①②用本地 LLM（默认 qwen3:0.6b），③是纯字符串渲染，不调模型。

【为什么值得做】
- 它把"长文档 → 短汇报"的信息压缩过程显式化，正是上下文工程的练兵场。
- 它演示了 Skill 的本质：一份 SKILL.md（含 name/description + 流程说明）+ 背后的脚本。
  本目录的 skill.md 就是这份"能力包"的说明书。

【如何运行】
  pip install -r experiments/ch02/exp2-6/requirements.txt
  python experiments/ch02/exp2-6/main.py --demo            # 用内置示例论文跑通全流程
  python experiments/ch02/exp2-6/main.py --paper 论文.txt   # 自己的论文（.txt/.md/.pdf）
  python experiments/ch02/exp2-6/main.py --paper 论文.pdf --out out.html --slides 8

说明：默认【关闭】模型原生思考（向 Qwen3 注入 enable_thinking:false），理由同 2-4/2-5——
原生思考会吃掉 token 预算、增加随机性、污染对照纯度。需观察思考影响时加 --think。
"""

import os
import sys
import json
import argparse
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 0. 配置加载：与 2-1~2-5 完全一致的变量名，切换服务商只改 .env 的值
# ---------------------------------------------------------------------------
load_dotenv()  # 读取仓库根目录的 .env（已被 .gitignore 忽略，切勿提交）

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
MODEL = os.getenv("OPENAI_MODEL", "qwen3:0.6b")

RAW_EXTRA = os.getenv("EXTRA_BODY", "{}")
try:
    EXTRA_BODY = json.loads(RAW_EXTRA) if RAW_EXTRA.strip() else {}
except json.JSONDecodeError:
    print(f"[警告] EXTRA_BODY 不是合法 JSON：{RAW_EXTRA!r}，已忽略")
    EXTRA_BODY = {}

try:
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))
except ValueError:
    MAX_TOKENS = 1500

# 本实验内部两个阶段的默认最大生成长度（抽取偏短，大纲偏长）
EXTRACT_MAX_TOKENS = min(MAX_TOKENS, 900)
OUTLINE_MAX_TOKENS = MAX_TOKENS

from prompts import EXTRACT_SYSTEM, EXTRACT_USER, OUTLINE_SYSTEM, OUTLINE_USER
from render import render_slides


def build_client():
    """构造 OpenAI 兼容客户端；缺 Key 直接报错退出，绝不发起请求。"""
    if not API_KEY:
        print("❌ 未检测到 OPENAI_API_KEY。请复制仓库根目录 .env.example 为 .env 并填入密钥。")
        print("   本地 Ollama 路径可填 OPENAI_API_KEY=ollama 占位（模型已在本地，无需真实 Key）。")
        sys.exit(1)
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ 缺少依赖 openai，请先 pip install -r experiments/ch02/exp2-6/requirements.txt")
        sys.exit(1)
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60)


def call_model(client, system, user, think):
    """发一次请求，返回文本。默认关原生思考、temperature=0 保证可复现。"""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    extra = dict(EXTRA_BODY)
    if not think and "enable_thinking" not in extra:
        extra["enable_thinking"] = False
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        extra_body=extra,
        temperature=0.0,
    )
    text = resp.choices[0].message.content or ""
    if not text and getattr(resp.choices[0].message, "reasoning_content", None):
        text = resp.choices[0].message.reasoning_content or ""
    return text


def extract_first_json(text):
    """从模型输出里稳健地抠出第一个 {...} JSON 块。

    0.6B 模型常在 JSON 外裹 markdown 代码围栏、或前后加废话，
    这里直接找最外层首个 '{' 到配平 '}' 的子串再 json.loads。
    """
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


# ---------------------------------------------------------------------------
# 1. 论文读取（.txt / .md / .pdf）
# ---------------------------------------------------------------------------
def read_paper(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            print("❌ 读取 PDF 需要 pypdf，请 pip install pypdf，或改用 .txt/.md 纯文本。")
            sys.exit(1)
        reader = PdfReader(path)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 2. 两阶段流水线
# ---------------------------------------------------------------------------
def extract_meta(client, paper_text, think):
    """阶段①：论文 → 结构化要素 dict。失败返回最小可用骨架。"""
    user = EXTRACT_USER.replace("<<PAPER>>", paper_text[:6000])  # 截断，避免超出上下文
    raw = call_model(client, EXTRACT_SYSTEM, user, think)
    data = extract_first_json(raw)
    if not isinstance(data, dict):
        print("  ⚠️ 结构化抽取未返回合法 JSON，使用最小骨架（标题取自文本首行）。")
        first_line = paper_text.strip().splitlines()[0] if paper_text.strip() else "未命名论文"
        return {
            "title": first_line[:60],
            "authors": "", "venue": "",
            "problem": "", "method": "",
            "contributions": [], "key_results": [], "conclusions": [],
        }
    # 兜底字段，保证后续大纲阶段不会 KeyError
    for k in ("title", "authors", "venue", "problem", "method"):
        data.setdefault(k, "")
    for k in ("contributions", "key_results", "conclusions"):
        if not isinstance(data.get(k), list):
            data[k] = []
    return data


def build_outline(client, meta, think, max_slides):
    """阶段②：结构化摘要 → 幻灯片大纲 list[dict]。失败用规则兜底。"""
    summary = json.dumps(meta, ensure_ascii=False, indent=2)
    user = OUTLINE_USER.replace("<<SUMMARY>>", summary).replace("<<MAX>>", str(max_slides))
    raw = call_model(client, OUTLINE_SYSTEM, user, think)
    data = extract_first_json(raw)
    slides = data.get("slides") if isinstance(data, dict) else None
    if not isinstance(slides, list) or not slides:
        print("  ⚠️ 大纲生成未返回合法 JSON，改用规则兜底（按结构化要素铺成幻灯片）。")
        return _fallback_outline(meta)
    cleaned = []
    for sl in slides:
        if not isinstance(sl, dict):
            continue
        cleaned.append({
            "title": str(sl.get("title") or "未命名"),
            "bullets": [str(b) for b in sl.get("bullets", []) if str(b).strip()],
        })
    if not cleaned:
        return _fallback_outline(meta)
    return cleaned


def _fallback_outline(meta):
    """解析失败时，用结构化要素直接铺成幻灯片（不调模型）。"""
    slides = []
    if meta.get("problem"):
        slides.append({"title": "研究背景与问题", "bullets": [meta["problem"]]})
    if meta.get("method"):
        slides.append({"title": "方法/核心思路", "bullets": [meta["method"]]})
    if meta.get("contributions"):
        slides.append({"title": "主要贡献", "bullets": meta["contributions"]})
    if meta.get("key_results"):
        slides.append({"title": "关键结果", "bullets": meta["key_results"]})
    if meta.get("conclusions"):
        slides.append({"title": "总结与启示", "bullets": meta["conclusions"]})
    if not slides:
        slides.append({"title": "论文要点", "bullets": ["（模型未给出结构化信息，请检查输入或换更强模型）"]})
    # 确保最后一张是总结
    if slides[-1]["title"] != "总结与启示":
        slides.append({"title": "总结与启示", "bullets": ["详见论文原文"]})
    return slides


# ---------------------------------------------------------------------------
# 3. 主流程
# ---------------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="实验 2-6：论文 → 演示文稿（Agent Skill 演示）")
    ap.add_argument("--paper", help="论文路径（.txt/.md/.pdf）")
    ap.add_argument("--demo", action="store_true", help="使用内置示例论文（examples/sample_paper.txt）")
    ap.add_argument("--out", help="输出 HTML 路径（默认 output/<论文名>_slides.html）")
    ap.add_argument("--slides", type=int, default=8, help="幻灯片最大张数（默认 8，含封面）")
    ap.add_argument("--think", action="store_true", help="开启模型原生思考（默认关闭）")
    ap.add_argument("--model", help="覆盖 .env 中的 OPENAI_MODEL")
    args = ap.parse_args()

    global MODEL
    if args.model:
        MODEL = args.model

    # 解析输入论文路径
    if args.demo:
        paper_path = os.path.join(here, "examples", "sample_paper.txt")
    elif args.paper:
        paper_path = args.paper
    else:
        print("❌ 请指定 --paper <路径> 或 --demo（内置示例）。")
        sys.exit(1)

    if not os.path.exists(paper_path):
        print(f"❌ 找不到论文文件：{paper_path}")
        sys.exit(1)

    # 输出路径
    if args.out:
        out_path = args.out
    else:
        base = os.path.splitext(os.path.basename(paper_path))[0]
        out_dir = os.path.join(here, "output")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base}_slides.html")

    print(f"模型：{MODEL} | 端点：{BASE_URL}")
    print(f"论文：{paper_path}")
    print(f"输出：{out_path} | 最大 {args.slides} 张 | think={args.think}")
    print("=" * 72)

    client = build_client()

    # 读论文
    print("[1/3] 读取论文文本 ...")
    paper_text = read_paper(paper_path)
    print(f"      字数：{len(paper_text)}")

    # 阶段① 抽取
    print("[2/3] 抽取结构化要素 ...")
    meta = extract_meta(client, paper_text, args.think)
    print(f"      标题：{meta.get('title')}")
    print(f"      作者：{meta.get('authors')}  出处：{meta.get('venue')}")

    # 阶段② 大纲
    print("[3/3] 生成幻灯片大纲 ...")
    slides = build_outline(client, meta, args.think, args.slides)
    print(f"      幻灯片数：{len(slides)} 张（含封面）")

    # 阶段③ 渲染
    html = render_slides(meta, slides, max_slides=args.slides)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✅ 完成，演示文稿已写出：{out_path}")
    print("      用浏览器打开即可翻页（←/→/空格 翻页，F 全屏）。")


if __name__ == "__main__":
    main()
