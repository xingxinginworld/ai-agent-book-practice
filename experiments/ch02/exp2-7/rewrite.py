# -*- coding: utf-8 -*-
"""
实验 2-7（配套脚本二）：按风格画像改写草稿、去 AI 味 —— rewrite.py
================================================================================
对应《深入理解 AI Agent》第 2 章 · Agent Skills · 实验 2-7
（本脚本是 Skill 的"执行体"：读 style_profile.json，把 AI 腔草稿改写成人味文字）。

【这个脚本在做什么】
skill.md 把"去 AI 味"定义成一套规则；本脚本把规则"实例化"——
先读 build_skill.py 产出的 `style_profile.json`（某人的真实文风），
再把一份 AI 生成的/套路化草稿，按该文风改写成自然、像真人写的版本：

    style_profile.json + AI草稿 ──▶ 本地 LLM 改写 ──▶ 人味文本

改写原则（来自 skill.md 的规则）：只改"说法"不改"意思"；删套路连接词与空洞总结；
长句拆短、适当口语化；不替作者添加没说过的观点/数据/情绪。

【关于"小模型可靠性"——本实验要观察的重点】
2.5B 级别的本地模型在"长系统提示 + 自由改写"上指令跟随并不稳：它可能不去改写，
而是跑去"分析原文"或"原样复述 + 加改写说明"。所以本脚本不一次性信任模型输出，
而是加了三道防线（这正是 Agent 该有的健壮性）：
  1) 自检：输出是否为空、是否夹带分析/说明、AI 味是否真的下降；
  2) 重试：未通过就换一茬温度重来，最多 3 次；
  3) 兜底：3 次都不行，降级到确定性的"规则去 AI 味"（删套路词、拆长句），绝不交白卷。

【附带的小度量】
用一组"AI 味关键词"（首先/其次/最后、值得注意的是、综上所述、在当今社会、
毫无疑问…）粗略打分，对比改写前后的"AI 味指数"，直观验证效果。

【如何运行】
  python experiments/ch02/exp2-7/rewrite.py                       # 用默认 ai_draft.txt
  python experiments/ch02/exp2-7/rewrite.py --draft 我的草稿.txt   # 指定草稿
  python experiments/ch02/exp2-7/rewrite.py --draft 我的草稿.txt --out 成品.txt
"""

import os
import re
import sys
import json
import argparse
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 0. 配置加载：与 2-1~2-7 完全一致的变量名
# ---------------------------------------------------------------------------
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
MODEL = os.getenv("OPENAI_MODEL", "minicpm-32k")

RAW_EXTRA = os.getenv("EXTRA_BODY", "{}")
try:
    EXTRA_BODY = json.loads(RAW_EXTRA) if RAW_EXTRA.strip() else {}
except json.JSONDecodeError:
    EXTRA_BODY = {}

try:
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))
except ValueError:
    MAX_TOKENS = 1500

# AI 味关键词表（用于粗略打分，命中越多越"像机器写的"）
AI_FLAVOR_WORDS = [
    "值得注意的是", "综上所述", "在当今社会", "毫无疑问", "因此可见",
    "首先", "其次", "最后", "总而言之", "换言之", "在此基础上",
    "不仅如此", "显而易见", "至关重要", "不可或缺", "亟待解决",
]

# 模型"跑偏"去分析/说明而非改写的标志性字符串
BAD_MARKERS = [
    "改写说明", "---", "**", "问题主要在于", "如果你需要", "如您",
    "如有具体的", "请随时告诉我", "可以进一步", "这段文字试图表达",
    "分析一下", "以下是我的",
]


def build_client():
    if not API_KEY:
        print("❌ 未检测到 OPENAI_API_KEY。请复制仓库根目录 .env.example 为 .env 并填入密钥。")
        sys.exit(1)
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ 缺少依赖 openai，请先 pip install -r experiments/ch02/exp2-7/requirements.txt")
        sys.exit(1)
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)


def call_model(client, system, user, think, temperature=0.3):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    extra = dict(EXTRA_BODY)
    if not think and "enable_thinking" not in extra:
        extra["enable_thinking"] = False
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=MAX_TOKENS,
        extra_body=extra, temperature=temperature,
    )
    text = resp.choices[0].message.content or ""
    if not text and getattr(resp.choices[0].message, "reasoning_content", None):
        text = resp.choices[0].message.reasoning_content or ""
    return text


# ---------------------------------------------------------------------------
# 1. 加载风格画像 + AI 味打分
# ---------------------------------------------------------------------------
def load_profile(path):
    if not os.path.exists(path):
        print(f"❌ 找不到风格画像 {path}。请先运行 build_skill.py 生成。")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ai_flavor_score(text):
    """粗略 AI 味指数：命中关键词次数 / 字数 × 1000，便于前后对比。"""
    hits = sum(text.count(w) for w in AI_FLAVOR_WORDS)
    return round(hits / max(len(text), 1) * 1000, 1), hits


def is_bad_rewrite(rewritten, before_score):
    """自检：这次改写是否"跑偏"（空 / 夹带分析说明 / AI 味没降）。"""
    if not rewritten or not rewritten.strip():
        return True
    if any(m in rewritten for m in BAD_MARKERS):
        return True
    after_score, _ = ai_flavor_score(rewritten)
    if after_score >= before_score:
        return True
    return False


# ---------------------------------------------------------------------------
# 2. 改写（核心）：自检 + 重试 + 规则兜底
# ---------------------------------------------------------------------------
def build_rewrite_prompt(profile, draft):
    p = profile
    vocab = "、".join(p.get("vocabulary", [])) or "（未特别标注）"
    patterns = "；".join(p.get("sentence_patterns", [])) or "（未特别标注）"
    avoid = "；".join(p.get("avoid", [])) or "（无）"
    examples = "\n".join(f"  · {e}" for e in p.get("examples", [])) or "  · （无）"
    system = (
        "你是一位中文润色助手，擅长把'AI 生成/套路化的书面腔'改写成自然、像真人写的文字。\n"
        "请严格遵循下面这份'作者风格画像'来改写，目标不是'漂亮'，而是'像这个人写的'。\n\n"
        f"【作者风格画像】\n"
        f"整体语气：{p.get('tone', '平实口语')}\n"
        f"用词偏好：{vocab}\n"
        f"句式特征：{patterns}\n"
        f"段落节奏：{p.get('rhythm', '短句为主')}\n"
        f"明确避开的写法：{avoid}\n"
        f"该作者代表性原句（请贴近这种味道）：\n{examples}\n\n"
        "【改写规则】\n"
        "1. 只改'说法'，不改'意思'与立场；不替作者添加没说过的观点、数据或情绪。\n"
        "2. 删掉套路连接词（首先/其次/最后、值得注意的是、综上所述…）与空洞总结升华。\n"
        "3. 长句拆短、适当口语化，但不媚俗、不强行加网络热词。\n"
        "4. 保留信息密度，不为了'像人'而注水。\n\n"
        "【输出格式——必须严格遵守】\n"
        "只输出改写后的中文正文本身。严禁输出任何分析、说明、前后对比、markdown 标题或符号。"
        "不要写'改写说明''以下是'之类的话。直接给改写完的文字。"
    )
    user = f"请改写下面这段文字：\n\n{draft}"
    return system, user


def rule_based_rewrite(text):
    """确定性兜底：删套路前缀词/连接词、拆长句，至少去掉最明显的 AI 腔。"""
    # 1) 删常见套路前缀（含标点）
    for m in ["值得注意的是，", "值得注意的是", "综上所述，", "综上所述",
              "毫无疑问，", "毫无疑问", "在当今社会", "因此可见，",
              "总而言之，", "换言之，", "在此基础上，", "不仅如此，"]:
        text = text.replace(m, "")
    # 2) 删句首连接词（首先/其次/最后）
    text = re.sub(r"([。；]|^)首先[，,]", r"\1", text)
    text = re.sub(r"([。；]|^)其次[，,]", r"\1", text)
    text = re.sub(r"([。；]|^)最后[，,]", r"\1", text)
    # 3) 去掉"空口号句"：以"毫无疑问"残留或纯喊话收尾的短句
    text = re.sub(r"[。；]?毫无疑问[^。；]*[。；]?", "。", text)
    # 4) 清理多余空白与首尾标点
    text = re.sub(r"\s+", "", text)
    text = text.strip("，。；")
    if not text.endswith("。"):
        text += "。"
    return text


def rewrite_with_retry(client, system, user, before_score, think):
    """自检 + 重试（最多 3 次），全失败降级规则兜底。返回 (文本, 来源)。"""
    for attempt in range(1, 4):
        cand = call_model(client, system, user, think, temperature=0.3).strip()
        if not is_bad_rewrite(cand, before_score):
            return cand, f"LLM 第 {attempt} 次改写"
        print(f"  ⚠️ 第 {attempt} 次输出未通过自检（跑偏去分析/复述或 AI 味未降），重试…")
    print("  ⚠️ 三次 LLM 改写均未达预期（小模型指令跟随不稳），启用规则兜底。")
    return rule_based_rewrite(user.split("请改写下面这段文字：\n\n", 1)[-1]), "规则兜底"


# ---------------------------------------------------------------------------
# 3. 主流程
# ---------------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="实验 2-7 配套：按风格画像去 AI 味改写")
    ap.add_argument("--draft", default=os.path.join(here, "examples", "ai_draft.txt"),
                    help="待改写的草稿路径（默认 examples/ai_draft.txt）")
    ap.add_argument("--profile", default=os.path.join(here, "style_profile.json"),
                    help="风格画像路径（默认 style_profile.json）")
    ap.add_argument("--out", default=os.path.join(here, "output", "rewritten.txt"),
                    help="输出路径（默认 output/rewritten.txt）")
    ap.add_argument("--think", action="store_true", help="开启模型原生思考（默认关闭）")
    ap.add_argument("--model", help="覆盖 .env 中的 OPENAI_MODEL")
    args = ap.parse_args()

    global MODEL
    if args.model:
        MODEL = args.model

    if not os.path.exists(args.draft):
        print(f"❌ 找不到草稿：{args.draft}")
        sys.exit(1)

    with open(args.draft, encoding="utf-8") as f:
        draft = f.read().strip()

    print(f"模型：{MODEL} | 端点：{BASE_URL}")
    print(f"草稿：{args.draft}  ({len(draft)} 字)")
    print(f"画像：{args.profile}")
    print("=" * 72)

    profile = load_profile(args.profile)
    before_score, before_hits = ai_flavor_score(draft)
    print(f"改写前 AI 味指数：{before_score}（命中 {before_hits} 处关键词）")

    client = build_client()
    system, user = build_rewrite_prompt(profile, draft)
    print("[改写] 让本地 LLM 按风格画像重写（自检+重试+兜底）…")
    rewritten, source = rewrite_with_retry(client, system, user, before_score, args.think)

    after_score, after_hits = ai_flavor_score(rewritten)
    print(f"改写后 AI 味指数：{after_score}（命中 {after_hits} 处关键词）| 来源：{source}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(rewritten)
    print(f"\n✅ 改写完成，已写出：{args.out}")
    print("-" * 72)
    print("【改写前】\n" + draft)
    print("-" * 72)
    print("【改写后】\n" + rewritten)


if __name__ == "__main__":
    main()
