# -*- coding: utf-8 -*-
"""
实验 2-7（配套脚本一）：从个人范文抽取「风格画像」 —— build_skill.py
================================================================================
对应《深入理解 AI Agent》第 2 章 · Agent Skills · 实验 2-7
（原书正文实验：从个人范文创建"去 AI 味"写作 Skill；本脚本把"范文 → 风格画像"自动化）。

【这个脚本在做什么】
Skill 的本质是一份「说明书」(skill.md) + 背后的资源。要让"去 AI 味"真正贴合
某个人的文风，光靠手写规则不够，得先弄清楚这个人到底"怎么写"。本脚本就是把
几篇个人范文喂给本地 LLM，让它抽出可复用的风格要素，固化成 `style_profile.json`：

    个人范文(若干 .txt) ──▶ 本地 LLM 风格抽取 ──▶ style_profile.json

style_profile.json 随后被 rewrite.py 当作"改写准则"读取，保证每次改写都对齐
该作者的真实语气 / 用词 / 句式 / 节奏 / 避坑点。

【为什么不直接手写规则】
- 人的文风是模糊的、多面的，手写容易漏；让模型从真实样本里归纳更稳。
- 抽成结构化 JSON 后，Skill 就能"数据驱动"地迭代：补范文 → 重跑 → 画像更新。

【如何运行】
  pip install -r experiments/ch02/exp2-7/requirements.txt
  python experiments/ch02/exp2-7/build_skill.py                 # 用 examples/personal_samples 默认范文
  python experiments/ch02/exp2-7/build_skill.py --samples 目录   # 指定自己的范文目录
  python experiments/ch02/exp2-7/build_skill.py --samples 目录 --model qwen3:0.6b

说明：默认【关闭】模型原生思考（向 miniCPM/Qwen3 注入 enable_thinking:false），
理由同 2-4~2-6——原生思考会吃掉 token 预算、增加随机性、污染对照纯度。
"""

import os
import sys
import json
import argparse
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 0. 配置加载：与 2-1~2-6 完全一致的变量名，切换服务商只改 .env 的值
# ---------------------------------------------------------------------------
load_dotenv()  # 读取仓库根目录的 .env（已被 .gitignore 忽略，切勿提交）

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
MODEL = os.getenv("OPENAI_MODEL", "minicpm-32k")

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


# ---------------------------------------------------------------------------
# 1. 客户端与模型调用
# ---------------------------------------------------------------------------
def build_client():
    """构造 OpenAI 兼容客户端；缺 Key 直接报错退出，绝不发起请求。"""
    if not API_KEY:
        print("❌ 未检测到 OPENAI_API_KEY。请复制仓库根目录 .env.example 为 .env 并填入密钥。")
        print("   本地 Ollama 路径可填 OPENAI_API_KEY=ollama 占位（模型已在本地，无需真实 Key）。")
        sys.exit(1)
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ 缺少依赖 openai，请先 pip install -r experiments/ch02/exp2-7/requirements.txt")
        sys.exit(1)
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)


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
    """从模型输出里稳健地抠出第一个 {...} JSON 块（0.6B/2.5B 常在 JSON 外裹废话）。"""
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
# 2. 读取范文
# ---------------------------------------------------------------------------
def read_samples(samples_dir):
    """读取目录下所有 .txt 范文（UTF-8），返回 [(文件名, 文本)]。"""
    if not os.path.isdir(samples_dir):
        print(f"❌ 找不到范文目录：{samples_dir}")
        sys.exit(1)
    files = sorted(f for f in os.listdir(samples_dir) if f.endswith(".txt"))
    if not files:
        print(f"❌ 范文目录里没有任何 .txt：{samples_dir}")
        sys.exit(1)
    pairs = []
    for f in files:
        with open(os.path.join(samples_dir, f), encoding="utf-8") as fh:
            pairs.append((f, fh.read().strip()))
    return pairs


# ---------------------------------------------------------------------------
# 3. 风格抽取（核心）
# ---------------------------------------------------------------------------
STYLE_SYSTEM = (
    "你是一位严谨的中文写作风格分析师。我会给你几段同一个人的真实随笔/公众号文字。"
    "请只依据这些样本，归纳出此人'怎么写'的可复用风格画像，不要脑补样本里没有的特征。"
    "必须严格输出 JSON（不要 markdown 代码围栏，不要任何额外解释），结构如下：\n"
    "{\n"
    "  \"tone\": \"用一两句话概括整体语气（如：平实、自嘲、娓娓道来）\",\n"
    "  \"vocabulary\": [\"该作者偏好的用词/口头禅，列 3~6 条\"],\n"
    "  \"sentence_patterns\": [\"典型的句式特征，列 3~6 条，如：多用短句、爱用逗号断句\"],\n"
    "  \"rhythm\": \"对段落节奏与停顿的描述\",\n"
    "  \"avoid\": [\"此人写作时明显避开的东西，如：不堆砌成语、不用首先其次最后\"],\n"
    "  \"examples\": [\"从样本里挑 2~3 句最具代表性的原句，原样引用\"]\n"
    "}"
)

STYLE_USER_TMPL = (
    "以下是同一个人的 {n} 段真实文字，请按系统要求归纳风格画像（只输出 JSON）：\n\n"
    "{samples}"
)


def build_profile(client, pairs, think):
    """调用 LLM 抽取风格画像；失败则用启发式兜底。"""
    samples_block = "\n\n".join(f"【{name}】\n{text}" for name, text in pairs)
    user = STYLE_USER_TMPL.replace("{n}", str(len(pairs))).replace("{samples}", samples_block)
    raw = call_model(client, STYLE_SYSTEM, user, think)
    data = extract_first_json(raw)
    if not isinstance(data, dict):
        print("  ⚠️ 风格抽取未返回合法 JSON，改用启发式兜底画像。")
        return _heuristic_profile(pairs)
    # 兜底字段，保证后续 rewrite 不会 KeyError
    data.setdefault("tone", "平实、口语化")
    data.setdefault("vocabulary", [])
    data.setdefault("sentence_patterns", [])
    data.setdefault("rhythm", "短句为主，逗号断句")
    data.setdefault("avoid", ["不用首先/其次/最后", "不用值得注意的是/综上所述"])
    data.setdefault("examples", [t for _, t in pairs[:2]])
    return data


def _heuristic_profile(pairs):
    """解析失败时的兜底：用极简规则包住原样本，至少保证 examples 可用。"""
    return {
        "tone": "平实、口语、第一人称",
        "vocabulary": [],
        "sentence_patterns": ["短句为主", "逗号/句号断句", "少用连接词"],
        "rhythm": "一句一停，节奏松弛",
        "avoid": ["不用首先/其次/最后", "不用值得注意的是/综上所述", "不堆砌成语"],
        "examples": [t for _, t in pairs[:3]],
    }


# ---------------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="实验 2-7 配套：从个人范文抽取风格画像")
    ap.add_argument("--samples", default=os.path.join(here, "examples", "personal_samples"),
                    help="范文目录（默认 examples/personal_samples）")
    ap.add_argument("--out", default=os.path.join(here, "style_profile.json"),
                    help="输出画像路径（默认 style_profile.json）")
    ap.add_argument("--think", action="store_true", help="开启模型原生思考（默认关闭）")
    ap.add_argument("--model", help="覆盖 .env 中的 OPENAI_MODEL")
    args = ap.parse_args()

    global MODEL
    if args.model:
        MODEL = args.model

    print(f"模型：{MODEL} | 端点：{BASE_URL}")
    print(f"范文目录：{args.samples}")
    print("=" * 72)

    pairs = read_samples(args.samples)
    print(f"读取范文 {len(pairs)} 篇：")
    for name, text in pairs:
        print(f"  - {name}  ({len(text)} 字)")

    client = build_client()
    print("\n[抽取] 让本地 LLM 归纳风格画像 ...")
    profile = build_profile(client, pairs, args.think)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 风格画像已写出：{args.out}")
    print("--- 画像摘要 ---")
    print(f"语气：{profile.get('tone')}")
    print(f"节奏：{profile.get('rhythm')}")
    print(f"用词偏好：{profile.get('vocabulary')}")
    print(f"句式特征：{profile.get('sentence_patterns')}")
    print(f"避坑点：{profile.get('avoid')}")
    print("\n下一步：把任意 AI 草稿交给 rewrite.py，即可按此画像去 AI 味。")


if __name__ == "__main__":
    main()
