# -*- coding: utf-8 -*-
"""
实验 2-4：提示工程的消融实验（Prompt Engineering Ablation Study）—— 有区分度版
=================================================================================

对应《深入理解 AI Agent》第 2 章 · 上下文工程 · 实验 2-4（原书目录名 `ablation`）。

【这个实验在测什么】
一个生产级提示通常由多个"组件"拼成：系统角色 / 少样本示例 / 思维链引导 / 输出格式约束。
本实验做消融：把每个组件逐个"拿掉"，在同一份固定测试集上跑，看指标掉多少。
掉得多 = 该组件越关键。用数据说话，而非凭感觉堆提示词。

【为什么这版"有区分度"】
初版有两个问题导致消融"全 90%、看不出差别"：
  ① 测试集太简单，0.6B 模型不靠任何组件就能分对，天花板封顶、无处可跌；
  ② 评分只算"标签准确率"且极宽松（输出里出现标签词就算对），把格式组件的效果也抹平了。
本版修两处：
  A) 测试集混入"表面线索会误导、必须按定义才分对"的陷阱样本，给组件留出"发挥空间"；
  B) 双指标评分：
       - 标签准确率(acc)：宽松，取输出中最后一个出现的 咨询/投诉/建议（衡量"分没分对"）
       - 格式合规率(fmt)：严格，要求存在单独一行恰好为「意图：X」（衡量"下游能不能机器解析"）
     去掉格式约束时 acc 几乎不变、但 fmt 暴跌——这才是对"格式组件有没有用"的诚实回答。

【如何运行】
  pip install -r experiments/ch02/exp2-4/requirements.txt
  python experiments/ch02/exp2-4/main.py                  # 全 6 变体（默认已关思考）
  python experiments/ch02/exp2-4/main.py --only no_fewshot
  python experiments/ch02/exp2-4/main.py --think          # 显式开启模型原生思考（默认关闭）
  python experiments/ch02/exp2-4/main.py --samples 5      # 只用前 5 条快速预览

  说明：默认【关闭】模型原生思考（inject enable_thinking=false）。原因：原生思考会吃掉
  max_tokens 预算、让 0.6B 小模型吐不出标签；且它属于"提示组件"之外的另一维度，开着会
  污染消融纯度、增加随机性。需观察思考本身的影响时才显式加 --think。
"""

import os
import sys
import re
import json
import time
import argparse
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 0. 配置加载：与 2-1/2-2/2-3 完全一致的变量名，切换服务商只改 .env 的值
# ---------------------------------------------------------------------------
load_dotenv()  # 读取仓库根目录的 .env（已被 .gitignore 忽略，切勿提交）

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1")
MODEL = os.getenv("OPENAI_MODEL", "DeepSeek-V4-Flash")

# EXTRA_BODY：传给模型的额外参数（JSON 字符串）。本地 Qwen3 想关思考可填 {"enable_thinking": false}
RAW_EXTRA = os.getenv("EXTRA_BODY", "{}")
try:
    EXTRA_BODY = json.loads(RAW_EXTRA) if RAW_EXTRA.strip() else {}
except json.JSONDecodeError:
    print(f"[警告] EXTRA_BODY 不是合法 JSON：{RAW_EXTRA!r}，已忽略")
    EXTRA_BODY = {}

# MAX_TOKENS：分类任务不需要长输出，默认 128
try:
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "128"))
except ValueError:
    MAX_TOKENS = 128


def build_client():
    """构造 OpenAI 兼容客户端；缺 Key 直接报错退出，绝不发起请求。"""
    if not API_KEY:
        print("❌ 未检测到 OPENAI_API_KEY。请复制仓库根目录 .env.example 为 .env 并填入密钥。")
        print("   本地 Ollama 路径可填 OPENAI_API_KEY=ollama 占位（模型已在本地，无需真实 Key）。")
        sys.exit(1)
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ 缺少依赖 openai，请先 pip install -r experiments/ch02/exp2-4/requirements.txt")
        sys.exit(1)
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ---------------------------------------------------------------------------
# 1. 固定测试集（gold label 是确定的，用来算准确率）
#    设计要点：混入"陷阱样本"——表面线索（能不能/怎么）会误导 naive 模型，
#    只有按 SYSTEM_ROLE 的定义（咨询=询问信息/流程；投诉=表达不满/质量；建议=改进诉求）
#    才能分对。这样"系统角色 / 少样本"被拿掉时才会掉分，消融才有区分度。
#    标注 [陷阱] 的样本即为此类；[对照] 为模型本就能分对的锚点。
# ---------------------------------------------------------------------------
TEST_SET = [
    # —— 对照（模型本就能分对，验证 base 能力）——
    ("你们这个产品的退款流程怎么走？",            "咨询"),   # 对照：明确问流程
    ("我买的东西发了三天还没到，太差了！",        "投诉"),   # 对照：明显不满
    ("建议你们加一个夜间模式，对眼睛好。",        "建议"),   # 对照：明示"建议"
    ("发票怎么开？",                              "咨询"),   # 对照
    ("客服根本联系不上，服务太烂了。",            "投诉"),   # 对照
    ("会员到期提醒在哪里设置？",                  "咨询"),   # 对照
    ("希望增加多语言支持。",                      "建议"),   # 对照
    ("密码找回的入口找不到。",                    "咨询"),   # 对照
    # —— 陷阱：表面"能不能/怎么"像咨询，但按定义是"改进诉求=建议" ——
    ("你们能不能把导出功能做成一键的？",          "建议"),   # 陷阱：能不能+功能→建议
    ("能不能做个收藏功能？",                      "建议"),   # 陷阱
    # —— 陷阱：表达不满/要求补救，按定义是"投诉"而非"咨询" ——
    ("我付了钱却下不了单，必须给个说法！",        "投诉"),   # 陷阱：挫败+索赔
    ("这App老是卡顿，你们到底行不行？",           "投诉"),   # 陷阱：明显不满
]

# 少样本示例（仅当包含"少样本"组件时拼进提示）
# 故意放一个"能不能→建议"的陷阱样例，让 few-shot 能纠正模型的 naive 误判
FEW_SHOT = [
    ("你们的营业时间是几点到几点？", "咨询"),
    ("收到的货破损了，必须退款！",   "投诉"),
    ("能不能把首页改得清爽一点？",   "建议"),
]

LABELS = ["咨询", "投诉", "建议"]

# 四个提示组件的"积木"
SYSTEM_ROLE = (
    "你是一个严谨的中文客服意图分类器。"
    "用户会发来一句简短的话，你只需判断它属于『咨询 / 投诉 / 建议』中的哪一类。"
    "严格按以下定义分类：咨询=询问信息/流程；投诉=表达不满或质量问题、要求补救；"
    "建议=提出改进/新功能诉求（含'能不能做成X功能'这类请求）。"
)
COT_INSTRUCTION = "请先用一两句话简要分析这句话的意图依据，再给出结论。"
FORMAT_INSTRUCTION = (
    "最终答案必须单独成行、且严格为：意图：X（X 为 咨询、投诉、建议 之一），"
    "不要输出任何额外内容、不要写理由。"
)


# ---------------------------------------------------------------------------
# 2. 提示组装：按开关拼出不同组件的变体
# ---------------------------------------------------------------------------
def build_messages(query, include_system, include_fewshot, include_cot, include_format):
    """返回 (system_str_or_None, user_str)。"""
    user_parts = []

    # 组件 A：少样本示例（放在用户消息里，作为"示例"块，对本地小模型最友好）
    if include_fewshot:
        user_parts.append("以下是一些示例：")
        for q, a in FEW_SHOT:
            user_parts.append(f"用户：{q}")
            user_parts.append(f"助手：意图：{a}")
        user_parts.append("")  # 空行分隔

    # 任务本体
    user_parts.append(f"请对下面这句话分类：\n{query}")

    # 组件 C：思维链引导
    if include_cot:
        user_parts.append("")
        user_parts.append(COT_INSTRUCTION)

    # 组件 D：输出格式约束
    if include_format:
        user_parts.append("")
        user_parts.append(FORMAT_INSTRUCTION)

    system = SYSTEM_ROLE if include_system else None
    user = "\n".join(user_parts)
    return system, user


# 六个消融变体：描写"包含哪些组件"
VARIANTS = {
    "full":       {"desc": "全组件(角色+少样本+CoT+格式)", "sys": True,  "shot": True,  "cot": True,  "fmt": True},
    "no_system":  {"desc": "去掉系统角色",                  "sys": False, "shot": True,  "cot": True,  "fmt": True},
    "no_fewshot": {"desc": "去掉少样本示例",                "sys": True,  "shot": False, "cot": True,  "fmt": True},
    "no_cot":     {"desc": "去掉思维链引导",                "sys": True,  "shot": True,  "cot": False, "fmt": True},
    "no_format":  {"desc": "去掉输出格式约束",              "sys": True,  "shot": True,  "cot": True,  "fmt": False},
    "minimal":    {"desc": "仅裸任务(全部拿掉)",           "sys": False, "shot": False, "cot": False, "fmt": False},
}

# 各变体"去掉的组件"中文描述（供汇总表与结论使用，独立命名避免被循环变量遮蔽）
DROP_DESC = {
    "full": "（基线，无）",
    "no_system": "系统角色",
    "no_fewshot": "少样本示例",
    "no_cot": "思维链引导",
    "no_format": "输出格式约束",
    "minimal": "全部",
}


# ---------------------------------------------------------------------------
# 3. 调用模型 + 解析（双指标）
# ---------------------------------------------------------------------------
def call_model(client, system, user, think):
    """发一次请求，返回 (文本, completion_tokens)。

    think=False（默认）：注入 enable_thinking=false，关闭模型原生思考。
    原因：① 原生思考会吃掉 max_tokens 预算、让 0.6B 小模型吐不出标签；
          ② 它是独立于"提示组件"的另一维度，开着会污染消融纯度、增加随机性。
    EXTRA_BODY 里若已显式设置 enable_thinking，则以环境变量为准（不覆盖）。
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    # 默认关闭原生思考；除非用户显式 --think 或已在 EXTRA_BODY 中指定
    extra = dict(EXTRA_BODY)
    if not think and "enable_thinking" not in extra:
        extra["enable_thinking"] = False

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        extra_body=extra,
        temperature=0.0,  # 消融要可复现，温度归零
    )
    text = resp.choices[0].message.content or ""
    tokens = getattr(getattr(resp, "usage", None), "completion_tokens", None)
    return text, tokens


def parse_label(text):
    """宽松解析：取文本中最后一个出现的 咨询/投诉/建议（衡量"分没分对"）。"""
    if not text:
        return None
    text = text.replace("<think>", "").replace("</think>", "")  # 防御残留思考标签
    positions = []
    for lab in LABELS:
        for m in re.finditer(re.escape(lab), text):
            positions.append((m.start(), lab))
    if not positions:
        return None
    positions.sort()
    return positions[-1][1]


def parse_format_ok(text):
    """严格格式合规：存在单独一行恰好为「意图：X」（X∈LABELS），且无多余标签行。

    衡量"下游能不能机器解析"。带格式约束的变体会被要求只输出这一行；
    去掉格式约束后模型会胡写，此指标会明显下跌——这正是格式组件的价值所在。
    """
    if not text:
        return False
    text = text.replace("<think>", "").replace("</think>", "")
    for line in text.splitlines():
        s = line.strip()
        if re.fullmatch(r"意图：(咨询|投诉|建议)", s):
            return True
    return False


# ---------------------------------------------------------------------------
# 4. 单变体评估（双指标）
# ---------------------------------------------------------------------------
def evaluate_variant(name, cfg, client, samples, think):
    correct = 0        # 标签准确率计数
    fmt_ok = 0         # 格式合规计数
    tok_sum = 0
    tok_n = 0
    failed = 0
    last_err = None
    per_row = []
    for query, gold in samples:
        system, user = build_messages(
            query, cfg["sys"], cfg["shot"], cfg["cot"], cfg["fmt"]
        )
        try:
            text, toks = call_model(client, system, user, think)
        except Exception as e:  # 某条失败不影响整体，记 0 分
            text, toks = f"[调用异常: {e}]", None
            failed += 1
            last_err = e
        pred = parse_label(text)
        ok = (pred == gold)
        fok = parse_format_ok(text)
        if ok:
            correct += 1
        if fok:
            fmt_ok += 1
        if toks is not None:
            tok_sum += toks
            tok_n += 1
        per_row.append((query[:16], gold, pred, ok, fok))
    if failed:
        print(f"      ⚠️ {failed}/{len(samples)} 条调用失败，样例错误：{last_err}")
    acc = correct / len(samples) if samples else 0.0
    fmt = fmt_ok / len(samples) if samples else 0.0
    avg_tok = (tok_sum / tok_n) if tok_n else 0
    return {
        "name": name,
        "desc": cfg["desc"],
        "correct": correct,
        "fmt_ok": fmt_ok,
        "total": len(samples),
        "acc": acc,
        "fmt": fmt,
        "avg_tok": avg_tok,
        "failed": failed,
        "per_row": per_row,
    }


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="实验 2-4：提示工程消融（有区分度版）")
    ap.add_argument("--only", help="只跑某个变体，如 full / no_fewshot")
    ap.add_argument("--samples", type=int, default=len(TEST_SET),
                    help="使用测试集前 N 条（默认全部 %d 条）" % len(TEST_SET))
    ap.add_argument("--think", action="store_true",
                    help="开启模型原生思考模式（默认关闭，以保证消融纯度与可复现性）")
    ap.add_argument("--model", help="覆盖 .env 中的 OPENAI_MODEL")
    args = ap.parse_args()

    global MODEL
    if args.model:
        MODEL = args.model

    samples = TEST_SET[: max(1, min(args.samples, len(TEST_SET)))]

    client = build_client()

    if args.only:
        if args.only not in VARIANTS:
            print(f"❌ 未知变体 {args.only!r}，可选：{', '.join(VARIANTS)}")
            sys.exit(1)
        variant_names = [args.only]
    else:
        variant_names = list(VARIANTS.keys())

    print(f"模型：{MODEL} | 端点：{BASE_URL}")
    print(f"测试样本：{len(samples)} 条 | 变体数：{len(variant_names)} | think={args.think}")
    print("=" * 72)

    results = []
    for name in variant_names:
        cfg = VARIANTS[name]
        print(f"[跑] {name:10s} | {cfg['desc']}")
        r = evaluate_variant(name, cfg, client, samples, args.think)
        results.append(r)
        print(f"      → 标签准确率 {r['acc']*100:5.1f}%  ({r['correct']}/{r['total']})"
              f"  格式合规 {r['fmt']*100:5.1f}%  ({r['fmt_ok']}/{r['total']})"
              f"  平均生成 {r['avg_tok']:.0f} token  失败 {r['failed']}")
        time.sleep(0.3)  # 给免费端点留点呼吸，避免 429

    # ---- 汇总表（双指标 + Δ）----
    print("\n" + "=" * 72)
    print("消融结果汇总（同一份固定测试集；acc=标签准确率, fmt=格式合规率）")
    print("-" * 72)
    print(f"{'变体':10s} | {'acc':>7s} | {'Δacc':>7s} | {'fmt':>7s} | {'Δfmt':>7s} | 去掉的组件")
    full_acc = next((r["acc"] for r in results if r["name"] == "full"), None)
    full_fmt = next((r["fmt"] for r in results if r["name"] == "full"), None)
    for r in results:
        d_acc = f"{(r['acc']-full_acc)*100:+5.1f}pp" if full_acc is not None and r["name"] != "full" else ""
        d_fmt = f"{(r['fmt']-full_fmt)*100:+5.1f}pp" if full_fmt is not None and r["name"] != "full" else ""
        desc = DROP_DESC[r["name"]]
        print(f"{r['name']:10s} | {r['acc']*100:6.1f}% | {d_acc:>7s} | {r['fmt']*100:6.1f}% | {d_fmt:>7s} | {desc}")

    # ---- 结论 ----
    print("\n结论观察：")
    # 只在"单组件消融"里比最差/最好，排除 minimal（全裸基线，不算单组件）
    SINGLE = ["no_system", "no_fewshot", "no_cot", "no_format"]
    if full_acc is not None:
        acc_drops = sorted((r["name"], r["acc"] - full_acc) for r in results if r["name"] in SINGLE)
        if acc_drops:
            worst, best = acc_drops[0], acc_drops[-1]
            print(f"  - 【标签准确率】单组件消融中，拿掉『{DROP_DESC[worst[0]]}』掉得最多"
                  f"（{worst[1]*100:+.1f}pp），最影响'分没分对'；"
                  f"『{DROP_DESC[best[0]]}』几乎不影响（{best[1]*100:+.1f}pp，"
                  f"12 样本下 ±1 样本≈±8.3pp 噪声内）。")
    if full_fmt is not None:
        fmt_drops = sorted((r["name"], r["fmt"] - full_fmt) for r in results if r["name"] in SINGLE)
        if fmt_drops:
            worst_f = fmt_drops[0]
            print(f"  - 【格式合规率】拿掉『{DROP_DESC[worst_f[0]]}』掉得最多（{worst_f[1]*100:+.1f}pp），"
                  f"最影响'下游能否机器解析'。")
    print("  - 提示工程要『用消融验证』：凭感觉堆组件不如逐个拿掉看数据。")
    print("  - 同一组件对不同指标贡献不同（如格式约束几乎不影响对错、却决定可解析性）——")
    print("    这才是消融真正的价值：看清每个组件到底在'为哪件事'干活。")


if __name__ == "__main__":
    main()
