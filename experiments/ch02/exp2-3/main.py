"""
实验 2-3（修订版）：常见的错误上下文管理模式（KV Cache 前缀复用视角）
================================================================
本实验用「OpenAI 兼容端点」演示（推荐本地 Ollama，也支持 AMD 等云端 API）：
同一个 Agent 上下文里，哪些写法会让 KV Cache 前缀复用失效、TTFT 变慢。

【相对原版的修订要点】
1. dynamic_system / dynamic_profile：把变化点提前到 system 最开头，并放大差异
   → 确保在 token 级前缀匹配下真的能观测到 cached_tokens 下降。
2. shuffled_tools：把工具描述序列化进 system message，让顺序变化进入 messages
   → 原版只改 tools 参数，messages 完全没变，所以"失效"不成立。
3. 新增 totally_different 负对照：完全不同的 system + 历史 + 问题
   → 如果它仍返回 cached 恒为同一高值（与内容无关），说明服务端缓存粒度是固定块。
4. 新增 shared_prefix_chars：打印 cold/warm 的最长公共前缀（字符级）
   → 与 cached_tokens 对照，判断服务端缓存是 token 级还是块级。
5. sliding_window 拆成 drop_oldest / append_new，单一变量，便于归因。

【核心机制（与模型参数规模无关，是推理服务的机制）】
- 大模型处理输入时，会为「前缀 token」计算并缓存 KV Cache。
- 如果两次请求共享「完全相同的稳定前缀」，第二次可直接复用缓存
  → 省掉前缀重算 → TTFT（首 token 延迟）下降。
- 一旦前缀在某处变了（时间戳、工具乱序、丢历史、压成纯文本……），
  缓存就失效，第二次仍要全量重算。

【为什么用 OpenAI 兼容端点（本地 Ollama / 云端 API 皆可）？】
- 本地 Ollama（如 Qwen3-0.6B）经其 OpenAI 兼容端点调用时，会返回
  cached_tokens（usage.prompt_tokens_details.cached_tokens）——直接告诉我们
  这次请求的 prompt 有多少 token 是从 KV 缓存里复用的，信号干净、免费、无配额。
- 云端 API（如 AMD Radeon Cloud · Token Factory）同样返回 cached_tokens，
  适合没有本地 GPU 的场景；它默认带块级缓存，细节见 README「〇、本地 Ollama 跑法」。
- 注意 Qwen3 在 Ollama 上默认开启 thinking，思考 token 会占用 max_tokens 预算；
  本脚本已把预算抬高，保证思考结束后仍产出可见内容。判定一律以 cached_tokens 为主信号，
  TTFT 受思考/排队噪声影响，仅作旁证。

【运行前准备】
1. 在 .env（已 gitignore，勿提交）填入 OpenAI 兼容端点的连接信息，例如本地 Ollama：
     OPENAI_API_KEY=ollama
     OPENAI_BASE_URL=http://localhost:11434/v1
     OPENAI_MODEL=qwen3:0.6b
   或 AMD 云端：
     OPENAI_API_KEY=rc-你的AMD令牌
     OPENAI_BASE_URL=https://developer.amd.com.cn/radeon/api/v1
     OPENAI_MODEL=DeepSeek-V4-Flash
2. 安装依赖：pip install -r requirements.txt

【运行】
  python main.py                      # 跑全部场景
  python main.py --only correct       # 只跑某个场景
  python main.py --only totally_different
  python main.py --warm-repeats 3     # 热请求重复次数（取最快）
"""

import argparse
import json
import os
import random
import re
import time
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ───────────────────────────── 配置（全部来自 .env，可覆盖） ─────────────────────────────
API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("AMD_API_KEY")
    or os.getenv("RADEON_API_KEY")
    or os.getenv("MODELSCOPE_API_KEY")
)
BASE_URL = (
    os.getenv("OPENAI_BASE_URL")
    or os.getenv("AMD_BASE_URL")
    or os.getenv("LLM_BASE_URL")
    or os.getenv("MODELSCOPE_BASE_URL")
    or "https://developer.amd.com.cn/radeon/api/v1"
)
MODEL_NAME = os.getenv("OPENAI_MODEL") or os.getenv("MODEL_NAME", "DeepSeek-V4-Flash")
EXTRA_BODY = json.loads(os.getenv("EXTRA_BODY", "{}"))
# Qwen3 在 Ollama 上默认开启 thinking，思考 token 走 delta.reasoning 字段、SDK 不暴露、
# 且无法用 extra_body 关掉；思考会占用 max_tokens 预算，预算太小则 content 被饿掉、TTFT 变 None。
# 这里把上限抬高到 256，保证思考结束后仍产出可见 content；cached_tokens 仍是主信号。
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

# 判定「真实前缀缓存命中」的阈值（仅用于服务商不返回 cached_tokens 时的 TTFT 回退）
HIT_FLOOR = 2.0
WARMUP_FLOOR = 1.15


# ───────────────────────────── 共享上下文（希望被缓存的「稳定前缀」） ─────────────────────────────
SYSTEM_PROMPT = """你是一个高效、严谨的个人智能助理，服务于一位经常出差的产品经理。
你的职责是：根据用户的自然语言请求，决定调用哪些工具来完成任务，并在没有工具能满足时直接给出简洁回答。
工作准则：
1. 优先使用已有工具完成可结构化的操作（查天气、检索知识库、建日历、发邮件、查订单）。
2. 不要编造工具不存在的能力；当请求超出工具范围时，明确说明并给出建议。
3. 输出简洁、面向行动；除非用户要求，不要长篇大论。
4. 所有时间相关判断以用户当前所在时区为准，并在涉及时间时显式标注日期。
5. 保护用户隐私：涉及订单、邮件等敏感操作时，先确认关键信息再执行。"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市在某一天的天气情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如 北京"},
                    "date": {"type": "string", "description": "日期，如 2026-09-18"},
                },
                "required": ["city", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "在个人知识库中检索与查询相关的文档片段",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "在日历中创建一个事件",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                },
                "required": ["title", "start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "发送一封邮件",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_order_status",
            "description": "按订单号查询物流与状态",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                },
                "required": ["order_id"],
            },
        },
    },
]

# 早期对话历史（希望它稳定地待在前缀里被缓存）
HISTORY = [
    ("帮我查一下明天北京的天气", "已为你查询：北京明天晴，18~27℃，适宜出行。需要我帮你安排行程吗？"),
    ("把明天上午 10 点的团队站会加到日历", "已创建日历事件「团队站会」，时间 2026-09-18 10:00-10:30。"),
    ("我们知识库里有没有关于 KV Cache 的笔记", "检索到 3 篇相关笔记，主题涵盖前缀缓存、注意力机制与推理优化。"),
    ("上次那个订单到哪了", "订单 A20260912 已发货，预计 2026-09-19 送达。"),
]

CURRENT_QUESTION = "北京后天天气怎么样？顺便把后天下午 3 点和客户的评审会加到日历。"

# 每次运行生成一个唯一标签，拼到每个场景的系统提示最前面。
# 作用：① 隔离各场景之间的前缀缓存，避免互相污染；
#       ② 同一场景内 cold/warm 共享同一标签，保证 warm 能复用 cold 建立的前缀缓存。
RUN_TAG = ""


def scenario_tag(name):
    return f"[实验场景：{name}｜运行ID：{datetime.now().strftime('%Y%m%d%H%M%S%f')}]\n"


# ───────────────────────────── 消息组装辅助 ─────────────────────────────
def assemble(system_text, history, question, tools=None):
    """把系统提示 + 历史 + 当前问题拼成结构化 messages。"""
    messages = [{"role": "system", "content": system_text}]
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question})
    return messages, tools


def fmt_history(history):
    """把历史压成纯文本（供 text_format 场景使用）。"""
    lines = []
    for u, a in history:
        lines.append(f"用户：{u}")
        lines.append(f"助手：{a}")
    return "\n".join(lines)


def tools_as_text(tools):
    """把工具 schema 序列化成稳定文本，便于塞进 system message。"""
    return json.dumps(tools, ensure_ascii=False, indent=2)


# ───────────────────────────── 场景构造器 ─────────────────────────────
# 每个构造器返回 (cold_messages, warm_messages, cold_tools, warm_tools)
# cold = 第一次请求；warm = 第二次请求（希望命中缓存）

def build_correct():
    """正确：前缀完全稳定，仅末尾问题变化。"""
    cold = assemble(RUN_TAG + SYSTEM_PROMPT, HISTORY, CURRENT_QUESTION, TOOLS)
    warm = assemble(RUN_TAG + SYSTEM_PROMPT, HISTORY, CURRENT_QUESTION, TOOLS)
    return cold[0], warm[0], cold[1], warm[1]


def build_dynamic_system():
    """错误①（修订）：系统提示最前塞时间戳，且日期跨度大 → 第 0 个 token 就分叉。"""
    sys_cold = RUN_TAG + "[系统时间：2025-01-01 08:00:00]\n" + SYSTEM_PROMPT
    sys_warm = RUN_TAG + "[系统时间：2026-09-18 23:59:59]\n" + SYSTEM_PROMPT
    cold = assemble(sys_cold, HISTORY, CURRENT_QUESTION, TOOLS)
    warm = assemble(sys_warm, HISTORY, CURRENT_QUESTION, TOOLS)
    return cold[0], warm[0], cold[1], warm[1]


def build_shuffled_tools():
    """错误②（修订）：工具描述序列化进 system message，再打乱顺序 → 变化进入 messages。"""
    cold_tools = list(TOOLS)
    warm_tools = list(reversed(TOOLS))
    sys_cold = RUN_TAG + SYSTEM_PROMPT + "\n\n[可用工具]\n" + tools_as_text(cold_tools)
    sys_warm = RUN_TAG + SYSTEM_PROMPT + "\n\n[可用工具]\n" + tools_as_text(warm_tools)
    cold = assemble(sys_cold, HISTORY, CURRENT_QUESTION, None)
    warm = assemble(sys_warm, HISTORY, CURRENT_QUESTION, None)
    return cold[0], warm[0], cold[1], warm[1]


def build_dynamic_profile():
    """错误③（修订）：把变化的用户状态塞进 system 最开头 → 分叉点提前。"""
    prof_cold = "[用户档案] 当前积分：1280；登录城市：成都；会员等级：黄金\n"
    prof_warm = "[用户档案] 当前积分：9999；登录城市：北京；会员等级：钻石\n"
    cold = assemble(RUN_TAG + prof_cold + SYSTEM_PROMPT, HISTORY, CURRENT_QUESTION, TOOLS)
    warm = assemble(RUN_TAG + prof_warm + SYSTEM_PROMPT, HISTORY, CURRENT_QUESTION, TOOLS)
    return cold[0], warm[0], cold[1], warm[1]


def build_drop_oldest():
    """错误④-a：只丢弃最早一轮历史 → 历史块整体改写。"""
    cold_history = HISTORY
    warm_history = HISTORY[1:]
    cold = assemble(RUN_TAG + SYSTEM_PROMPT, cold_history, CURRENT_QUESTION, TOOLS)
    warm = assemble(RUN_TAG + SYSTEM_PROMPT, warm_history, CURRENT_QUESTION, TOOLS)
    return cold[0], warm[0], cold[1], warm[1]


def build_append_new():
    """错误④-b：只在末尾追加新一轮 → 历史前半段保持，变化点在末尾。"""
    extra_turn = ("后天的评审会主题定了吗", "主题暂定为「Q3 交付复盘与风险评审」，待你确认。")
    cold_history = HISTORY
    warm_history = HISTORY + [extra_turn]
    cold = assemble(RUN_TAG + SYSTEM_PROMPT, cold_history, CURRENT_QUESTION, TOOLS)
    warm = assemble(RUN_TAG + SYSTEM_PROMPT, warm_history, CURRENT_QUESTION, TOOLS)
    return cold[0], warm[0], cold[1], warm[1]


def build_totally_different():
    """负对照：cold/warm 从 system 第一个字符就不同 → 理论上应 cached≈0。
    若它也返回 cached 恒为同一高值（与内容无关），说明服务端缓存粒度是固定块。"""
    sys_cold = RUN_TAG + SYSTEM_PROMPT
    sys_warm = RUN_TAG + "你是一位诗人，只用四字短句回复。"
    cold = assemble(sys_cold, HISTORY, CURRENT_QUESTION, TOOLS)
    warm = assemble(sys_warm, [], "写一首关于秋天的诗。", None)
    return cold[0], warm[0], cold[1], warm[1]


def build_text_format():
    """错误⑤（修订）：历史压成一段纯文本后，把『最新一轮』插到文本最前（newest-first）。
    扁平化丢弃了结构化 messages 的顺序锚点，一旦重排，头部就改写 → 前缀从第 0 个 token 分叉 →
    缓存失效。这正是『把历史压成纯文本而非结构化消息』的真实风险：结构化消息的顺序天然稳定，
    压成纯文本后再重排，易变的最新轮次被顶到最前，直接击穿缓存。"""
    extra_turn = ("后天的评审会主题定了吗", "主题暂定为「Q3 交付复盘与风险评审」，待你确认。")
    q2 = "北京后天天气怎么样？顺便把后天下午 3 点和客户的评审会加到日历。"
    # cold：扁平化、oldest-first（最新轮在末尾，前缀头稳定）
    cold_text = RUN_TAG + fmt_history(HISTORY) + f"\n\n系统：{SYSTEM_PROMPT}\n问题：{CURRENT_QUESTION}"
    # warm：扁平化、但把『最新一轮』插到最前（newest-first）→ 头部改写，前缀从第 0 个 token 分叉
    warm_text = RUN_TAG + fmt_history([extra_turn] + HISTORY) + f"\n\n系统：{SYSTEM_PROMPT}\n问题：{q2}"
    cold_messages = [{"role": "user", "content": cold_text}]
    warm_messages = [{"role": "user", "content": warm_text}]
    return cold_messages, warm_messages, None, None


SCENARIOS = {
    "correct":           (build_correct,           "稳定前缀：系统提示+工具+历史全部固定 → 应命中缓存"),
    "dynamic_system":    (build_dynamic_system,    "系统提示最前塞时间戳（跨度大）→ 前缀从第 0 个 token 分叉"),
    "shuffled_tools":    (build_shuffled_tools,    "工具描述进 system 并打乱 → 前缀分叉"),
    "dynamic_profile":   (build_dynamic_profile,   "用户状态塞进 system 最前 → 前缀分叉"),
    "drop_oldest":       (build_drop_oldest,       "丢弃最早一轮历史 → 历史块改写"),
    "append_new":        (build_append_new,        "仅在末尾追加新一轮 → 前半段仍可复用"),
    "totally_different": (build_totally_different, "负对照：完全不同的 system/历史/问题 → 应 cached≈0"),
    "text_format":       (build_text_format,       "历史压成纯文本且最新轮插到最前 → 头部改写 → 失效"),
}


# ───────────────────────────── 度量 ─────────────────────────────
def timed_call(client, model, messages, tools, extra_body, max_tokens):
    """发起一次流式调用，返回 (TTFT_ms, usage, 文本)。"""
    kwargs = dict(
        model=model,
        messages=messages,
        stream=True,
        max_tokens=max_tokens,
        stream_options={"include_usage": True},
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if extra_body:
        kwargs["extra_body"] = extra_body

    start = time.perf_counter()
    first_ts = None
    usage = None
    text = ""
    with client.chat.completions.create(**kwargs) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            # 首 token 判定：content / tool_calls / function_call 都算"有产出"；
            # 额外把 reasoning_content 也算上（防御：某些端点思考 token 走此字段），
            # 避免首 token 永远抓不到、TTFT 变成 None。最终只把 content 累加进 text。
            _has_out = (
                delta
                and (
                    delta.content
                    or delta.tool_calls
                    or delta.function_call
                    or getattr(delta, "reasoning_content", None)
                )
            )
            if _has_out:
                if first_ts is None:
                    first_ts = time.perf_counter()
                if delta.content:
                    text += delta.content
            if getattr(chunk, "usage", None):
                usage = chunk.usage
    ttft_ms = (first_ts - start) * 1000 if first_ts else None
    return ttft_ms, usage, text


def cached_tokens_of(usage):
    """从 usage 里读取 cached_tokens（服务商支持才有）。"""
    if not usage:
        return None
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return None
    return getattr(details, "cached_tokens", None)


def prompt_tokens_of(usage):
    """从 usage 里读取总 prompt token 数。"""
    if not usage:
        return None
    return getattr(usage, "prompt_tokens", None)


def shared_prefix_chars(msgs_a, tools_a, msgs_b, tools_b):
    """cold / warm 序列化后的最长公共前缀（字符级），并返回两侧总长。

    注意：这是字符级近似。真实的 token 级公共前缀可能比这个略短或略长，
    但用来判断「服务端缓存的粒度是否和内容相关」已经足够。
    """
    def ser(msgs, tools):
        parts = []
        for m in msgs:
            parts.append(f"<{m['role']}>")
            parts.append(str(m.get("content") or ""))
        if tools:
            parts.append("<tools>")
            parts.append(json.dumps(tools, ensure_ascii=False))
        return "".join(parts)

    a = ser(msgs_a, tools_a)
    b = ser(msgs_b, tools_b)
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n, len(a), len(b)


def classify_speedup(speedup, correct_ref):
    """把加速比翻译成可读结论（仅作 TTFT 回退）。"""
    if speedup is None:
        return "❌ 无法测量 TTFT"
    if speedup >= HIT_FLOOR and (correct_ref is None or speedup >= 0.5 * correct_ref):
        return "✅ 真实前缀缓存命中（前缀被复用，热 >> 冷）"
    if speedup >= WARMUP_FLOOR:
        return "⚠️ 仅预热加速，前缀未复用（缓存失效）"
    return "❌ 热未见提速/更慢（无前缀缓存）"


def verdict_from_cache(warm_cached, cold_prompt, speedup, correct_ref):
    """以 cached_tokens 为主信号判定前缀缓存是否命中。"""
    if warm_cached is None:
        return classify_speedup(speedup, correct_ref)
    ratio = (warm_cached / cold_prompt) if (cold_prompt and cold_prompt > 0) else (1.0 if warm_cached >= 100 else 0.0)
    if ratio >= 0.8 and warm_cached >= 50:
        return "✅ 前缀被完整复用（命中缓存，热请求跳过整段前缀 prefill）"
    if warm_cached >= 50:
        return "⚠️ 前缀部分复用（变化点之后失效，缓存部分命中）"
    return "❌ 前缀未复用（缓存失效：前缀被改动/重塑）"


def run_scenario(name, client, model, extra_body, max_tokens, warm_repeats, correct_ref=None):
    global RUN_TAG
    RUN_TAG = scenario_tag(name)
    builder, note = SCENARIOS[name]
    cold_messages, warm_messages, cold_tools, warm_tools = builder()

    # 理论最长公共前缀（字符级）
    sp_chars, cold_chars, warm_chars = shared_prefix_chars(
        cold_messages, cold_tools, warm_messages, warm_tools
    )

    cold_ttft, cold_usage, _ = timed_call(
        client, model, cold_messages, cold_tools, extra_body, max_tokens
    )
    warm_ttfts = []
    warm_usage = None
    for _ in range(warm_repeats):
        t, u, _ = timed_call(
            client, model, warm_messages, warm_tools, extra_body, max_tokens
        )
        warm_ttfts.append(t)
        # 只用「第一次」热请求的 usage 作为 warm_cached：它测量的是 cold→warm 的前缀复用。
        # 后续热请求彼此 prompt 完全相同，会复用首次热请求的 KV，其 cached 接近整个 prompt，
        # 不能反映 cold→warm 复用，故不再覆盖。
        if warm_usage is None:
            warm_usage = u
    # 容错：个别调用的 TTFT 可能因思考预算不足而抓不到（None），不计入 min。
    warm_ttfts_valid = [t for t in warm_ttfts if t is not None]
    warm_ttft = min(warm_ttfts_valid) if warm_ttfts_valid else None
    warm_avg = sum(warm_ttfts_valid) / len(warm_ttfts_valid) if warm_ttfts_valid else None

    speedup = (cold_ttft / warm_ttft) if (cold_ttft and warm_ttft and warm_ttft > 0) else None
    cold_cached = cached_tokens_of(cold_usage)
    warm_cached = cached_tokens_of(warm_usage)
    cold_pt = prompt_tokens_of(cold_usage)
    warm_pt = prompt_tokens_of(warm_usage)

    verdict = verdict_from_cache(warm_cached, cold_pt, speedup, correct_ref)

    _ms = lambda v: f"{v:8.1f}" if v is not None else f"{'—':>8}"
    print(f"\n🧪 场景：{name}  —— {note}")
    print(f"   共享前缀(字符) : {sp_chars}   (cold {cold_chars} chars, warm {warm_chars} chars)")
    print(f"   冷启动 TTFT    : {_ms(cold_ttft)} ms   prompt={cold_pt}   cached={cold_cached}")
    print(f"   热请求 TTFT    : {_ms(warm_ttft)} ms (min of {warm_repeats})   prompt={warm_pt}   cached={warm_cached}")
    print(f"   判定           : {verdict}")
    if speedup:
        print(f"   （TTFT 加速比参考：{speedup:6.2f}x；免费端点噪声大，仅作旁证）")
    return {
        "name": name,
        "cold_ttft": cold_ttft,
        "warm_ttft": warm_ttft,
        "warm_avg": warm_avg,
        "speedup": speedup,
        "cold_cached": cold_cached,
        "warm_cached": warm_cached,
        "shared_chars": sp_chars,
        "cold_chars": cold_chars,
        "verdict": verdict,
        "mark": "✅" if verdict.startswith("✅") else ("⚠️" if verdict.startswith("⚠️") else "❌"),
    }


def resolve_model(client, requested):
    """把「控制台显示名」对齐成 API 真实可用的 model ID。"""
    try:
        resp = client.models.list()
        available = {m.id for m in resp.data} if getattr(resp, "data", None) else set()
    except Exception:
        return requested
    if not available:
        return requested
    if requested in available:
        return requested
    short = re.sub(r"-(\d{4,8})$", "", requested)
    if short and short in available:
        return short
    low = {a.lower(): a for a in available}
    if requested.lower() in low:
        return low[requested.lower()]
    if short and short.lower() in low:
        return low[short.lower()]
    return requested


def main():
    ap = argparse.ArgumentParser(description="实验 2-3（修订版）：KV Cache 前缀复用")
    ap.add_argument("--model", default=MODEL_NAME, help="模型 ID，默认取自 .env 的 OPENAI_MODEL")
    ap.add_argument("--base-url", default=BASE_URL, help="OpenAI 兼容端点")
    ap.add_argument("--only", default=None, choices=list(SCENARIOS.keys()),
                    help="只跑某个场景；默认全部")
    ap.add_argument("--warm-repeats", type=int, default=3, help="热请求重复次数（取最快），默认 3")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="生成长度上限，默认 256")
    args = ap.parse_args()

    if not API_KEY:
        raise SystemExit(
            "❌ 未找到 API Key。请在 .env 填入 OPENAI_API_KEY\n"
            "（本地 Ollama 填 ollama 占位即可；AMD 的 Key 以 rc- 开头）。\n"
            "获取地址：https://developer.amd.com.cn/radeon/tokenfactory"
        )

    client = OpenAI(api_key=API_KEY, base_url=args.base_url)
    resolved = resolve_model(client, args.model)
    if resolved != args.model:
        print(f"🔧 模型名已自动对齐：{args.model} → {resolved}")
    args.model = resolved
    print(f"🔗 端点：{args.base_url}")
    print(f"🤖 模型：{args.model}   思考控制：{EXTRA_BODY}")

    # 预热：消除「首个请求异常慢」对 correct 冷启动基线的污染。
    # 用较大的 max_tokens（思考 token 占用预算），保证能产出 content、拿到真实 TTFT。
    _w_ttft, _, _ = timed_call(client, args.model,
                               [{"role": "user", "content": "ping"}], [], EXTRA_BODY, 256)
    _w_ttft_s = f"{_w_ttft:.0f}" if _w_ttft is not None else "?"
    print(f"🔥 预热完成（ping TTFT ≈ {_w_ttft_s} ms），开始逐场景测量…\n")

    names = [args.only] if args.only else list(SCENARIOS.keys())
    results = []
    correct_ref = None
    for n in names:
        r = run_scenario(n, client, args.model, EXTRA_BODY, args.max_tokens,
                         args.warm_repeats, correct_ref=correct_ref)
        if n == "correct" and r["speedup"]:
            correct_ref = r["speedup"]
        results.append(r)

    # 汇总表
    print("\n" + "=" * 104)
    print("📊 汇总：前缀缓存命中判定（主信号 = cached_tokens，即 prompt 中被复用的 token 数）")
    print("-" * 104)
    print(f"{'场景':<20}{'共享字符':>10}{'冷TTFT':>10}{'热TTFT':>10}{'热cached':>10}{'判定':>8}")
    for r in results:
        cached = r["warm_cached"]
        cached_s = str(cached) if cached is not None else "—"
        _ct = f"{r['cold_ttft']:>10.1f}" if r['cold_ttft'] is not None else f"{'—':>10}"
        _wt = f"{r['warm_ttft']:>10.1f}" if r['warm_ttft'] is not None else f"{'—':>10}"
        print(f"{r['name']:<20}{r['shared_chars']:>10}{_ct}{_wt}{cached_s:>10}{r['mark']:>8}")
    print("=" * 104)
    print("解读：")
    print("  1. 'totally_different' 是负对照：如果它也返回 cached>0，说明服务端缓存粒度是")
    print("     固定块，与内容无关，则前面所有场景的恒定高值都不能作为'命中'证据。")
    print("  2. '共享字符' 是理论最长公共前缀（字符级），与 cached_tokens 对照，")
    print("     可以判断服务端缓存是 token 级、块级，还是做了更宽容的匹配。")
    print("  3. TTFT 在免费共享端点/本地思考模型上受排队与思考噪声影响大，仅作旁证。")
    print("  4. 若 5 个错误场景的 cached 明显低于 correct，且 totally_different 接近 0，")
    print("     说明本次实验真正区分开了「稳定前缀」和「前缀被改动」。")
    print("     若 5 个错误场景的 cached 也都接近 correct（无差异），则需要换更长的 prompt 或换端点复测。")

if __name__ == "__main__":
    main()
