# -*- coding: utf-8 -*-
"""
实验 2-1 精简版：用本地 Ollama 跑通 Qwen3-0.6B 的工具调用 ReAct 循环
============================================================================
前置：
    1) 安装依赖  : pip install openai
    2) 启动 Ollama: ollama serve        （另开一个终端常驻）
    3) 拉取模型   : ollama pull qwen3:0.6b
运行：
    python main.py
说明：
    这是「教学用最小可运行版」，专注演示核心循环；官方完整项目见
    bojieli/ai-agent-book/chapter2/local_llm_serving（含流式富文本、benchmark、
    多工具、vLLM 路径等）。生产/完整体验请走官方项目。

    本版本额外内置两个练习增强：
    - 一段约 200 token 的系统提示词 SYSTEM_PROMPT，作为「长共享前缀」，
      用于直观体会 KV Cache / prefix caching 对首 token 延迟(TTFT)的影响；
    - 每轮打印 TTFT（Time To First Token，首 token 延迟），
      同一进程内对比第 1 轮 / 第 2 轮即可观察前缀复用效果。
"""

from openai import OpenAI
import urllib.request
import json
from datetime import datetime

# 时间戳工具：练习时用于观察「思考→调用→观察→回答」各阶段的耗时
def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ---- 1. 连接本地 Ollama 的 OpenAI 兼容端点 ----
# Ollama 默认在 11434 端口暴露 /v1；api_key 随意填（Ollama 不校验）
BASE_URL = "http://localhost:11434/v1"
MODEL = "qwen3:0.6b"
client = OpenAI(base_url=BASE_URL, api_key="ollama")


# ---- 2. 工具定义（标准 OpenAI function-calling schema）----
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的当前气温（摄氏度）",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "城市英文名，如 Tokyo"}
            },
            "required": ["location"],
        },
    },
}]


# ---- 2.5 系统提示词（约 200 token，作为 KV Cache 共享前缀的演示用长前缀）----
SYSTEM_PROMPT = """你是一个严谨、可靠的智能助理，擅长利用工具获取实时信息并给出准确、简洁的回答。

工作原则：
1. 遇到需要实时数据（如天气、汇率、股价、时间）的问题，必须优先调用工具获取一手数据，绝不凭记忆编造数字。
2. 调用工具前先在内心确认参数完整、格式正确；若用户未提供必要信息（如城市名），应主动、礼貌地追问，而不是猜测。
3. 拿到工具返回结果后，直接基于数据作答，不画蛇添足；如果数据异常或缺失，要如实说明并给出可行建议。
4. 回答使用用户提问所用的语言；中文提问就用中文回答，保持口语化、不堆砌术语。
5. 若一次无法完成复杂任务，先给出可立即执行的部分结果，再说明后续步骤，避免让用户空等。
6. 涉及不确定或可能过时的信息时，明确告知用户"以下基于最近一次工具返回"，不要把推测当作事实。

你的目标是在保证准确性的前提下，让每一次交互都高效、可信、有温度。"""


# ---- 3. 工具实现：真实调用 Open-Meteo（免 API key）----
def get_weather(location: str) -> str:
    # 3.1 地理编码：城市名 -> 经纬度
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
    geo = json.load(urllib.request.urlopen(geo_url))
    if not geo.get("results"):
        return f"未找到城市: {location}"
    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]
    # 3.2 当前天气：取气温
    wx_url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}"
              f"&longitude={lon}&current=temperature_2m")
    wx = json.load(urllib.request.urlopen(wx_url))
    temp = wx["current"]["temperature_2m"]
    return f"{location} 当前气温 {temp}°C"


# ---- 4. ReAct 循环：思考 -> 工具调用 -> 观察 -> 回答 ----
def run(user_msg: str, max_turns: int = 5) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    for turn in range(max_turns):
        print(f"\n[{_ts()}] --- 第 {turn + 1} 轮 ---")
        # 4.1 带 tools 发起（流式），让模型决定是否调用工具
        t_req = datetime.now()          # 发请求的时刻（TTFT 起点）
        stream = client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools, stream=True
        )
        content = ""          # 模型思考/最终回复文本
        name = None           # 工具名
        args_str = ""         # 工具参数字符串（JSON 分片拼接）
        ttft = None           # 首 token 延迟 Time To First Token
        for chunk in stream:
            delta = chunk.choices[0].delta
            # 首个携带内容的 chunk 到达 = 模型吐出第一个 token（TTFT 终点）
            if ttft is None and (delta.content or delta.tool_calls):
                ttft = (datetime.now() - t_req).total_seconds()
            if delta.content:
                content += delta.content
                print(delta.content, end="", flush=True)   # 实时打印思考
            if delta.tool_calls:
                tc = delta.tool_calls[0]
                if tc.function.name:
                    name = tc.function.name
                if tc.function.arguments:
                    args_str += tc.function.arguments       # 增量 JSON 拼接
        print()
        # TTFT 才是 KV Cache 真正作用的指标：
        # 前缀被缓存复用 -> 省掉重算前缀的 prefill -> TTFT 下降
        print(f"  [{_ts()}] ⏱ TTFT(首 token 延迟): {ttft*1000:.0f} ms")

        # 4.2 若模型决定调用工具
        if name:
            args = json.loads(args_str or "{}")
            print(f"  [{_ts()}] 🔧 调用工具 {name}({args})")
            result = get_weather(**args) if name == "get_weather" else "未知工具"
            print(f"  [{_ts()}] 👁 观察: {result}")
            # 4.3 把「assistant 的工具调用」+「tool 的结果」回灌上下文
            messages.append({
                "role": "assistant", "content": content,
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": name, "arguments": args_str}
                }],
            })
            messages.append({
                "role": "tool", "tool_call_id": "call_1", "content": result
            })
        else:
            # 4.4 没有工具调用 = 信息已足够，输出最终回答
            print(f"\n[{_ts()}] 💬 最终回答: {content}")
            return content
    return "（达到最大轮次，仍未产出最终回答）"


if __name__ == "__main__":
    run("What's the weather in Tokyo?")
