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
"""

from openai import OpenAI
import urllib.request
import json

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
    messages = [{"role": "user", "content": user_msg}]
    for turn in range(max_turns):
        print(f"\n--- 第 {turn + 1} 轮 ---")
        # 4.1 带 tools 发起（流式），让模型决定是否调用工具
        stream = client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools, stream=True
        )
        content = ""          # 模型思考/最终回复文本
        name = None           # 工具名
        args_str = ""         # 工具参数字符串（JSON 分片拼接）
        for chunk in stream:
            delta = chunk.choices[0].delta
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

        # 4.2 若模型决定调用工具
        if name:
            args = json.loads(args_str or "{}")
            print(f"  🔧 调用工具 {name}({args})")
            result = get_weather(**args) if name == "get_weather" else "未知工具"
            print(f"  👁 观察: {result}")
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
            print(f"\n💬 最终回答: {content}")
            return content
    return "（达到最大轮次，仍未产出最终回答）"


if __name__ == "__main__":
    run("What's the weather in Tokyo?")
