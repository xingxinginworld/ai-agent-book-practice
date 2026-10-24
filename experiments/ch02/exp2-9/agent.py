# -*- coding: utf-8 -*-
"""
实验 2-9 配套（二）：轻量 Agent（可开关状态栏 + 指标统计）—— agent.py
================================================================================
对应《深入理解 AI Agent》第 2 章 · 上下文工程 · 实验 2-9（几种好用的 Agent 状态栏技术）。

【这个脚本在做什么】
把 2-9 的状态栏机制落成可运行 Agent：一个会"多步决策"的轻量 Agent，每轮生成前
**可选择**把状态栏作为临时 user 消息注入上下文末尾（不写进历史），然后看状态栏是否
帮它少走弯路、不掉进重复调用。

核心对照（实验目的）：
    同一任务，跑两臂 ——
      对照臂（无状态栏）  vs  干预臂（注入状态栏：工具计数 + TODO + 时间戳 …）
    比较指标：迭代步数、工具调用次数、是否陷入重复循环、是否成功给出结论。

【为什么轻量可跑】
- 工具用 mock（search/calculate），不碰真实文件系统，安全、确定、快。
- 模型用本地 Ollama（minicpm-32k），不联网、不花钱。
- 状态栏本身就是纯 Python（见 status_bar.py），本文件只负责"注入 + 决策 + 统计"。

【关于小模型可靠性（与 2-7 同一观察）】
2.5B 模型做"结构化 JSON 动作"尚可（2-6 已验证），但偶尔输出不合法 JSON 或乱选动作。
本 Agent 对解析失败做容错（视为一步停滞、不崩溃），并把"失败/循环"也计入指标——
这恰恰能体现"没有状态栏时更容易跑偏"这一现象。
"""

import os
import sys
import json
import re
from datetime import datetime
from dotenv import load_dotenv

from status_bar import AgentState, build_status_bar, all_off

# ---------------------------------------------------------------------------
# 0. 配置加载：与 2-1~2-7 完全一致
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

# 重复调用达到此次数即记为"循环"
LOOP_THRESHOLD = 3


# ---------------------------------------------------------------------------
# 1. 客户端与模型调用（沿用 2-6 写法）
# ---------------------------------------------------------------------------
def build_client():
    if not API_KEY:
        print("❌ 未检测到 OPENAI_API_KEY。请复制仓库根目录 .env.example 为 .env 并填入密钥。")
        sys.exit(1)
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ 缺少依赖 openai，请先 pip install -r experiments/ch02/exp2-9/requirements.txt")
        sys.exit(1)
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)


def call_model(client, messages, think):
    """发一次请求（messages 为完整列表），返回文本。默认关原生思考、temperature=0。"""
    extra = dict(EXTRA_BODY)
    if not think and "enable_thinking" not in extra:
        extra["enable_thinking"] = False
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=MAX_TOKENS,
        extra_body=extra, temperature=0.0,
    )
    text = resp.choices[0].message.content or ""
    if not text and getattr(resp.choices[0].message, "reasoning_content", None):
        text = resp.choices[0].message.reasoning_content or ""
    return text


def extract_first_json(text):
    """从模型输出抠出第一个 {...} JSON 块（0.6B/2.5B 常在 JSON 外裹废话）。"""
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
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ---------------------------------------------------------------------------
# 2. Mock 工具（不碰真实文件，安全可控）
# ---------------------------------------------------------------------------
def mock_search(query: str) -> str:
    return f"关于『{query}』的资料摘要：注意力机制让模型对不同位置动态分配权重，" \
           "近期研究强调其在长文本与工具调用场景下的关键作用。"


def mock_calculate(expr: str) -> str:
    """只放行极简算术，避免任意代码执行。

    先把常见的「乘号」字形归一化（× → *、÷ → /、全角 → 半角），
    否则 '25×37' 这种任务会被白名单挡掉、工具永远失败。
    """
    if not expr:
        return "无法计算：表达式为空"
    norm = (expr.replace("×", "*").replace("÷", "/")
                .replace("（", "(").replace("）", ")")
                .replace("＋", "+").replace("－", "-"))
    allowed = set("0123456789+-*/().% ")
    if norm and all(c in allowed for c in norm):
        try:
            val = eval(norm)   # noqa: eval 仅限白名单字符
            return f"{expr} = {val}"
        except Exception as e:  # noqa: BLE001
            return f"计算出错：{e}"
    return "无法计算：表达式包含非法字符"


TOOLS = {
    "search": mock_search,
    "calculate": mock_calculate,
}

TOOL_TO_TODO = {"search": 1, "calculate": 2}  # 把工具映射到 TODO 步骤

SYSTEM_PROMPT = (
    "你是一个执行多步任务的 Agent。可用工具只有三个：\n"
    "1) search   参数 {\"query\": 搜索词}      —— 检索资料\n"
    "2) calculate 参数 {\"expr\": 算术表达式}  —— 做四则运算\n"
    "3) final_answer 参数 {\"text\": 结论}     —— 给出最终答案并结束任务\n\n"
    "请根据'任务'与已发生的历史，决定下一步动作。严格只输出一行 JSON，格式：\n"
    "{\"action\": \"search\"|\"calculate\"|\"final_answer\", \"args\": {...}}\n"
    "不要输出任何解释、不要输出 markdown 代码围栏、不要输出多余文字。"
)


# ---------------------------------------------------------------------------
# 3. 轻量 Agent
# ---------------------------------------------------------------------------
class StatusBarAgent:
    """可开关状态栏注入的轻量 Agent；跑完返回指标字典。"""

    def __init__(self, config, max_iterations=6, verbose=False):
        self.config = config
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.client = build_client()
        self.task = ""
        self._reset()

    def _reset(self):
        self.state = AgentState(
            now=datetime.now(),
            todos=[
                {"id": 1, "text": "搜索资料", "status": "in_progress"},
                {"id": 2, "text": "计算 25×37", "status": "pending"},
                {"id": 3, "text": "给出结论", "status": "pending"},
            ],
            tool_calls={},
            last_error=None,
            cwd="D:/agent-workspace",
            shell="bash",
            system="Windows 10",
            python_version="3.13",
        )
        self.history = []  # [(role, content)] 真正的对话历史（不含状态栏）
        self.metrics = {"iterations": 0, "tool_calls": {}, "loops": 0,
                        "success": False, "trace": []}

    def run(self, task: str) -> dict:
        self._reset()
        self.task = task
        use_bar = self.config != all_off()

        for i in range(self.max_iterations):
            self.metrics["iterations"] += 1

            # 组装消息：系统提示 + 任务 + 历史
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "任务：" + task},
            ]
            messages += [{"role": r, "content": c} for r, c in self.history]

            # 状态栏：作为【临时 user 消息】注入（不写进 history，避免污染历史）
            if use_bar:
                bar = build_status_bar(self.state, self.config)
                if bar:
                    messages.append({
                        "role": "user",
                        "content": "【Agent 状态栏（每轮临时注入，勿视作对话内容）】\n" + bar,
                    })

            raw = call_model(self.client, messages, think=False)
            action = self._parse_action(raw)
            self.metrics["trace"].append({
                "iter": i, "action": action, "raw": raw[:160],
            })
            if self.verbose:
                print(f"    [步 {i}] 动作={action}  状态栏={'开' if use_bar else '关'}")

            if action is None:
                # 解析失败：容错，视为停滞，记录错误后继续
                self.state.last_error = {
                    "type": "解析失败", "message": "模型未返回合法 JSON 动作",
                    "suggestion": "检查输入或换更强模型",
                }
                self.history.append(("assistant", raw))
                self.history.append(("user", "[系统] 你的输出不是合法 JSON 动作，请严格按格式输出。"))
                continue

            a = action["action"]
            args = action.get("args", {}) or {}

            if a == "final_answer":
                self.metrics["success"] = True
                self.history.append(("assistant", raw))
                return self.metrics

            if a in TOOLS:
                # 更新工具调用计数
                self.state.tool_calls[a] = self.state.tool_calls.get(a, 0) + 1
                if self.state.tool_calls[a] >= LOOP_THRESHOLD:
                    self.metrics["loops"] += 1
                # 推进 TODO
                self._advance_todo(a)
                # 执行工具：按工具名从 args 取对应字段（args 是模型返回的 dict）
                if a == "search":
                    result = mock_search(args.get("query", ""))
                elif a == "calculate":
                    result = mock_calculate(args.get("expr", ""))
                else:
                    result = "[工具未实现]"
                self.history.append(("assistant", raw))
                self.history.append(("user", f"[工具 {a} 返回] {result}"))
            else:
                # 未知动作：提醒后继续
                self.history.append(("assistant", raw))
                self.history.append(("user", f"[系统] 未知动作 '{a}'，请用 search/calculate/final_answer。"))

        # 步数耗尽仍未 final_answer
        return self.metrics

    def _advance_todo(self, action):
        tid = TOOL_TO_TODO.get(action)
        if tid is None:
            return
        for t in self.state.todos:
            if t["id"] == tid:
                t["status"] = "completed"
        # 把下一个 pending 置为 in_progress
        for t in self.state.todos:
            if t["status"] == "pending":
                t["status"] = "in_progress"
                break

    @staticmethod
    def _parse_action(raw: str):
        data = extract_first_json(raw)
        if not isinstance(data, dict) or "action" not in data:
            return None
        return data

    # ---------------------------------------------------------------------------
    # 4. 轨迹导出（对应书中"自动轨迹保存"，便于调试与复现）
    # ---------------------------------------------------------------------------
    def to_trajectory(self) -> dict:
        """把一次运行的关键状态汇成字典，便于落盘分析。"""
        return {
            "task": self.task,
            "config": self.config,
            "state": {
                "now": self.state.now.isoformat() if self.state.now else None,
                "tool_calls": self.state.tool_calls,
                "todos": self.state.todos,
            },
            "history": [{"role": r, "content": c} for r, c in self.history],
            "metrics": self.metrics,
        }

    def save_trajectory(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_trajectory(), f, ensure_ascii=False, indent=2)
