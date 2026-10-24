# -*- coding: utf-8 -*-
"""
实验 2-9 配套（一）：Agent 状态栏（System Hint）生成器 —— status_bar.py
================================================================================
对应《深入理解 AI Agent》第 2 章 · 上下文工程 · 实验 2-9（几种好用的 Agent 状态栏技术）。

【这个脚本在做什么】
Agent 状态栏 = 把 Agent 运行时的真实状态，整理成一条「状态摘要」，作为**临时 user 消息**
注入上下文末尾（不写进对话历史），让模型每次生成前都能"瞥一眼"当前进度。

本脚本是这个概念的核心、纯 Python、不调模型，离线就能跑。它实现书里讲的 5 种技术：
  1) 时间戳        —— 给消息/工具结果打时间，补时间上下文
  2) 工具调用计数  —— 统计每工具调用次数，抑制无限循环/重复
  3) TODO 列表     —— 多步任务的进度管理（pending/in_progress/completed/cancelled）
  4) 详细错误      —— 错误类型/参数/建议，帮助从失败中恢复
  5) 系统状态感知  —— 当前目录/Shell/系统/Python 版本

【为什么放在末尾】-> 见 2-2 注意力可视化的"位置偏好"：模型对上下文开头和结尾注意力更高，
把关键元信息放末尾，能"强制引导注意力"（这也是 2-8 想验证的点，但 2-8 需抽注意力权重）。

【如何运行】
  python experiments/ch02/exp2-9/main.py --mode preview   # 离线看 5 种技术的对比
"""

from datetime import datetime
from dataclasses import dataclass, field


# TODO 四态 -> 图标（与书一致：pending/in_progress/completed/cancelled）
_STATUS_ICON = {
    "pending": "⏳",
    "in_progress": "🔄",
    "completed": "✅",
    "cancelled": "❌",
}


@dataclass
class AgentState:
    """Agent 运行时的真实状态快照。由程序根据真实运行情况填充，不是模型自己写的计划。"""
    now: datetime = None                          # 当前时间
    tool_calls: dict = field(default_factory=dict)  # {工具名: 调用次数}
    todos: list = field(default_factory=list)       # [{id, text, status}, ...]
    last_error: dict = None                        # {type, message, suggestion}
    cwd: str = ""                                  # 当前工作目录
    shell: str = ""                                # Shell 类型
    system: str = ""                               # 操作系统
    python_version: str = ""                       # Python 版本

    @classmethod
    def example(cls):
        """构造一个"进行了两步"的示例状态，便于预览与测试。"""
        return cls(
            now=datetime(2024, 12, 13, 10, 30, 45),
            tool_calls={"search": 2, "calculate": 1},
            todos=[
                {"id": 1, "text": "搜索'注意力机制'资料", "status": "completed"},
                {"id": 2, "text": "计算 25 × 37", "status": "in_progress"},
                {"id": 3, "text": "给出结论", "status": "pending"},
            ],
            last_error=None,
            cwd="/home/user/projects/report",
            shell="bash",
            system="Linux 5.15.0",
            python_version="3.10.0",
        )


def _status_icon(status: str) -> str:
    return _STATUS_ICON.get(status, "❓")


def build_status_bar(state: AgentState, config: dict) -> str:
    """按 config 开关，把 state 渲染成状态栏文本。

    config 键（都可单独开关）：
      enable_timestamps / enable_tool_counter / enable_todo_list /
      enable_detailed_errors / enable_system_state
    返回空串表示所有技术都关了（即"无状态栏"）。
    """
    blocks = []

    if config.get("enable_timestamps", True) and state.now:
        blocks.append("=== 当前时间 ===\n" + state.now.strftime("%Y-%m-%d %H:%M:%S"))

    if config.get("enable_tool_counter", True) and state.tool_calls:
        lines = "\n".join(f"  · {name}：已调用 {cnt} 次" for name, cnt in state.tool_calls.items())
        blocks.append("=== 工具调用计数 ===\n" + lines +
                      "\n  （若某工具调用过多，请避免重复调用同一工具）")

    if config.get("enable_todo_list", True) and state.todos:
        items = "\n".join(f"  [{t['id']}] {_status_icon(t['status'])} {t['text']}" for t in state.todos)
        blocks.append("=== 当前任务 TODO ===\n" + items +
                      "\n  （同一时间只推进一个 in_progress；完成的及时标记 completed）")

    if config.get("enable_detailed_errors", True) and state.last_error:
        e = state.last_error
        blocks.append("=== 最近错误 ===\n"
                      f"  类型：{e.get('type', '未知')}\n"
                      f"  信息：{e.get('message', '')}\n"
                      f"  建议：{e.get('suggestion', '检查参数后重试')}")

    if config.get("enable_system_state", True):
        ss = []
        if state.cwd:
            ss.append(f"当前目录：{state.cwd}")
        if state.shell:
            ss.append(f"Shell：{state.shell}")
        if state.system:
            ss.append(f"系统：{state.system}")
        if state.python_version:
            ss.append(f"Python：{state.python_version}")
        if ss:
            blocks.append("=== 系统状态 ===\n" + "\n".join(ss))

    if not blocks:
        return ""
    return "\n\n".join(blocks)


def all_on() -> dict:
    """全开配置（默认）。"""
    return {
        "enable_timestamps": True,
        "enable_tool_counter": True,
        "enable_todo_list": True,
        "enable_detailed_errors": True,
        "enable_system_state": True,
    }


def all_off() -> dict:
    """全关配置（即"无状态栏"对照臂）。"""
    return {k: False for k in all_on()}


if __name__ == "__main__":
    # 自测：打印一份示例状态栏
    st = AgentState.example()
    print("【示例状态栏（全开）】\n")
    print(build_status_bar(st, all_on()))
    print("\n【同一状态、全关（无状态栏）】\n")
    print("(空)" if build_status_bar(st, all_off()) == "" else build_status_bar(st, all_off()))
