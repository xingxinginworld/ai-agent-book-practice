# -*- coding: utf-8 -*-
"""
实验 2-9 主入口 —— main.py
================================================================================
对应《深入理解 AI Agent》第 2 章 · 上下文工程 · 实验 2-9（几种好用的 Agent 状态栏技术）。

用法：
  python experiments/ch02/exp2-9/main.py --mode preview
      # 离线预览：5 种状态栏技术各自补了什么（不调模型）

  python experiments/ch02/exp2-9/main.py --mode demo
      # 实跑对比：同一任务「无状态栏」vs「有状态栏」两臂，用本地 Ollama 跑，对比指标

  python experiments/ch02/exp2-9/main.py --mode single --task "你的任务"
      # 单任务实跑（可加 --no-timestamps / --no-counter / --no-todo / --no-errors / --no-state 关某项）

详见 experiments/ch02/exp2-9/README.md。
"""

import os
import sys
import json
import argparse
from dotenv import load_dotenv

from status_bar import AgentState, build_status_bar, all_on, all_off
from agent import StatusBarAgent, TOOLS

load_dotenv()

SAMPLE_TASK = ("请先搜索'注意力机制'的资料，再计算 25×37，"
              "最后用 final_answer 给出一句结论。")


# ---------------------------------------------------------------------------
# 模式一：离线预览（不调模型）
# ---------------------------------------------------------------------------
def mode_preview():
    st = AgentState.example()
    print("=" * 72)
    print("离线预览：Agent 状态栏（System Hint）的 5 种技术")
    print("=" * 72)
    print("\n【对照：完全无状态栏】—— 模型每轮生成前看不到任何运行时状态")
    print("  (空)\n")

    print("【干预：全开状态栏】—— 模型每轮生成前能看到：")
    print(build_status_bar(st, all_on()))

    # 逐项单独开关，看清每种技术各自补了什么
    techs = [
        ("enable_timestamps", "① 时间戳"),
        ("enable_tool_counter", "② 工具调用计数"),
        ("enable_todo_list", "③ TODO 列表"),
        ("enable_detailed_errors", "④ 详细错误"),
        ("enable_system_state", "⑤ 系统状态感知"),
    ]
    # ④ 示例状态无错误，临时塞一个让演示完整
    st.last_error = {"type": "FileNotFound", "message": "data/input.txt 不存在",
                     "suggestion": "确认路径或先创建文件"}

    print("\n--- 逐项单独开启，看每种技术各自补了什么 ---")
    for key, name in techs:
        cfg = {k: (k == key) for k in all_on()}  # 只开这一项
        print(f"\n>> 仅开 {name}")
        out = build_status_bar(st, cfg)
        print(out if out else "  (该项在示例状态下无内容，故为空)")


# ---------------------------------------------------------------------------
# 模式二：实跑对比（调本地 Ollama）
# ---------------------------------------------------------------------------
def _run_arm(cfg, task, runs, max_iterations, save_path=None):
    iter_list, loop_list, succ, final_traces = [], [], 0, []
    agent = None
    for idx in range(runs):
        agent = StatusBarAgent(cfg, max_iterations=max_iterations, verbose=False)
        m = agent.run(task)
        iter_list.append(m["iterations"])
        loop_list.append(m["loops"])
        if m["success"]:
            succ += 1
        final_traces.append(m["trace"])
        if idx == 0 and save_path and agent is not None:
            agent.save_trajectory(save_path)  # 落盘首轮，与报告指标一致
    return iter_list, loop_list, succ, runs, final_traces


def mode_demo(runs=1, task=SAMPLE_TASK, max_iterations=6):
    print("=" * 72)
    print("实跑对比：同一任务，无状态栏 vs 有状态栏（本地 Ollama / minicpm-32k）")
    print("=" * 72)
    print(f"任务：{task}\n")
    print(f"每臂运行 {runs} 次，单臂最大步数 {max_iterations}\n")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    off = _run_arm(all_off(), task, runs, max_iterations,
                  os.path.join(out_dir, "trajectory_control_off.json"))
    on = _run_arm(all_on(), task, runs, max_iterations,
                 os.path.join(out_dir, "trajectory_intervention_on.json"))

    # 汇总
    def avg(xs): return round(sum(xs) / len(xs), 2) if xs else 0

    print(f"{'指标':<22}{'对照(无状态栏)':<20}{'干预(有状态栏)':<20}")
    print("-" * 62)
    print(f"{'平均步数':<20}{avg(off[0]):<20}{avg(on[0]):<20}")
    print(f"{'平均循环次数':<18}{avg(off[1]):<20}{avg(on[1]):<20}")
    print(f"{'成功次数':<18}{off[2]}/{off[3]}{'':<12}{on[2]}/{on[3]}")

    # 打印首轮轨迹，便于"看模型实际怎么走"
    print("\n--- 首轮轨迹（对照臂：无状态栏）---")
    for step in off[4][0]:
        print(f"  步{step['iter']}: 动作={step['action']}")
    print("--- 首轮轨迹（干预臂：有状态栏）---")
    for step in on[4][0]:
        print(f"  步{step['iter']}: 动作={step['action']}")

    # 结论（朴素判读，供讨论）
    if on[2] >= off[2] and avg(on[1]) <= avg(off[1]):
        verdict = "✅ 状态栏至少没有拖后腿，且在减少重复循环上表现更好（符合预期）。"
    else:
        verdict = "⚠️ 本次对照中状态栏未明显占优——小模型方差较大，建议多跑几次或换更强模型后再下结论。"
    print("\n判读：" + verdict)
    print("（注：2.5B 本地模型在结构化决策上偶有抖动，结论需结合多次运行；见 README 踩坑。）")

    # 落盘两份轨迹，作为可复现工件（已在 _run_arm 内完成）
    print(f"\n📁 轨迹已保存到 {out_dir}/（trajectory_control_off.json / trajectory_intervention_on.json）")


# ---------------------------------------------------------------------------
# 模式三：单任务实跑
# ---------------------------------------------------------------------------
def mode_single(task, cfg, max_iterations=6, verbose=True):
    agent = StatusBarAgent(cfg, max_iterations=max_iterations, verbose=verbose)
    m = agent.run(task)
    print(f"\n任务：{task}")
    print(f"步数={m['iterations']}  工具调用={m['tool_calls']}  "
          f"循环次数={m['loops']}  成功={m['success']}")
    if verbose:
        print("轨迹：")
        for step in m["trace"]:
            print(f"  步{step['iter']}: 动作={step['action']}")
    return m


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="实验 2-9：Agent 状态栏技术")
    ap.add_argument("--mode", choices=["preview", "demo", "single"], default="preview",
                    help="preview=离线看 5 种技术；demo=实跑无/有状态栏对比；single=单任务")
    ap.add_argument("--task", default=SAMPLE_TASK, help="single 模式的任务文本")
    ap.add_argument("--runs", type=int, default=1, help="demo 每臂运行次数（默认 1）")
    ap.add_argument("--max-iterations", type=int, default=6, help="单臂最大步数")
    ap.add_argument("--no-timestamps", action="store_true")
    ap.add_argument("--no-counter", action="store_true")
    ap.add_argument("--no-todo", action="store_true")
    ap.add_argument("--no-errors", action="store_true")
    ap.add_argument("--no-state", action="store_true")
    ap.add_argument("--model", help="覆盖 .env 中的 OPENAI_MODEL")
    args = ap.parse_args()

    # 覆盖模型
    if args.model:
        os.environ["OPENAI_MODEL"] = args.model

    if args.mode == "preview":
        mode_preview()
    elif args.mode == "demo":
        mode_demo(runs=args.runs, task=SAMPLE_TASK, max_iterations=args.max_iterations)
    elif args.mode == "single":
        cfg = all_on()
        if args.no_timestamps: cfg["enable_timestamps"] = False
        if args.no_counter: cfg["enable_tool_counter"] = False
        if args.no_todo: cfg["enable_todo_list"] = False
        if args.no_errors: cfg["enable_detailed_errors"] = False
        if args.no_state: cfg["enable_system_state"] = False
        mode_single(args.task, cfg, max_iterations=args.max_iterations, verbose=True)


if __name__ == "__main__":
    main()
