# 实验 2-9：几种好用的 Agent 状态栏技术（轻量本地可跑版）

> 对应《深入理解 AI Agent》第 2 章 · 上下文工程 · 实验 2-9。
> 原书用 Kimi API，本仓库改为**本地 Ollama（minicpm-32k）**实跑，零费用、可离线、可复现。
> 核心交付物：`status_bar.py`（状态栏生成器，纯 Python）+ `agent.py`（轻量 Agent，可开关状态栏）+ `main.py`（preview 离线演示 / demo 实跑对比）。

---

## 一、这个实验在练什么

**Agent 状态栏（System Hint）** = 把 Agent 运行时的真实状态（时间、工具计数、TODO、错误、系统环境），整理成一条**结构化状态摘要**，作为一条**临时 `user` 消息注入上下文末尾**（不写进对话历史），让模型每轮生成前都能"瞥一眼"当前进度。

为什么放在**末尾**？（这是 2-2 / 2-8 验证过的现象）模型对上下文的**开头和结尾**注意力更高，把关键元信息放末尾能"强制引导注意力"——这也正是 2-8 想用真实注意力权重验证的点（2-8 需要 `transformers` 抽注意力，本沙箱连不上 HuggingFace，故先做 2-9）。

本实验练 5 种状态栏技术，并做**对照实验**：同一任务跑两臂——
- **对照臂（无状态栏）**：模型每轮看不到任何运行时状态；
- **干预臂（全开状态栏）**：每轮注入时间戳 + 工具计数 + TODO + 详细错误 + 系统状态。

比较指标：步数、工具调用、是否陷入重复循环、是否成功。

---

## 二、前置知识

- **上下文工程（Context Engineering）**：不是把所有信息一股脑塞给模型，而是*有选择、有位置*地组织上下文。
- **位置偏好（Position Bias）**：语言模型对长上下文的开头/结尾更"上心"，中间易被忽略（2-2 注意力可视化已观察）。
- **System Hint / 状态栏**：一种低成本、高杠杆的上下文技巧——把"程序知道的真相"（而非模型自己猜的计划）以固定格式喂回去。
- 本实验的 Agent 是 ReAct 风格的极简版：模型输出 `{"action": ..., "args": ...}`，环境执行工具，结果回填历史。

---

## 三、环境准备

```bash
# 1) 依赖（openai 1.x + python-dotenv，已在托管 venv 装好，可跳过）
C:/Users/guanxx/.workbuddy/binaries/python/envs/default/Scripts/python.exe \
    -m pip install "openai<2" python-dotenv

# 2) 配置：仓库根目录已有 .env（Ollama 本地，minicpm-32k），关键四项：
#    OPENAI_API_KEY=ollama
#    OPENAI_BASE_URL=http://127.0.0.1:11434/v1
#    OPENAI_MODEL=minicpm-32k      # D:\ollama 下的模型（openbmb/minicpm5-2b，2.5B）
#    MAX_TOKENS=2000

# 3) 启动 Ollama（务必指向 D:\ollama，否则 ollama list 为空）
OLLAMA_MODELS=D:/ollama ollama serve
```

> ⚠️ **重要坑**：本机模型在 `D:\ollama`，Ollama 默认读 `C:\Users\guanxx\.ollama`（空的）。
> 必须设 `OLLAMA_MODELS=D:\ollama` 再起服务，否则 `ollama list` 显示空、API 报 404。

---

## 四、分步操作

```bash
# 第 1 步：离线预览——5 种技术各自补了什么（不调模型，秒出）
python experiments/ch02/exp2-9/main.py --mode preview

# 第 2 步：实跑对比——同一任务「无状态栏」vs「有状态栏」（调本地 Ollama）
python experiments/ch02/exp2-9/main.py --mode demo --runs 3
#   --runs 3 表示每臂跑 3 次取均值，用来捕捉小模型的方差

# 第 3 步（可选）：单任务 + 关掉某项技术，看消融影响
python experiments/ch02/exp2-9/main.py --mode single --no-todo --task "你的任务"
```

产出：
- `output/trajectory_control_off.json` —— 对照臂首轮轨迹（可复现工件）；
- `output/trajectory_intervention_on.json` —— 干预臂首轮轨迹。

---

## 五、代码结构与中文注释

| 文件 | 职责 |
|---|---|
| `status_bar.py` | **核心**。`AgentState` 状态快照 + `build_status_bar(state, config)` 渲染。纯 Python、不调模型，5 种技术可单独开关。 |
| `agent.py` | 轻量 ReAct Agent：`StatusBarAgent`。每轮把状态栏作为**临时 user 消息**注入末尾（不写历史）；含 mock 工具（search/calculate）、JSON 动作解析、循环检测、指标统计、轨迹导出。 |
| `main.py` | CLI：`preview`（离线对比）/ `demo`（实跑两臂）/ `single`（单任务 + 消融）。 |
| `requirements.txt` | `openai<2`、`python-dotenv`（`matplotlib` 本实验未用，留作扩展画图）。 |

**状态栏注入的关键代码（`agent.py`）：**
```python
# 状态栏作为【临时 user 消息】注入（不写进 history，避免污染历史）
if use_bar:
    bar = build_status_bar(self.state, self.config)
    if bar:
        messages.append({
            "role": "user",
            "content": "【Agent 状态栏（每轮临时注入，勿视作对话内容）】\n" + bar,
        })
```

**5 种技术（`status_bar.py` 的 `build_status_bar`）：**
1. `enable_timestamps` —— 当前时间，补时间上下文；
2. `enable_tool_counter` —— 每工具调用次数，并提示"调用过多请避免重复"（**抑制循环的关键**）；
3. `enable_todo_list` —— 多步任务进度（pending/in_progress/completed/cancelled）；
4. `enable_detailed_errors` —— 最近错误的类型/信息/建议，帮模型从失败中恢复；
5. `enable_system_state` —— 当前目录/Shell/系统/Python 版本。

---

## 六、踩坑与排错

1. **Ollama 指向错目录**：`ollama list` 为空 → 设 `OLLAMA_MODELS=D:\ollama` 后重启服务。
2. **`No module named 'dotenv'`**：用了系统 `python` 而非托管 venv。务必用
   `C:/Users/guanxx/.workbuddy/binaries/python/envs/default/Scripts/python.exe`（装了 openai/dotenv 的那个）。
3. **小模型偶发输出不合法 JSON**：`extract_first_json` 抠第一个 `{...}` 容错；解析失败视为一步停滞（记录错误、不崩溃），并把"失败/循环"计入指标——这恰好能体现"无状态栏时更易跑偏"。
4. **乘号 `×` 被计算器拒**：初版 `mock_calculate` 的白名单不含 `×`（U+00D7），`25×37` 会判为非法。已加归一化（`× → *`、`÷ → /`、全角转半角）。
5. **状态栏不是免费午餐**：在简单任务 + 2.5B 小模型上，多余上下文偶尔会**干扰**决策（见第七节），这是真实的权衡，不是 bug。

---

## 七、实测现象与解读（本地 minicpm-32k 实跑）

**环境**：minicpm-32k（openbmb/minicpm5-2b，2.5B，上下文 32k），temperature=0，关原生思考。
**任务**：「请先搜索'注意力机制'的资料，再计算 25×37，最后用 final_answer 给出一句结论。」
**每臂 3 次，单臂最大步数 6。**

### 指标对比（一次典型 3 次运行）

| 指标 | 对照（无状态栏） | 干预（全开状态栏） |
|---|---|---|
| 平均步数 | 3.0 | 4.33 |
| 平均循环次数 | 0.0 | 0.0 |
| 成功次数 | 3/3 | 2/3 |

### 首轮轨迹（两条都走通 `search → calculate → final_answer`）
- 对照臂：步0 search → 步1 calculate（返回 `25×37 = 925`）→ 步2 final_answer ✅
- 干预臂：步0 search → 步1 calculate（返回 925）→ 步2 final_answer ✅

### 怎么解读这个结果（重点）

- **在这个简单任务上，状态栏没有带来明显增益，偶尔还略拖后腿。** 这恰恰是本实验最有价值的发现：
  **上下文工程是"权衡"，不是"越多越好"**。把 5 行状态栏塞进本就简单的决策，对 2.5B 小模型而言是"噪声"多于"信号"。
- **状态栏的真正价值在别的 regime 里显现**：
  1. **长程任务 / 多步编排**：模型容易"忘了做到哪"，TODO 列表能把它拉回正轨；
  2. **易跑偏 / 易循环**：工具计数 + "调用过多请避免重复"的提示，是抑制死循环最直接的手段；
  3. **从失败恢复**：详细错误（类型/建议）比一句"出错了"有用得多；
  4. **更强模型**：能真正"读懂"状态栏并据此调整策略的模型，获益明显大于 2.5B。
- **小模型方差大**：2.5B 在结构化决策上偶有抖动（同一任务 3 次里可能 1 次跑偏）。结论需多次运行或换更强模型后再下，不能凭一次下结论。

> 想直观看到"状态栏帮上忙"的场景，可换一个**更长、更容易重复**的任务跑 `single` 模式，例如
> `--task "依次完成：搜索A、计算B、搜索C、计算D、汇总"`，观察对照臂是否更容易陷入重复调用、干预臂的工具计数提示能否把它拽出来。

---

## 八、公众号要点 / 复盘

- **一句话**：Agent 状态栏 = 把"程序知道的真相"以固定格式、塞进上下文**末尾**的临时提示，用极低成本换取模型少走弯路。
- **为什么在末尾**：模型对开头/结尾注意力更高（2-2、2-8 的注意力可视化观察），末尾注入 = 强制引导注意力。
- **5 个趁手的开关**：时间戳、工具计数（抑制循环）、TODO、详细错误、系统状态。
- **别迷信"加上下文"**：小模型 + 简单任务上，状态栏可能反成噪声——工程上要按需开启、按任务消融。
- **延伸**：2-8 用真实注意力权重验证"状态栏到底有没有引导注意力"；2-10 接着讲"上下文压缩"，和状态栏是同一个上下文工程主题的两面。

---

## 九、与原书差异说明

- 原书用 Kimi API（需 Key）；本仓库改用本地 Ollama `minicpm-32k`，离线免费、可复现。
- 原书的"Agent 状态栏"是概念 + 在线演示；本仓库落地为一个**可开关注入的轻量 Agent + 对照实验**，把"有没有用"变成可量化的指标。
- 依赖精简为 `openai<2` + `python-dotenv`（不引入 torch/transformers），保持"轻量本地"定位。
