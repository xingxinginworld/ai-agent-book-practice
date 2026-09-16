# 实验 2-1 ★ 本地 LLM 服务部署与工具调用

> 所属章节：第 2 章 上下文工程 ｜ 分类：工程搭建（本地部署 + 工具调用）｜ 资源档：轻量本地 ｜ 三档：🔵 必做 ｜ 适合对象：通用型 / 工程向

## 一、实验目标

通过 `local_llm_serving` 项目，在本地用 **vLLM 或 Ollama** 部署一个 **OpenAI 兼容**的 LLM 服务，并用 **0.6B 超小模型（Qwen3-0.6B）** 跑通「思考 → 工具调用 → 观察 → 回答」的 **ReAct 循环**，验证：

> 具备思维链（CoT）思考和工具调用能力的模型，不一定需要很大的参数量；即使是 0.6B 的超小模型，在合理的提示词与系统架构下，也能可靠地完成工具调用。

## 二、你会观察到什么（原书要点）

1. **小模型的能力**：0.6B 模型在适当 prompt 工程下，能准确理解并执行工具调用。
2. **性能表现**：作者 M2 芯片上 >100 token/s，对实时交互完全够用。
3. **ReAct 循环**：模型通过多轮「思考—行动—观察」解决复杂问题。
4. **流式响应**：实时看到模型思考、工具调用决策、结果处理。
5. **KV Cache 伏笔（关键，但有前提）**：若服务端开启 prefix caching、且系统提示词（共享前缀）足够长，保持前缀不变连续两次对话，第二次首 token 延迟（TTFT）会明显更低；改动前缀任意字符后再对话，缓存失效、延迟变高、成本上升——这正是下一节「KV Cache 友好的上下文设计」的引子。注意：默认 Ollama + 极短/无系统提示词时，这个差异可能小到看不见（见第八节踩坑）。本目录 `main.py` 已内置 TTFT 测量，可亲手验证。

## 三、前置条件

- **Python 3.12**（官方用 `uv`；本文额外给 `pip` 方案）。
- **后端二选一**：
  - Ollama（macOS / 原生 Windows / 无 GPU 的 Linux）——默认走这条。
  - vLLM（仅 Linux / WSL2 + NVIDIA GPU）——性能更好。
- **模型**：`qwen3:0.6b`（Ollama）或 `Qwen/Qwen3-0.6B`（vLLM）。
- 联网（天气工具走 Open-Meteo，免 API key）。

## 四、两条路线

- **路线 A（推荐先跑通）**：克隆原书官方仓库 `bojieli/ai-agent-book`，直接运行完整 `chapter2/local_llm_serving` 项目（含流式、`run_experiment.py`、benchmark 等）。
- **路线 B（本仓库提供）**：自包含精简版 `main.py`，不依赖大仓库，直接连本地 Ollama，专注演示核心 ReAct 工具调用循环，便于逐行讲透。

## 五、路线 A：官方项目运行步骤

```bash
# 1) 克隆并安装第2章环境（需要 uv）
git clone https://github.com/bojieli/ai-agent-book.git
cd ai-agent-book
uv sync --locked --python 3.12 --extra ch2
#   Linux/WSL + GPU 可加: --extra vllm
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2) 拉起 Ollama 并下载小模型
ollama serve                       # 另开一个终端常驻
ollama pull qwen3:0.6b

# 3) 运行工具调用演示
cd chapter2/local_llm_serving
python main.py --backend ollama --mode single  --task "What's the weather in Tokyo?"
python main.py --backend ollama --mode interactive
python main.py --backend ollama --info
```

关键文件：`main.py`（统一入口）、`agent.py`/`ollama_native.py`（Agent 循环）、`tools.py`（工具实现）、`server.py`（vLLM 服务）、`benchmark.py`（吞吐/TTFT/KV Cache 压测）。

## 六、路线 B：自包含精简版（main.py 详解）

本目录 `main.py` 用 `openai` Python 客户端指向本地 Ollama 的 OpenAI 兼容端点，完整复现 ReAct 工具调用。核心结构（逐段注释见源文件）：

1. **工具定义**：标准 OpenAI `function-calling` schema，声明 `get_weather(location)`。
2. **工具实现**：真实调用 Open-Meteo（地理编码 + 当前气温），免 key。
3. **ReAct 循环**：
   - 带 `tools` 调 `chat.completions.create(stream=True)`；
   - 流式聚合 `content`（思考/回答）与 `tool_calls`（函数名 + 参数 JSON）；
   - 若模型产出 `tool_calls` → 解析参数 → 执行工具 → 把「assistant(tool_calls) + tool 结果」回灌 `messages` → 再请求下一轮；
   - 若无 `tool_calls` → 本轮即最终回答，结束。

增强：本 `main.py` 额外内置 (a) 一段约 200 token 的 `SYSTEM_PROMPT` 作为「长共享前缀」；(b) 每轮打印 `⏱ TTFT(首 token 延迟)`；(c) 捕获 Qwen3 原生思考 `reasoning_content` 并实时打印 `🧠 思考: ...`，让「模型在想什么」可见。同一进程内对比第 1 轮 / 第 2 轮的 TTFT 即可观察前缀复用；跨多次 `python main.py` 是独立进程，KV 不共享，看不到差异属正常。

运行：

```bash
pip install openai
ollama serve && ollama pull qwen3:0.6b   # 另开终端常驻
python main.py                            # 默认问 "What's the weather in Tokyo?"
```

## 七、关键机制注解

- **输出顺序**：支持 CoT 的模型（如 Qwen3）先输出 `<think>` 思考（分析意图、评估工具、规划顺序），再输出给用户的文本，最后才是 `tool_call` 请求。理解顺序对实现流式响应很关键——`<think>` 出现即切「思考中」状态，首个工具参数校验通过即可并行执行。**本目录 `main.py` 已单独捕获 `reasoning_content` 并打印 `🧠 思考:` 前缀**，运行 `python main.py` 即可实时看见思考过程；若模型对极简单问题不输出思考（如 0.6B 直接调工具）、或 Ollama 关闭了 think，则不显示，属正常。
- **并行工具调用**：模型发现子问题间无依赖（如温哥华时间 + 天气），会在一次输出里同时生成多个 `tool_call`，框架可并行执行加速。
- **终止判断**：工具结果送回后，模型自行判断是否信息足够；足够则输出最终回复（不含工具调用），否则继续产出新的 `tool_call` 进入下一轮。
- **工具调用 JSON 结构（OpenAI 兼容）**：
  ```json
  {
    "tool_calls": [{
      "id": "call_123", "type": "function",
      "function": { "name": "get_weather", "arguments": {"location": "Tokyo"} }
    }]
  }
  ```

## 八、踩坑备注

- **原生 Windows 只能用 Ollama**：vLLM 官方仅支持 Linux，Windows 需走 WSL2 或 Linux 容器。
- **Ollama 版本**：`qwen3:0.6b` 需要较新的 Ollama 才支持**原生工具调用**；老版本可能退化成文本里吐函数名，导致解析失败。
- **流式聚合**：`tool_calls` 在流里是分片增量 JSON，需把每段 `arguments` 字符串拼接后再 `json.loads`。
- **KV Cache 观察实验（须明前提）**：本 `main.py` 已内置 TTFT 测量（每轮打印 `⏱ TTFT`）。同一进程内对比第 1 轮 vs 第 2 轮的 TTFT 才有意义；跨多次 `python main.py` 是三个独立进程，KV 不共享，自然「差不多」。要想稳定看到「第二次明显更低」，需要 (a) 较长的共享前缀（脚本已内置约 200 token 系统提示词）且 (b) 服务端开启 prefix caching（如 vLLM `--enable-prefix-caching`）。Ollama 默认跨请求未必缓存前缀；若把系统提示词清空/改短，差异也会立刻消失。

## 九、公众号记录要点（写作线索）

- 核心洞察：**「模型大小不是唯一决定因素」**——0.6B 也能干活，端侧 Agent 比预期更近。
- 对比：大模型的「思考—行动—观察」循环，和小时候「想清楚再动手」是同一套认知。
- 埋伏笔：为什么改一行系统提示词就变慢？引出后续的 KV Cache 与上下文工程。

## 十、参考

- 原书：第 2 章 2.1 / 2.3（KV Cache 友好的上下文设计）
- 官方代码：`bojieli/ai-agent-book` → `chapter2/local_llm_serving/`
