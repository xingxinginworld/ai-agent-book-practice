# 实验 2-6：使用 Agent Skills 从论文生成演示文稿

> 对应《深入理解 AI Agent》第 2 章 · 上下文工程 · 实验 2-6（原书目录名 `agent-skills-ppt`）
> 本仓库版本：**轻量本地可跑**，推荐本地 Ollama（`qwen3:0.6b`，免费、离线），也支持任意 OpenAI 兼容端点。

## 〇、推荐：本地 Ollama 跑法

与 2-4 / 2-5 完全一致：模型你本来就有、离线、无 429 限额、可确定性解析。

### 前置准备
```bash
ollama pull qwen3:0.6b      # 若已下过可跳过
curl http://127.0.0.1:11434/api/tags   # 确认服务在跑
```

### 配置 .env（变量名与 2-1~2-5 完全一致，只改值）
```env
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://127.0.0.1:11434/v1
OPENAI_MODEL=qwen3:0.6b
```
> ⚠️ base_url 务必用 `127.0.0.1` 而非 `localhost`：部分环境 `localhost` 解析到 IPv6 `::1`，Ollama 只听 IPv4，Python 客户端会报 `Connection error`。
> ⚠️ Ollama 服务必须本机运行中（同一条命令内 `ollama serve` 或桌面客户端常驻）。服务随终端关闭而退出会报 `ConnectionRefusedError`。

### 运行
```bash
pip install -r experiments/ch02/exp2-6/requirements.txt
python experiments/ch02/exp2-6/main.py --demo                # 用内置示例论文跑通全流程
python experiments/ch02/exp2-6/main.py --paper 论文.txt      # 自己的论文（.txt/.md）
python experiments/ch02/exp2-6/main.py --paper 论文.pdf --out out.html --slides 8
```

## 一、实验目标

通过"论文 → 演示文稿"这一具体任务，**亲手体会 Agent Skills 这一上下文工程机制**：

- 理解 **Skill 是什么**：一份带 `name` / `description` 的说明书（`skill.md`）+ 背后的一组脚本，把"某类任务怎么做"固化下来，Agent 按 description 在需要时自动加载。
- 把"长论文压缩成短汇报"这一信息整理过程**显式拆成流水线**：① 抽取结构化要素 → ② 生成幻灯片大纲 → ③ 渲染成文稿。
- 用本地小模型（qwen3:0.6b）端到端跑通，理解"小模型在结构化抽取/大纲这类任务上能做到什么、做不到什么"。

## 二、核心概念

- **Agent Skill（智能体技能）**：把一类任务的能力打包成一个可复用单元。核心是一份 `SKILL.md`——`description` 决定"何时该用我"，正文写"怎么做"。本目录的 `skill.md` 就是这份说明书。
- **上下文工程（Context Engineering）**：第 2 章主线——如何把对模型有用的信息组织好、按需喂给模型。Skill 是"把常用上下文/流程预制成模块"的典型手段。
- **两阶段生成**：先让模型做"忠实的要素抽取"（结构化、低自由），再基于结构化结果做"创意性大纲"（高自由）。分阶段比"一步到位要 PPT"更稳，尤其对小模型。
- **离线 HTML 演示文稿**：输出一份不依赖任何 CDN 的单文件 HTML，双击即可翻页（←/→/空格/F 全屏），契合"轻量本地"定位。

## 三、前置条件

- Python 3.10+（本机 `my-test/.venv` 即可）
- 依赖：`openai` + `python-dotenv`（见 `requirements.txt`）；读 PDF 还需 `pypdf`（可选）
- 一个 OpenAI 兼容端点（推荐本地 Ollama `http://127.0.0.1:11434/v1`）；配置写在仓库根 `.env`（参考 `.env.example`）

## 四、实验约束（跑之前必读）

1. **`.env` 的 `OPENAI_API_KEY` 切勿提交**（已被 `.gitignore` 忽略）；本地 Ollama 填 `ollama` 占位即可。
2. **默认关闭原生思考（thinking）**：脚本默认向 Qwen3 注入 `enable_thinking:false`，理由同 2-4/2-5（吃预算、增随机、污染对照）。只有 `--think` 才打开。
3. **温度固定为 0** 以保证可复现。
4. **0.6B 对 JSON 不稳**：代码层已做"截取首个 `{...}` 块"的兜底解析；任一阶段失败会自动降级为基于结构化字段的规则大纲，保证总能产出可用文稿。

## 五、流水线设计

```
论文文本(.txt/.md/.pdf)
   │  read_paper()
   ▼
[阶段① 抽取]  EXTRACT_SYSTEM + EXTRACT_USER  → 结构化要素 JSON
   │              (title/authors/venue/problem/method/contributions/...)
   ▼
[阶段② 大纲]  OUTLINE_SYSTEM + OUTLINE_USER   → 幻灯片大纲 JSON
   │              (slides: [{title, bullets}, ... , 总结与启示])
   ▼
[阶段③ 渲染]  render_slides()  →  自包含离线 HTML 演示文稿
```

| 模块 | 文件 | 职责 |
| :-- | :-- | :-- |
| 入口 | `main.py` | 编排：读论文→抽取→大纲→渲染→写文件 |
| 提示词 | `prompts.py` | 两阶段提示词模板（占位符 `<<PAPER>>`/`<<SUMMARY>>`/`<<MAX>>`） |
| 渲染 | `render.py` | 结构化数据 → 离线 HTML 幻灯片（无 CDN） |
| 能力包 | `skill.md` | Agent Skill 说明书（name/description + 流程） |
| 示例 | `examples/sample_paper.txt` | 内置示例论文，供 `--demo` 离线跑通 |

**关键防御（让小模型也能跑通）**：
- `extract_first_json()`：从模型可能裹着废话/代码围栏的输出里抠出第一个合法 `{...}` 块。
- `_fallback_outline()`：大纲阶段失败时用结构化字段直接铺成幻灯片，绝不空手而归。

## 六、预期产物（程序逻辑推演，未实跑）

> 以下为按代码逻辑推演的"形态说明"，**非实跑结果**。实跑数字待本地 Ollama 环境下执行后回填第七节。

- 控制台依次打印：读取字数 → 抽取出的标题/作者 → 生成的幻灯片张数 → 输出 HTML 路径。
- 产出 `output/sample_paper_slides.html`（或 `--out` 指定路径），含：
  - 第 1 页封面（标题 + 作者/出处）
  - 若干内容页（研究背景/问题、方法、贡献、关键结果、总结与启示…）
  - 浏览器打开后可 ←/→/空格 翻页、F 全屏。
- 用 `--demo` + 内置示例论文，应得到约 6~8 张幻灯片的完整文稿。

## 七、实测现象与解读（待实跑回填）

> 本节留空，待本地 Ollama 环境执行 `python main.py --demo` 后，把真实张数、抽取质量、0.6B 在 JSON 上的翻车/兜底情况补进来。重点关注：
> 1. 阶段①抽取是否拿到合法 JSON，还是走了最小骨架兜底；
> 2. 阶段②大纲是否拿到合法 JSON，还是走了规则大纲兜底；
> 3. 最终幻灯片是否"看得过去"（要点不空、结构完整）。

## 八、踩坑备注

- 未配 `OPENAI_API_KEY` 时脚本直接报错退出，不会发起请求。
- 若直接 `python main.py` 报 `Connection error`/`ConnectionRefusedError`：确认 Ollama 服务在 `127.0.0.1:11434` 运行，且 base_url 用的是 `127.0.0.1` 而非 `localhost`。
- 读 `.pdf` 需 `pip install pypdf`；纯文本/示例论文无需。
- 阶段①对输入做了 6000 字截断（防超上下文）；长论文建议先摘核心部分传入。
- 0.6B 生成的要点可能偏"模板化"，这是小模型固有限制；要质量换 `OPENAI_MODEL` 为更大本地模型即可，代码无需改。

## 九、公众号记录要点（写作线索）

- 钩子：你有没有遇到"读了一篇好论文，却半天做不出汇报PPT"？Agent Skill 把这件事固化成"能力包"。
- 主线：上下文工程 ≠ 堆 prompt，而是把常用流程**预制为可复用模块**（Skill）。
- 落点：两阶段生成（先抽取后大纲）比一步到位稳；小模型也能跑通，关键在"分阶段 + 兜底"。
- 彩蛋：用 `--demo` 真跑一遍，把生成的 HTML 截图当封面素材。

## 十、参考

- 原书实验：`bojieli/ai-agent-book` → `chapter2/agent-skills-ppt`
- Agent Skills 机制：参见原书第 2 章"使用 Agent Skills"一节
