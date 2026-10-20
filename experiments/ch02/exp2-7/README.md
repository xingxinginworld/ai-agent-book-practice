# 实验 2-7：从个人范文创建「去 AI 味」写作 Skill（轻量本地可跑版）

> 对应《深入理解 AI Agent》第 2 章 · Agent Skills · 实验 2-7
> 原书正文实验 🚧 从个人范文创建"去 AI 味"写作 Skill；练习 Skill 的**触发条件 / 规则 / 示例 / 作用域 / 迭代维护**，不依赖独立代码项目。
> 本目录把它落成**一份 skill.md（说明书）+ 两个轻量脚本（风格抽取 + 改写执行）**，用本地 LLM 跑通，不联网、不花钱。

---

## 一、这个实验在练什么

第 2 章讲"Agent Skills"——把一类任务封装成**可复用的能力包**：一份 `skill.md` 说清"何时用 / 怎么用 / 举例 / 作用域 / 怎么维护"，背后可以挂脚本。

本实验练的是**用你自己的文风，做一个"去 AI 味"写作 Skill**：

- 你有一堆自己写的、自然、不像机器的文字（公众号、随笔）；
- 你也常拿到 AI 生成或套路化的书面腔草稿，想"润色得像人写的"；
- 做法：先把**你的范文**喂给本地 LLM，抽出你的"风格画像"；再把画像当成改写准则，去改写任意草稿。

它同时是观察**小模型可靠性**的好样本——2.5B 级别的本地模型在"长提示 + 自由改写"上指令跟随并不稳，本实验用"自检 + 重试 + 规则兜底"三道防线来应对（这正是 Agent 该有的健壮性）。

---

## 二、前置知识

- **Agent Skills 的结构**：`name` + `description`（决定何时被触发）+ 正文（规则/示例/作用域/迭代）。
- **上下文工程**：把"对模型有用的信息"（这里是风格画像）组织好再喂给模型，比丢一句"写得像人"有效得多。
- **小模型的不确定性**：本地 2.5B 模型会"跑偏"（去分析原文、或原样复述 + 加说明），需要验证输出、必要时重试或降级。

---

## 三、环境准备

```bash
# 1) 依赖（与 2-1~2-6 同，openai 1.x 已装可跳过）
pip install -r experiments/ch02/exp2-7/requirements.txt

# 2) 配置：复制根目录 .env.example 为 .env，填入本地 Ollama 配置
#    OPENAI_API_KEY=ollama
#    OPENAI_BASE_URL=http://127.0.0.1:11434/v1
#    OPENAI_MODEL=minicpm-32k      # 本实验在 D:\ollama 的 minicpm-32k（基于 openbmb/minicpm5-2b，2.5B）上跑通
#    MAX_TOKENS=2000
```

> 模型说明：仓库文档默认 `qwen3:0.6b`，但本机 `D:\ollama` 已就绪的是 `minicpm-32k`（2.5B，上下文 32k）。2.5B 比 0.6B 在 JSON 抽取与改写上更稳，推荐用它。`.env` 已被 `.gitignore` 忽略，不会提交。

---

## 四、分步操作

```bash
# 第 1 步：从个人范文抽取"风格画像" → style_profile.json
python experiments/ch02/exp2-7/build_skill.py
#   默认读 examples/personal_samples/*.txt（占位范文，你后续替换为自己的文字即可）
#   想用你自己的范文：--samples 你的范文目录

# 第 2 步：按风格画像去 AI 味改写一份草稿 → output/rewritten.txt
python experiments/ch02/exp2-7/rewrite.py
#   默认读 examples/ai_draft.txt（一段典型 AI 腔文字）
#   想改你自己的草稿：--draft 你的草稿.txt --out 成品.txt
```

**数据流**：

```
个人范文(若干 .txt) ──build_skill.py──▶ style_profile.json ──┐
                                                            ├─▶ rewrite.py ──▶ 人味文本
AI 草稿(.txt) ──────────────────────────────────────────────┘
```

---

## 五、代码结构与中文注释

| 文件 | 作用 |
|---|---|
| `skill.md` | 本 Skill 的说明书：触发条件、规则、示例、作用域、迭代维护。**核心交付物**（原书要练的就是它）。 |
| `build_skill.py` | 读个人范文 → 调本地 LLM 抽风格画像（语气/用词/句式/节奏/避坑/代表句）→ 写 `style_profile.json`。解析失败有启发式兜底。 |
| `rewrite.py` | 读 `style_profile.json` + 草稿 → 调本地 LLM 改写。**含自检 `is_bad_rewrite` + 重试（最多 3 次）+ 规则兜底 `rule_based_rewrite`**。 |
| `examples/personal_samples/*.txt` | 占位范文（3 段随笔）。**换成你自己的文字即可**，Skill 结构不受影响（方案 B）。 |
| `examples/ai_draft.txt` | 一段典型 AI 腔草稿（"值得注意的是…首先…其次…最后…综上所述…毫无疑问"），作对照输入。 |
| `style_profile.json` | 由 `build_skill.py` 生成（运行时产物，已 gitignore 思路：见下）。 |
| `output/rewritten.txt` | 改写成品。 |

> 两个脚本都复用 2-6 的 OpenAI 兼容客户端写法（关原生思考、temperature 可控），切换服务商只改 `.env`。

---

## 六、踩坑与排错

1. **`ollama list` 显示空 / 模型找不到**：Ollama 服务默认读 `C:\Users\<你>\.ollama`。本机模型在 `D:\ollama`，需让服务认到它——把 `OLLAMA_MODELS` 设为 `D:\ollama`（用户环境变量）后**重启 Ollama 服务**，再 `ollama list` 应看到 `minicpm-32k:latest`。
2. **改写返回空 / 跑偏去"分析原文"**：这是 2.5B 模型指令跟随不稳的典型表现，不是代码 bug。`rewrite.py` 已用"自检 + 重试 + 规则兜底"兜底；若仍频繁跑偏，可换更强模型（如 qwen3:1.7b/4b）或把 `temperature` 调到 0。
3. **`openai` 装成 3.x 下载树过重在沙箱里超时**：锁 `openai<2`（API 与本实验完全兼容，依赖轻得多）。
4. **改写里夹英文词（code-switching）**：miniCPM-5-2B 偶发中英混写（如 `aren't optional` / `competition`）。可在 `rewrite.py` 末尾加一步"中英混写清洗"，或在画像 `avoid` 里加"不混用英文"。

---

## 七、实测现象与解读（本地 minicpm-32k 实跑）

- **风格抽取（build_skill.py）**：拿到合法 JSON，未走兜底。画像归纳到位——"温暖平实、娓娓道来""短句为主、逗号断句""避免堆砌辞藻/说教语气"，并摘出代表句。说明 2.5B 做"给样本→归纳风格"这种结构化任务相当稳。
- **去 AI 味改写（rewrite.py）**：
  - 输入 AI 腔草稿（141 字，"值得注意的是…首先/其次/最后…综上所述…毫无疑问"），改写前 **AI 味指数 49.6（命中 7 处）**。
  - 输出改写后 **AI 味指数 0.0（命中 0 处）**，套路连接词与空洞总结被清空，语气自然化。
  - 小模型可靠性观察：同一长提示跑 3 次，仅约 1/3 真正改写，另外 2/3 会"分析原文"或"原样复述 + 加改写说明"——正是 `is_bad_rewrite` 自检要拦的两种跑偏。加自检后脚本稳定交差。
  - 已知瑕疵：偶发中英混写（见踩坑第 4 条），不影响"去 AI 味"结论，但发布前建议人工过一眼。
- **成品示例（output/rewritten.txt）**：
  > 在快节奏的社会里，时间管理确实太重要了。制定计划、保持专注、坚持到底——这些 aren't optional 如果想稳步成长。光有热情不够，得真去安排每一天，踏踏实实往前走。这样才能不被 competition 甩在后面，这也是现代人该有的基本能力。

> 注：`style_profile.json` 与 `output/` 是运行时产物；为让仓库可复现，本目录**保留**了 `style_profile.json` 的示例（已提交）与 `output/rewritten.txt`，但 `.gitignore` 规则若覆盖它们，重新跑 `build_skill.py` / `rewrite.py` 即可再生。

---

## 八、公众号要点 / 复盘

- **Skill 的本质 = 说明书 + 资源**。`skill.md` 写"何时用、怎么改、举例、作用域、怎么维护"；背后的 `style_profile.json` 是数据驱动的部分——补范文 → 重跑 → 画像更新，Skill 就"越用越像你"。
- **"去 AI 味"不是套模板，是数据驱动**。手写规则容易漏，让模型从你真实范文里归纳文风，改出来才贴脸。
- **小模型要配"验证 + 兜底"**。2.5B 跑自由改写会跑偏，靠自检拦掉、重试、再不行就降级规则——这比"调参祈求它听话"靠谱得多。
- **下一步练习**：把 `examples/personal_samples/` 换成你自己的 3~5 篇随笔，重跑 `build_skill.py`；再拿一段你近期被说"太 AI"的草稿跑 `rewrite.py`，看像不像你。
