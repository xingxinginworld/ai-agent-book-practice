---
name: paper-to-slides
description: 当用户给出一篇论文（PDF 或纯文本）并要求把它整理/提炼成演示文稿、幻灯片、汇报大纲、PPT 时使用。基于本地 LLM 抽取论文结构化要素，并生成一份可在浏览器离线打开的自包含 HTML 幻灯片。
---

# Paper to Slides（论文转演示文稿）

把"长论文"压缩成"短汇报"的可复用能力包。这是《深入理解 AI Agent》实验 2-6 演示的
**Agent Skill** 形态：一份 `description`（告诉 Agent 何时该用我）+ 一套固定流程 + 背后的脚本。

## 适用场景
- 用户上传/粘贴一篇论文，想快速得到汇报用幻灯片。
- 用户说"把这篇论文做成 PPT""给我一份论文汇报大纲""总结这篇论文用于组会"。

## 工作流程
1. **读取论文文本**：`.pdf` 用 `pypdf` 抽取，` .txt/.md` 直接读。
2. **阶段① 抽取**：调用本地模型，把论文转成结构化 JSON
   （title / authors / venue / problem / method / contributions / key_results / conclusions）。
3. **阶段② 大纲**：把结构化摘要交给模型，生成幻灯片大纲（标题 + 要点，含"总结与启示"收尾）。
4. **阶段③ 渲染**：用 `render.py` 把大纲渲染成自包含 HTML（无 CDN、可双击打开）。
5. **交付**：输出 HTML 路径，提示用户在浏览器打开翻页。

## 关键约束
- 默认使用本地 Ollama（`qwen3:0.6b`），不调用任何外部 API；
  若需更强质量，改 `.env` 的 `OPENAI_MODEL` 即可，其余代码不动。
- 输出演示文稿为**单文件 HTML**，不依赖网络，离线可用。
- 0.6B 模型对 JSON 不稳，代码层已做"截取首个 `{...}` 块"的兜底解析；
  若两阶段任一失败，会自动降级为基于结构化字段的规则大纲，保证总能产出可用文稿。

## 调用入口
```bash
# 用内置示例论文跑通全流程
python experiments/ch02/exp2-6/main.py --demo
# 自己的论文
python experiments/ch02/exp2-6/main.py --paper 论文.pdf --out out.html --slides 8
```

## 文件职责
- `main.py`：编排入口（读论文 → 抽取 → 大纲 → 渲染 → 写文件）
- `prompts.py`：两阶段提示词模板
- `render.py`：结构化数据 → 离线 HTML 幻灯片
- `skill.md`：本文件，即"能力包"的说明书（Agent 按 description 决定何时加载）
