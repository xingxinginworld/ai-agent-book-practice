# ai-agent-book-practice

《深入理解 AI Agent》全书 **109** 个实验的系统性跟练仓库。

> 原书配套代码见作者仓库 [bojieli/ai-agent-book](https://github.com/bojieli/ai-agent-book)（已开源）。
> 本仓库在其基础上，按"实验 X-Y"逐实验产出**精简可运行 + 中文逐行注释**版本，并记录跟练过程，用于学习沉淀与公众号连载。

> 本仓库是「FDE / AI 综合学习集」的**独立子集**，与 [xx-fde-learning](https://github.com/xingxinginworld/xx-fde-learning) 相互交叉引用、各自独立开源。

## 跟练约定
- 每个实验一个目录：`experiments/chXX/expX-Y/README.md`，内含：目标 → 前置 → 环境 → 分步 → 核心代码+注释 → 备注/踩坑 → 公众号记录要点。
- **三档**：🔵 必做（本地轻量可实跑）· 🟡 选做（需外部 API/云密钥）· ⚪ 观察（需 GPU 训练或专用硬件，先理解不实跑）。
- 公众号：每完成一个实验对应一篇图文记录。

## 总进度（自动生成）
- 实验总数：**109**
- 三档：🔵 必做 75 · 🟡 选做 13 · ⚪ 观察 21
- 资源：轻量本地 75 · 需API密钥 13 · 需GPU训练 17 · 专用硬件 4
- 难度：★ 17 · ★★ 62 · ★★★ 30
- 当前进度：实验 2-1（进行中，首单元）

## 实验索引

### 第1章 AI Agent 入门（4 个）

| 编号 | 难度 | 二级分类 | 资源档 | 三档 | 标题 | 状态 |
| :--: | :--: | :--: | :--: | :--: | --- | :--: |
| [1-1](experiments/ch01/exp1-1/README.md) | ★★ | 直觉建立/观测演示 | 轻量本地 | 必做 | 上下文的关键作用 | 待跟练 |
| [1-2](experiments/ch01/exp1-2/README.md) | ★ | 直觉建立/观测演示 | 需API密钥 | 选做 | Kimi K3 原生Agent 能力 | 待跟练 |
| [1-3](experiments/ch01/exp1-3/README.md) | ★ | 直觉建立/观测演示 | 需API密钥 | 选做 | GPT-5.6 原生Deep Research 能力 | 待跟练 |
| [1-4](experiments/ch01/exp1-4/README.md) | ★ | 对照·消融·对比 | 轻量本地 | 必做 | 文生图工作流与原生图像生成的对照 | 待跟练 |

### 第2章 上下文工程（10 个）

| 编号 | 难度 | 二级分类 | 资源档 | 三档 | 标题 | 状态 |
| :--: | :--: | :--: | :--: | :--: | --- | :--: |
| [2-1](experiments/ch02/exp2-1/README.md) | ★ | 工程搭建 | 轻量本地 | 必做 | 本地LLM 服务部署与工具调用 | 进行中(首单元) |
| [2-2](experiments/ch02/exp2-2/README.md) | ★ | 直觉建立/观测演示 | 轻量本地 | 必做 | 注意力机制可视化 | 待跟练 |
| [2-3](experiments/ch02/exp2-3/README.md) | ★★ | 上下文工程实践 | 轻量本地 | 必做 | 常见的错误上下文管理模式 | 待跟练 |
| [2-4](experiments/ch02/exp2-4/README.md) | ★★ | 对照·消融·对比 | 轻量本地 | 必做 | 提示工程的消融实验 | 待跟练 |
| [2-5](experiments/ch02/exp2-5/README.md) | ★★ | 上下文工程实践 | 轻量本地 | 必做 | 提示注入攻防实验 | 待跟练 |
| [2-6](experiments/ch02/exp2-6/README.md) | ★★ | 工程搭建 | 轻量本地 | 必做 | 使用Agent Skills 从论文生成演示文稿 | 待跟练 |
| [2-7](experiments/ch02/exp2-7/README.md) | ★★ | 工程搭建 | 轻量本地 | 必做 | 从个人范文创建“去AI 味”写作Skill | 待跟练 |
| [2-8](experiments/ch02/exp2-8/README.md) | ★★ | 直觉建立/观测演示 | 轻量本地 | 必做 | 通过注意力可视化验证Agent 状态栏的效果 | 待跟练 |
| [2-9](experiments/ch02/exp2-9/README.md) | ★★ | 上下文工程实践 | 轻量本地 | 必做 | 几种好用的Agent 状态栏技术 | 待跟练 |
| [2-10](experiments/ch02/exp2-10/README.md) | ★★★ | 对照·消融·对比 | 轻量本地 | 必做 | 上下文压缩策略对比 | 待跟练 |

### 第3章 用户记忆和知识库（12 个）

| 编号 | 难度 | 二级分类 | 资源档 | 三档 | 标题 | 状态 |
| :--: | :--: | :--: | :--: | :--: | --- | :--: |
| [3-1](experiments/ch03/exp3-1/README.md) | ★ | 评测构建 | 轻量本地 | 必做 | 用三层次框架评估记忆系统 | 待跟练 |
| [3-2](experiments/ch03/exp3-2/README.md) | ★★ | 对照·消融·对比 | 轻量本地 | 必做 | 记忆策略的对比实验研究 | 待跟练 |
| [3-3](experiments/ch03/exp3-3/README.md) | ★★ | 记忆/RAG实践 | 轻量本地 | 必做 | 基于本地模型的智能日志脱敏 | 待跟练 |
| [3-4](experiments/ch03/exp3-4/README.md) | ★★ | 对照·消融·对比 | 轻量本地 | 必做 | 构建向量检索服务：ANN 索引算法的比较研究 | 待跟练 |
| [3-5](experiments/ch03/exp3-5/README.md) | ★★ | 记忆/RAG实践 | 轻量本地 | 必做 | 探究稀疏检索：从零实现BM25 搜索引擎 | 待跟练 |
| [3-6](experiments/ch03/exp3-6/README.md) | ★★ | 记忆/RAG实践 | 轻量本地 | 必做 | 混合检索流水线：结合稀疏、稠密与重排序 | 待跟练 |
| [3-7](experiments/ch03/exp3-7/README.md) | ★★★ | 记忆/RAG实践 | 轻量本地 | 必做 | 结构化索引：RAPTOR 与GraphRAG 的知识组织哲学 | 待跟练 |
| [3-8](experiments/ch03/exp3-8/README.md) | ★★ | 对照·消融·对比 | 轻量本地 | 必做 | 智能体化RAG 与非智能体化RAG 的对比研究 | 待跟练 |
| [3-9](experiments/ch03/exp3-9/README.md) | ★★ | 工程搭建 | 轻量本地 | 必做 | 利用智能体化RAG 构建用户记忆 | 待跟练 |
| [3-10](experiments/ch03/exp3-10/README.md) | ★★ | 工程搭建 | 轻量本地 | 必做 | 上下文感知检索：解决RAG 的上下文丢失问题 | 待跟练 |
| [3-11](experiments/ch03/exp3-11/README.md) | ★★★ | 工程搭建 | 轻量本地 | 必做 | 利用上下文感知检索增强用户记忆 | 待跟练 |
| [3-12](experiments/ch03/exp3-12/README.md) | ★★★ | 记忆/RAG实践 | 轻量本地 | 必做 | 从结构化数据中提取隐性知识：以司法判例分析为例 | 待跟练 |

### 第4章 工具（5 个）

| 编号 | 难度 | 二级分类 | 资源档 | 三档 | 标题 | 状态 |
| :--: | :--: | :--: | :--: | :--: | --- | :--: |
| [4-1](experiments/ch04/exp4-1/README.md) | ★★★ | 工具/MCP搭建 | 轻量本地 | 必做 | 主动工具发现 | 待跟练 |
| [4-2](experiments/ch04/exp4-2/README.md) | ★★ | 工程搭建 | 轻量本地 | 必做 | 感知工具MCP 服务器 | 待跟练 |
| [4-3](experiments/ch04/exp4-3/README.md) | ★★ | 对照·消融·对比 | 轻量本地 | 必做 | 多模态信息提取：三种技术范式的对比分析 | 待跟练 |
| [4-4](experiments/ch04/exp4-4/README.md) | ★★ | 工程搭建 | 轻量本地 | 必做 | 执行工具MCP 服务器 | 待跟练 |
| [4-5](experiments/ch04/exp4-5/README.md) | ★★ | 工程搭建 | 轻量本地 | 必做 | 协作工具MCP 服务器 | 待跟练 |

### 第5章 Coding Agent 与通用 Agent（16 个）

| 编号 | 难度 | 二级分类 | 资源档 | 三档 | 标题 | 状态 |
| :--: | :--: | :--: | :--: | :--: | --- | :--: |
| [5-1](experiments/ch05/exp5-1/README.md) | ★★★ | Coding/通用Agent | 轻量本地 | 必做 | 跨厂商的轨迹接管 | 待跟练 |
| [5-2](experiments/ch05/exp5-2/README.md) | ★★ | Coding/通用Agent | 轻量本地 | 必做 | 输出到一半断掉之后的接续 | 待跟练 |
| [5-3](experiments/ch05/exp5-3/README.md) | ★★ | 工程搭建 | 轻量本地 | 必做 | 使用代码生成工具提升数学解题能力 | 待跟练 |
| [5-4](experiments/ch05/exp5-4/README.md) | ★★ | 工程搭建 | 轻量本地 | 必做 | 使用代码生成工具提升逻辑思考能力 | 待跟练 |
| [5-5](experiments/ch05/exp5-5/README.md) | ★★ | 工程搭建 | 轻量本地 | 必做 | 小模型通过代码化知识提升执行规则的准确性 | 待跟练 |
| [5-6](experiments/ch05/exp5-6/README.md) | ★★ | 工程搭建 | 需API密钥 | 选做 | 基于论文的PPT 自动生成 | 待跟练 |
| [5-7](experiments/ch05/exp5-7/README.md) | ★★ | 工程搭建 | 需API密钥 | 选做 | 论文讲解视频的自动生成 | 待跟练 |
| [5-8](experiments/ch05/exp5-8/README.md) | ★★ | Coding/通用Agent | 需API密钥 | 选做 | 基于API 的智能视频剪辑 | 待跟练 |
| [5-9](experiments/ch05/exp5-9/README.md) | ★★ | 工程搭建 | 轻量本地 | 必做 | 同一个零件的两种生成路线——代码与生成模型 | 待跟练 |
| [5-10](experiments/ch05/exp5-10/README.md) | ★★★ | 工程搭建 | 轻量本地 | 必做 | 自适应的日志解析系统 | 待跟练 |
| [5-11](experiments/ch05/exp5-11/README.md) | ★★★ | 工程搭建 | 轻量本地 | 必做 | 生产日志的智能诊断系统 | 待跟练 |
| [5-12](experiments/ch05/exp5-12/README.md) | ★★ | 工程搭建 | 需API密钥 | 选做 | 动态表单生成的意图澄清系统 | 待跟练 |
| [5-13](experiments/ch05/exp5-13/README.md) | ★★ | Coding/通用Agent | 需API密钥 | 选做 | 自然语言交互的ERP Agent | 待跟练 |
| [5-14](experiments/ch05/exp5-14/README.md) | ★★ | 工程搭建 | 轻量本地 | 必做 | 对话式界面定制系统 | 待跟练 |
| [5-15](experiments/ch05/exp5-15/README.md) | ★★★ | 工程搭建 | 需API密钥 | 选做 | 动态生成软件的权限内嵌数据对象 | 待跟练 |
| [5-16](experiments/ch05/exp5-16/README.md) | ★★★ | 工程搭建 | 轻量本地 | 必做 | 开发一个能创造Agent 的Agent | 待跟练 |

### 第6章 交互：观察与动作空间的扩展（14 个）

| 编号 | 难度 | 二级分类 | 资源档 | 三档 | 标题 | 状态 |
| :--: | :--: | :--: | :--: | :--: | --- | :--: |
| [6-1](experiments/ch06/exp6-1/README.md) | ★★★ | 交互/多模态实践 | 轻量本地 | 必做 | 事件驱动的邮件处理Agent | 待跟练 |
| [6-2](experiments/ch06/exp6-2/README.md) | ★★★ | 工程搭建 | 轻量本地 | 必做 | 带并行执行和打断能力的异步Agent | 待跟练 |
| [6-3](experiments/ch06/exp6-3/README.md) | ★★★ | 交互/多模态实践 | 轻量本地 | 必做 | 模型原生异步与回合中途引导 | 待跟练 |
| [6-4](experiments/ch06/exp6-4/README.md) | ★ | 工程搭建 | 轻量本地 | 必做 | 构建传统语音Agent | 待跟练 |
| [6-5](experiments/ch06/exp6-5/README.md) | ★ | 工程搭建 | 轻量本地 | 必做 | 使用Qwen2-Audio 模拟流式语音感知 | 待跟练 |
| [6-6](experiments/ch06/exp6-6/README.md) | ★★ | 对照·消融·对比 | 轻量本地 | 必做 | 本地运行MiniCPM-o 4.5，对比端到端与自级联 | 待跟练 |
| [6-7](experiments/ch06/exp6-7/README.md) | ★★ | 交互/多模态实践 | 需API密钥 | 选做 | 基于Fish Audio 的控制标记驱动TTS | 待跟练 |
| [6-8](experiments/ch06/exp6-8/README.md) | ★ | 直觉建立/观测演示 | 轻量本地 | 必做 | 运行Computer Use（Anthropic 参考路径或开放模型路径） | 待跟练 |
| [6-9](experiments/ch06/exp6-9/README.md) | ★ | 交互/多模态实践 | 轻量本地 | 必做 | 使用browser-use 实现自动浏览器操作 | 待跟练 |
| [6-10](experiments/ch06/exp6-10/README.md) | ★ | 直觉建立/观测演示 | 专用硬件 | 观察 | 真机遥操作XLeRobot 整理桌面 | 待跟练 |
| [6-11](experiments/ch06/exp6-11/README.md) | ★ | 评测构建 | 轻量本地 | 必做 | 在模拟器中测量同任务的理想控制上限 | 待跟练 |
| [6-12](experiments/ch06/exp6-12/README.md) | ★★ | 直觉建立/观测演示 | 专用硬件 | 观察 | 使用Gemini Robotics-ER 1.5 驱动XLeRobot 自主整理桌面 | 待跟练 |
| [6-13](experiments/ch06/exp6-13/README.md) | ★★ | 对照·消融·对比 | 专用硬件 | 观察 | 在模拟器中比较三种自主整理桌面的闭环 | 待跟练 |
| [6-14](experiments/ch06/exp6-14/README.md) | ★★★ | 交互/多模态实践 | 专用硬件 | 观察 | 同一桌面任务的RGB 跨环境测试 | 待跟练 |

### 第7章 Agent 的评估（14 个）

| 编号 | 难度 | 二级分类 | 资源档 | 三档 | 标题 | 状态 |
| :--: | :--: | :--: | :--: | :--: | --- | :--: |
| [7-1](experiments/ch07/exp7-1/README.md) | ★ | 对照·消融·对比 | 轻量本地 | 必做 | 运行τ²-bench 并对比τ-bench 的演进 | 待跟练 |
| [7-2](experiments/ch07/exp7-2/README.md) | ★ | 评测构建 | 轻量本地 | 必做 | 人肉执行基准测试任务 | 待跟练 |
| [7-3](experiments/ch07/exp7-3/README.md) | ★★ | 评测构建 | 轻量本地 | 必做 | 构建基于Rubric 的用户记忆评估系统 | 待跟练 |
| [7-4](experiments/ch07/exp7-4/README.md) | ★★ | 对照·消融·对比 | 轻量本地 | 必做 | Advanced JSON Cards 与RAG 的对比评估 | 待跟练 |
| [7-5](experiments/ch07/exp7-5/README.md) | ★★ | 评测构建 | 轻量本地 | 必做 | 构建全自动TTS 质量评估流水线 | 待跟练 |
| [7-6](experiments/ch07/exp7-6/README.md) | ★★ | 评测构建 | 轻量本地 | 必做 | 对AndroidWorld 失败轨迹做失败归因 | 待跟练 |
| [7-7](experiments/ch07/exp7-7/README.md) | ★★ | 评测构建 | 轻量本地 | 必做 | 轨迹前缀边界评估：同一上下文的多种表示 | 待跟练 |
| [7-8](experiments/ch07/exp7-8/README.md) | ★★ | 对照·消融·对比 | 轻量本地 | 必做 | 从配对比较数据构建模型排行榜 | 待跟练 |
| [7-9](experiments/ch07/exp7-9/README.md) | ★★ | 评测构建 | 轻量本地 | 必做 | 在固定Coding Harness 中测量模型的行动阈值 | 待跟练 |
| [7-10](experiments/ch07/exp7-10/README.md) | ★ | 评测构建 | 轻量本地 | 必做 | Agent 任务的端到端成本分析 | 待跟练 |
| [7-11](experiments/ch07/exp7-11/README.md) | ★★ | 评测构建 | 轻量本地 | 必做 | 多维度模型性能基准测试 | 待跟练 |
| [7-12](experiments/ch07/exp7-12/README.md) | ★★ | 评测构建 | 轻量本地 | 必做 | 用户记忆系统的端到端选型评估 | 待跟练 |
| [7-13](experiments/ch07/exp7-13/README.md) | ★★★ | 评测构建 | 需API密钥 | 选做 | AndroidWorld 的评估和改进 | 待跟练 |
| [7-14](experiments/ch07/exp7-14/README.md) | ★★ | 评测构建 | 轻量本地 | 必做 | 配置OpenVLA 与RoboTwin2 的具身智能环境 | 待跟练 |

### 第8章 模型后训练（19 个）

| 编号 | 难度 | 二级分类 | 资源档 | 三档 | 标题 | 状态 |
| :--: | :--: | :--: | :--: | :--: | --- | :--: |
| [8-1](experiments/ch08/exp8-1/README.md) | ★ | 模型训练/后训练 | 轻量本地 | 必做 | Q-learning 在寻宝游戏中的表现 | 待跟练 |
| [8-2](experiments/ch08/exp8-2/README.md) | ★★ | 对照·消融·对比 | 轻量本地 | 必做 | 传统RL 与LLM Agent 的对比研究 | 待跟练 |
| [8-3](experiments/ch08/exp8-3/README.md) | ★★ | 模型训练/后训练 | 需GPU训练 | 观察 | 从头训练LLM——算法改进的威力 | 待跟练 |
| [8-4](experiments/ch08/exp8-4/README.md) | ★★ | 模型训练/后训练 | 需GPU训练 | 观察 | 自己训练VLM | 待跟练 |
| [8-5](experiments/ch08/exp8-5/README.md) | ★★ | 模型训练/后训练 | 需GPU训练 | 观察 | 继续预训练学习新语言 | 待跟练 |
| [8-6](experiments/ch08/exp8-6/README.md) | ★★★ | 模型训练/后训练 | 需GPU训练 | 观察 | 语音SFT——从“声音复制”到“副语言建模”[扩展实验] | 待跟练 |
| [8-7](experiments/ch08/exp8-7/README.md) | ★★★ | 模型训练/后训练 | 需GPU训练 | 观察 | 多语言思考——让模型用任意语言思考[扩展实验] | 待跟练 |
| [8-8](experiments/ch08/exp8-8/README.md) | ★★ | 模型训练/后训练 | 需GPU训练 | 观察 | Prompt 蒸馏——以更小开销复现可用能力 | 待跟练 |
| [8-9](experiments/ch08/exp8-9/README.md) | ★★★ | 模型训练/后训练 | 需GPU训练 | 观察 | 思维链（Chain of Thought, CoT）蒸馏 | 待跟练 |
| [8-10](experiments/ch08/exp8-10/README.md) | ★★ | 模型训练/后训练 | 需GPU训练 | 观察 | AdaptThink——学会“何时不思考” | 待跟练 |
| [8-11](experiments/ch08/exp8-11/README.md) | ★★ | 对照·消融·对比 | 需GPU训练 | 观察 | GeneralPoints——单轮RL 的“记忆与泛化”对照 | 待跟练 |
| [8-12](experiments/ch08/exp8-12/README.md) | ★★★ | 模型训练/后训练 | 需GPU训练 | 观察 | V-IRL-VL——多轮视觉导航 | 待跟练 |
| [8-13](experiments/ch08/exp8-13/README.md) | ★★★ | 模型训练/后训练 | 需GPU训练 | 观察 | SimpleVLA-RL——结果奖励下的开放探索[扩展实验] | 待跟练 |
| [8-14](experiments/ch08/exp8-14/README.md) | ★★★ | 模型训练/后训练 | 需GPU训练 | 观察 | ReTool——代码解释器增强数学解题 | 待跟练 |
| [8-15](experiments/ch08/exp8-15/README.md) | ★★★ | 模型训练/后训练 | 需GPU训练 | 观察 | AWorld-train——在沙盒中学习使用工具 | 待跟练 |
| [8-16](experiments/ch08/exp8-16/README.md) | ★★★ | 模型训练/后训练 | 需GPU训练 | 观察 | RLVP——奖励结果、惩罚路径 | 待跟练 |
| [8-17](experiments/ch08/exp8-17/README.md) | ★★ | 模型训练/后训练 | 需GPU训练 | 观察 | 从“过早结束”问题案例到DPO 修复 | 待跟练 |
| [8-18](experiments/ch08/exp8-18/README.md) | ★★ | 模型训练/后训练 | 需GPU训练 | 观察 | 作用域敏感的中文弯引号SFT | 待跟练 |
| [8-19](experiments/ch08/exp8-19/README.md) | ★★ | 模型训练/后训练 | 需GPU训练 | 观察 | 特殊字符串的精确复制SFT | 待跟练 |

### 第9章 Agent 的持续进化（9 个）

| 编号 | 难度 | 二级分类 | 资源档 | 三档 | 标题 | 状态 |
| :--: | :--: | :--: | :--: | :--: | --- | :--: |
| [9-1](experiments/ch09/exp9-1/README.md) | ★★ | 工程搭建 | 轻量本地 | 必做 | 为客服Agent 构建轨迹验证器 | 待跟练 |
| [9-2](experiments/ch09/exp9-2/README.md) | ★★ | 持续进化/自改进 | 轻量本地 | 必做 | 从τ²-bench 失败轨迹提炼转接与工具使用规则 | 待跟练 |
| [9-3](experiments/ch09/exp9-3/README.md) | ★★ | 工程搭建 | 轻量本地 | 必做 | 基于失败轨迹优化航空客服的系统Prompt | 待跟练 |
| [9-4](experiments/ch09/exp9-4/README.md) | ★★ | 工程搭建 | 轻量本地 | 必做 | 从用户反馈中进化需求澄清与Spec 确认Skill | 待跟练 |
| [9-5](experiments/ch09/exp9-5/README.md) | ★★★ | 工程搭建 | 轻量本地 | 必做 | 从浏览器轨迹生成可验证工作流 | 待跟练 |
| [9-6](experiments/ch09/exp9-6/README.md) | ★★★ | 持续进化/自改进 | 轻量本地 | 必做 | 由失败轨迹触发Agent 自我修改 | 待跟练 |
| [9-7](experiments/ch09/exp9-7/README.md) | ★★ | 持续进化/自改进 | 轻量本地 | 必做 | 由用户反馈触发高风险操作确认门禁 | 待跟练 |
| [9-8](experiments/ch09/exp9-8/README.md) | ★★★ | 持续进化/自改进 | 轻量本地 | 必做 | 把这本书交给Hermes：它能升级自己吗？ | 待跟练 |
| [9-9](experiments/ch09/exp9-9/README.md) | ★★★ | 评测构建 | 轻量本地 | 必做 | 评估Agent 是否在持续进化 | 待跟练 |

### 第10章 多 Agent 协作（6 个）

| 编号 | 难度 | 二级分类 | 资源档 | 三档 | 标题 | 状态 |
| :--: | :--: | :--: | :--: | :--: | --- | :--: |
| [10-1](experiments/ch10/exp10-1/README.md) | ★★ | 对照·消融·对比 | 轻量本地 | 必做 | 共享上下文中的多角色转换——系统提示词与Skill 的对比 | 待跟练 |
| [10-2](experiments/ch10/exp10-2/README.md) | ★★ | 工程搭建 | 轻量本地 | 必做 | 书籍翻译Agent | 待跟练 |
| [10-3](experiments/ch10/exp10-3/README.md) | ★★★ | 工程搭建 | 需API密钥 | 选做 | 自主编排的电话+ 电脑Agent | 待跟练 |
| [10-4](experiments/ch10/exp10-4/README.md) | ★★★ | 工程搭建 | 需API密钥 | 选做 | 同时从多个网站搜集信息的Agent | 待跟练 |
| [10-5](experiments/ch10/exp10-5/README.md) | ★ | 直觉建立/观测演示 | 轻量本地 | 必做 | 运行斯坦福AI 小镇 | 待跟练 |
| [10-6](experiments/ch10/exp10-6/README.md) | ★★★ | 工程搭建 | 需API密钥 | 选做 | 语音狼人杀Agent 系统 | 待跟练 |

## 参考
- 原书：《深入理解 AI Agent》（博杰力 著）
- 官方代码仓库：https://github.com/bojieli/ai-agent-book

## 关注与联系

- 公众号：**在不确定中交付（allisbeok）**
- 个人微信（扫码添加，图片已缩放至原图 1/3）：

<img src="assets/wechat-qr.png" alt="个人微信二维码" width="273" />
