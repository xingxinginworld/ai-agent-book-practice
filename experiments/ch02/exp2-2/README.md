# 实验 2-2 ★ 注意力机制可视化

> 所属章节：第 2 章 上下文工程 ｜ 分类：直觉建立/观测演示 ｜ 资源档：轻量本地（但需下载 ~1–2GB 模型权重）｜ 三档：🔵 必做 ｜ 适合对象：通用型 / 工程向 / 想"看见"模型内部的人

## 一、实验目标

用本地一个小模型（默认 `Qwen/Qwen3-0.6B`）加载一段文本，捕获模型**每一层、每一头**的自注意力（self-attention）权重，把它画成一张 **token-to-token 的注意力热力图**，从而直观回答一个问题：

> 模型在读一句话时，注意力到底"看"向了哪些 token？是均匀分散，还是有明显的汇聚点？

验证两个重要的直观现象：

1. **Attention Sink（注意力沉淀）**：最后一层里，大量注意力会"压"在序列第 1 个 token 上——即使它语义上并不重要。
2. **因果三角（Causal Triangle）**：每个 token 只能关注"自身及之前的 token"，因此热力图的上三角一定是空的（被因果掩码遮挡）。

这两个现象不是数学瑕疵，而是理解"上下文工程"与后面 KV Cache 的关键直觉起点。

## 二、你会观察到什么（原书要点）

- **最后一层 sink 占比高**：在 `Qwen3-0.6B` 上，最后一层 sink 常吸收每行约 **75%–85%** 的注意力；而**第 0 层**更接近"局部对角"——每个 token 主要看紧邻的前一个 token。
- **因果上三角被遮挡**：行是 Query 位置，列是 Key 位置；模型看不到"未来"，所以右上角一定是空白三角。
- **多层对比的演变**：从第 0 层的局部对角，到最后一层高度集中在首 token，注意力分布随层数明显演化——这正是信息传递与压缩的可视化证据。
- 行=Query（正在计算的 token），列=Key（被关注的 token）；颜色越亮表示注意力权重越大。

## 三、前置条件

- **Python 3.12**（官方用 `uv`；本文给 `pip` 方案）。
- **核心依赖**：`torch`、`transformers`、`matplotlib`、`numpy`。
- **模型权重**：首次运行会从 Hugging Face 自动下载 `Qwen3-0.6B`（约 **1–2 GB**），需要联网。
- **算力**：推荐 GPU / Apple MPS；**纯 CPU 也能跑短句**，只是生成阶段偏慢。
- 注意：本实验在书的"三档"里归为"轻量本地 / 必做"，但实测需要下载模型权重 + PyTorch，**本质仍是一次本地模型推理**，不是"零依赖"的轻量实验。

## 四、实验约束条件（跑之前必读）

> ⚠️ 本实验有几条"硬约束"，不满足会直接拿不到结果或触发重下。跑之前先逐条确认：

1. **模型来源：必须用 HuggingFace 原生 `Qwen/Qwen3-0.6B`，不能用 Ollama 已下好的 0.6B 替代。** 脚本走 `transformers` 直接加载 safetensors 权重；Ollama 里的是 GGUF 量化格式，是另一套加载体系——本实验不连 Ollama、也用不上它（格式、存储位置、加载方式都不同）。
2. **注意力实现：必须 `attn_implementation="eager"`（代码已写死，勿改）。** 默认的 Flash/SDPA 不 materialize 注意力矩阵，会让 `output_attentions` 返回 `None`——这是本实验拿不到热力图的最常见原因。
3. **首次运行需联网下载约 1.2 GB 权重，但仅此一次。** 下载后缓存在 `~/.cache/huggingface/hub`，之后每次只是本地读盘加载，不会重复下载。无网络 / 离线环境首次无法运行。
4. **模型名不要随意替换。** 层数、头数、sink 表现都由 `Qwen3-0.6B` 决定；换成其他模型，热力图解读与"75%–85% sink"等结论会漂移，且会触发重新下载新权重。本实验以该模型结论为准。
5. **依赖与版本**：需 `torch` / `transformers`（支持 Qwen3 的较新版本）/ `matplotlib` / `numpy`；Python 3.12（官方用 `uv`，本文给 `pip` 方案）。
6. **运行设备**：自动检测 `CUDA > MPS > CPU`；CPU 仅建议短句 + `--max-new-tokens 0`（只可视化输入、不生成），否则 `model.generate` 偏慢。
7. **缓存勿删、路径可固定**：不要人为删除 HF 缓存（删则下次重下）；若想固定权重位置，用 `--model <本地目录>` 指向已下载的副本。
8. **与 Ollama 互不干扰但也不共用**：本实验不调用 Ollama 服务；Ollama 是否在跑、占多少内存，都不影响本实验，也不为本实验省下任何下载。
9. **双模型对比模式（`--compare-models`，可选进阶）**：传入两个 HF 模型名即可并排对比热力图与 sink 占比，例如 `Qwen/Qwen3-0.6B` vs `Qwen/Qwen3-1.7B`。注意：① 两个模型都会触发各自首次下载（0.6B ~1.2GB + 1.7B ~3.4GB），仅此一次；② 脚本采用"逐模型加载→取注意力→释放"策略，内存峰值只同时存在一个模型，纯 CPU / 内存偏紧的机器也能跑，但仍建议内存 ≥ 8GB；③ **同架构对比最公平**——如 `Qwen3-0.6B` vs `Qwen3-1.7B` 同属 Qwen3，chat 模板一致，同一 prompt 分词出的 token 序列完全相同，两张热力图可直接行列对齐；若跨架构（如换成 Qwen2.5 / Qwen1.5），token 可能不一致，脚本会告警并改用第一个模型的 token 作坐标轴。默认单模型模式仍固定 `Qwen3-0.6B` 为基准。

## 五、两条路线

- **路线 A（推荐先跑通）**：克隆原书官方仓库 `bojieli/ai-agent-book`，运行完整 `chapter2/attention_visualization` 项目——含独立 CLI（`attention_cli.py`）、带工具调用的 ReAct Agent（`main.py`）、Next.js 交互前端（逐 token 热力图、统计熵等）。
- **路线 B（本仓库提供）**：自包含精简版 `main.py`，**单文件、纯 matplotlib 出图、无需前端**，专注演示"加载模型 → 取注意力 → 画热力图 → 看 sink"的核心链路，便于逐行讲透。

## 六、路线 A：官方项目运行步骤

```bash
# 1) 克隆并用第 2 章统一环境（需要 uv）
git clone https://github.com/bojieli/ai-agent-book.git
cd ai-agent-book
uv sync --locked --python 3.12 --extra ch2
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2) 最快复现：独立 CLI 直接出热力图（无需前端）
cd chapter2/attention_visualization
python attention_cli.py                                  # 默认提示词、最后一层、头平均
python attention_cli.py --prompt "北京 的 天气 怎么样" --layer 0 --head 3 --output layer0_head3.png
python attention_cli.py --prompt "用一句话解释注意力机制。" --max-new-tokens 40
python attention_cli.py --compare-layers 0 13 -1 --output layer_compare.png

# 3)（可选）交互式前端：先生成轨迹，再看网页
python agent.py          # 或 python main.py（带工具调用的 ReAct Agent）
cd frontend && npm install && npm run dev    # 打开 http://localhost:3000
```

关键文件：`attention_cli.py`（CLI 出图）、`agent.py`（基础注意力跟踪）、`main.py`（ReAct + 工具调用）、`visualization.py`（热力图/对比）、`run_attention_experiment.py`（规范化证据采集）。

## 七、路线 B：自包含精简版（main.py 详解）

本目录 `main.py` 用 `transformers` 本地加载 `Qwen3-0.6B`，单次前向拿到注意力，用 `matplotlib` 画因果热力图并统计 sink。核心结构（逐段注释见源文件）：

1. **加载模型**：`AutoModelForCausalLM.from_pretrained(..., attn_implementation="eager")`。**必须 eager**——Flash Attention 不返回注意力权重，会拿到 `None`。
2. **组装文本**：默认套用 Qwen3 的 chat 模板；`--no-chat-template` 可喂原始文本。
3. **取注意力**：`model(input_ids, output_attentions=True)`，返回的 `attentions` 是长度为层数的元组，每层形状 `[B, H, S, S]`（B=批大小、H=头数、S=序列长度）。
4. **选层/头并平均**：`-l/--layer` 选层（`-1`=最后一层），`--head` 选头（`-1`=跨头平均），得到 `[S, S]`。
5. **画热力图**：遮掉上三角（因果掩码），行列都标上 token 文本，保存 PNG。
6. **统计 sink**：打印每个 query 对第 1 个 token 的注意力占比，并给出平均值。

运行（任选其一）：

```bash
pip install torch transformers matplotlib numpy
python main.py                                                     # 默认提示词，最后一层
python main.py --prompt "北京 的 天气 怎么样" --layer 0 --head 3 --output layer0_head3.png
python main.py --prompt "用一句话解释注意力机制。" --max-new-tokens 40
python main.py --compare-layers 0 13 -1 --output layer_compare.png
python main.py --compare-models Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B -o compare.png   # 双模型并排对比（进阶）
python main.py --all-layers                                          # 单模型全部 28 层网格图（每层一格，标题标该层 sink）
python main.py --compare-models Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B --all-layers -o compare_all.png   # 两模型各出一张层网格图
```

## 八、关键机制注解

- **注意力形状**：`[B, H, S, S]`，`attentions[layer][0, head, q, k]` 表示"第 layer 层、第 head 头、由 query 位置 q 对 key 位置 k 的注意力权重"。每行（固定 q）对全部 k 求和 = 1。
- **为何要 eager**：现代推理框架默认 Flash Attention 提速，但它**不 materialize 注意力矩阵**，于是 `output_attentions` 拿不到值。教学可视化必须显式 `attn_implementation="eager"`。
- **head 平均的意义**：12/16/32 个头各自关注不同模式（有的看句法、有的看指代），平均会抹平细节但便于一眼看整体；想看差异就用 `--head` 单独抽。
- **因果掩码**：`np.triu(..., k=1)` 把上三角设为 masked，画出来就是"未来不可见"的空白三角——这是自回归模型的根本约束。
- **token 解码**：`tokenizer.decode([tid], skip_special_tokens=False)` 保留 `<|im_start|>` 等特殊符号，便于把注意力位置与文本一一对应。

## 九、实测发现：逐层 sink 演变（0.6B vs 1.7B）

> 数据来源：本仓库 `main.py --compare-models Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B --all-layers` 在本机 CPU 实测（提示词"北京 的 天气 怎么样"，17 tokens，跨头平均、取每行对首 token 的注意力占比为各层 sink）。

只看最后一层（`-1`）会得到"最后一层 sink ≈ 79%/84%"的静态结论。把全部 28 层摊开看，规律更清楚：**attention sink 不是第 1 层就有，而是从中层"突然开窍"的。**

逐层 sink 实测值（节选关键层，完整 28 层见 `compare_all_*.png`）：

| 层 | Qwen3-0.6B | Qwen3-1.7B | 阶段 |
|---|---|---|---|
| L0 | 0.111 | 0.109 | 浅层：几乎不看首 token，呈现"局部对角"（每个词只看自己/紧邻） |
| L1 | 0.126 | 0.143 | 同上 |
| L2 | 0.150 | 0.235 | 临界 |
| **L3** | **0.803** | **0.765** | **骤升**：从此层起突然死赖首 token |
| L10 | 0.670 | 0.619 | 中深层高位维持 |
| L13 | 0.709 | 0.694 | 同上 |
| L21 | 0.845 | 0.853 | 越深越重 |
| L26 | 0.868 | 0.837 | 峰值附近 |
| L27（末层） | 0.788 | 0.838 | 略回落 |

- **两模型节奏几乎一致**：L0–L2 局部 → **L3 骤变**（sink 从 ~0.1 跳到 0.7+）→ 中深层高位维持 → 末端最重。这说明 attention sink 是 Transformer **中深层的普遍行为**，不是某个模型独有，也不是"小模型才有的毛病"。
- **均值接近**：0.6B ≈ 0.690、1.7B ≈ 0.682；区别主要在末端层 1.7B 略高（L27：0.838 vs 0.788）。模型参数放大近 3 倍，但"注意力花在哪儿"的比例几乎没变。
- **生成逐层网格图**：`python main.py --compare-models Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B --all-layers -o compare_all.png`，每个模型各出一张 7×4 层网格图（每格标题标该层 sink 值，统一色标便于跨层比色）。

## 十、踩坑备注

- **attn_implementation 必须 eager**：忘了这一步会得到 `None` 或报错，是最常见坑。
- **首次下载 1–2 GB**：确保磁盘与网络；也可用 `--model` 指向已下载的本地路径避免重复拉取。
- **CPU 慢**：纯 CPU 上 `model.generate` 几十个 token 可能要等；教学建议 `--max-new-tokens 0`（只可视化输入）或短句。
- **特殊符号显示**：chat 模板会注入 `<|im_start|>` 等，热力图行列会出现这些"怪字"，属正常。
- **图像尺寸**：序列越长，图越宽；`figsize` 已按 `S*0.5` 自适应，超长序列建议先截断。

## 十一、公众号记录要点（写作线索）

- 核心洞察：**注意力不是均匀的**——模型把大量注意力"沉淀"在开头的 sink 上，这恰恰是后面"KV Cache 友好的上下文设计"的伏笔：前缀越稳定，缓存命中越好。
- 与实验 2-1 呼应：2-1 埋下"改一行系统提示词就变慢"的 KV Cache 引子；2-2 用可视化把"上下文如何被模型处理"具象化，二者共同铺垫第 2 章的上下文工程主线。
- 工程价值：可视化让"黑盒"变成"可观测"——排错 Agent 行为时，看注意力往往比看输出更先发现问题。

## 十二、参考

- 原书：第 2 章 2.2（注意力机制可视化）
- 官方代码：`bojieli/ai-agent-book` → `chapter2/attention_visualization/`
