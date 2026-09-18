# 实验 2-3：常见的错误上下文管理模式

> 对应《深入理解 AI Agent》第 2 章 · 上下文工程 · 实验 2-3（原书目录名 `kv-cache`）
> 本仓库版本：演示 KV Cache 前缀复用，**推荐本地 Ollama（`qwen3:0.6b`，信号最干净、免费、无限额）**；也支持真实 API（AMD 中国开发者平台 Radeon Cloud · Token Factory）+ 免费公共模型 `DeepSeek-V4-Flash`。

## 〇、推荐：本地 Ollama 跑法（信号最干净、免费、无限额）

本实验**首选本地 Ollama**，三个理由：

1. **模型你本来就有**：2-2 用的 `Qwen3-0.6B`（或 `Qwen3-1.7B`）已在本地，无需再下、无需联网、无 429 限额。
2. **缓存信号最干净**：Ollama/llama.cpp 采用「**从开头连续最长前缀匹配**」——前缀首个差异 token 处断开即整体失效。这正好让 8 个场景呈现教科书式的「命中 / 失效」对比。相比之下，AMD 端点是「块级哈希」缓存（vLLM 风格），稳定前缀块永远复用 → `cached` 恒为 768，反而区分不出对错模式。
3. **`cached_tokens` 直接可见**：Ollama 的 OpenAI 兼容端点会在 `usage.prompt_tokens_details.cached_tokens` 里如实回报命中 token 数，比 TTFT 推断更硬。

### 前置准备
```bash
# 1) 安装并启动 Ollama（按官方文档），确认服务在 http://localhost:11434
curl http://localhost:11434/api/tags
# 2) 拉取本地模型（2-2 已下过则可跳过）
ollama pull qwen3:0.6b
```

### 配置 .env（变量名与 AMD 路径完全一致，只改值）
```env
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=qwen3:0.6b        # 用 Ollama 的 tag，不要填 HF 的 Qwen3-0.6B
```
> ⚠️ 模型名务必用 Ollama 的 tag（`qwen3:0.6b`），不是 HF 的 `Qwen3-0.6B`，否则 `/v1/models` 对齐逻辑匹配不上会 404。

### 运行
```bash
pip install -r experiments/ch02/exp2-3/requirements.txt
python experiments/ch02/exp2-3/main.py                  # 全部 8 场景
python experiments/ch02/exp2-3/main.py --only correct   # 只跑某场景
```

### 注意事项
- **缓存随模型驻留内存**：Ollama 默认 `keep_alive` 5 分钟；分场景跑时只要间隔不太长、模型没被卸载，跨请求前缀就能复用（这正是要测的）。想保险可在 `.env` 设 `EXTRA_BODY={"keep_alive": "60m"}`（若端点支持透传），或常驻不卸载模型。
- **判定逻辑不变**：脚本仍以 `correct` 的加速比为「真实命中」基准、设 2.0x 地板区分「真实缓存命中」与「仅模型预热」。在 Ollama 的连续前缀匹配下，`dynamic_system` / `shuffled_tools` / `text_format` 等错误模式会干净地落到「失效」一档，`totally_different` 归 0，对比一目了然。
- **想换更大本地模型**：`ollama pull qwen3:1.7b` 后把 `OPENAI_MODEL` 改成 `qwen3:1.7b` 即可。模型大小不影响本实验结论。

## 一、实验目标

通过 8 个场景（含 1 个负对照），直观看到：**上下文前缀是否稳定、以及变化点放在哪里，直接决定 KV Cache 能否复用、TTFT（首 token 延迟）高低**。

- 正确的写法：系统提示 + 工具定义 + 早期历史保持固定，只有末尾的用户问题变化 → 前缀可复用 → 第二次请求更快。
- 常见错误写法：把时间戳、工具乱序、易变状态、丢历史、纯文本历史塞进前缀 → 前缀被改 → 缓存失效 → 每次都全量重算。其中「把易变内容塞到最前面」才是头号反模式（见第七节实测）。

## 二、核心概念

- **KV Cache**：模型读完输入 token 后，在每层注意力里算出的 Key/Value 张量。生成新 token 时直接复用，避免重算历史。
- **Prefix Cache / 提示词缓存**：多个请求共享**完全相同的前缀**时，这段前缀的 KV Cache 可复用。
- **缓存命中只认「前缀一致」**：内容必须出现在 prompt 开头、且 token 级别完全一致。最前面哪怕只变一个时间戳，后面再长的稳定内容也难复用。
- **TTFT（Time To First Token）**：从发出请求到收到第一个 token 的延迟。缓存命中最明显改善的就是它（省掉前缀的 prefill 重算）。

> 机制与**模型参数规模无关**：无论 0.6B 还是 27B，缓存能否生效只取决于「前缀是否稳定 + 服务商是否开启 prefix caching」。

## 三、前置条件

- Python 3.10+（本机 `my-test/.venv` 即可）
- 依赖：`openai` + `python-dotenv`（见 `requirements.txt`）
- 一个 OpenAI 兼容端点：推荐本地 Ollama（`http://localhost:11434/v1`，见「〇、推荐：本地 Ollama 跑法」），也可用 AMD Radeon Cloud · Token Factory 等真实 API；配置写在仓库根 `.env`（参考 `.env.example`）
- 联网：仅用真实 API 时需要；本地 Ollama 路径完全离线可跑

## 四、实验约束条件（跑之前必读）

1. **用真实 API 时必须配 Key（Ollama 本地路径无需真实 Key、可完全离线）**：调用 AMD 等真实 API 时，纯本地无密钥无法运行（与 2-1/2-2 的本地模型不同）；若用本仓库推荐的本地 Ollama（见「〇、推荐：本地 Ollama 跑法」），则填入 `OPENAI_API_KEY=ollama` 占位即可，模型已在本地、无需联网。无论哪条路，Key 都放根目录 `.env` 的 **`OPENAI_API_KEY`**（变量名固定，AMD 的 Key 以 `rc-` 开头），`.env` 已被 `.gitignore` 忽略，**切勿提交**。换服务商只改这一行的值，不用改变量名。
2. **模型默认 `DeepSeek-V4-Flash`（免费公共模型），可换但需 API 可见**：这是 AMD Token Factory 的免费公共文本模型（控制台卡片显示名带 -0731 后缀，但 API 的 model 字段必须用短 ID `DeepSeek-V4-Flash`；1M 上下文、支持 tool calling），开箱即用、不消耗 credits。若想换更大模型，可改 `OPENAI_MODEL` 为 `Qwen3.8-27B`（Dedicated，会消耗 credits）、`Qwen3.8-Flash-Next`（免费多模态）或 `MiniCPM5-1B/2B`（免费轻量）。模型大小不影响本实验结论。
3. **API 端点默认 AMD `https://developer.amd.com.cn/radeon/api/v1`**：这是 OpenAI 兼容端点，任何 OpenAI 兼容服务商（魔搭、DashScope、本地 vLLM 等）都能复用本脚本——把 `.env` 的 **`OPENAI_BASE_URL`** 指过去即可（也兼容旧 `AMD_BASE_URL`/`LLM_BASE_URL`）。换服务商只改值，不用改变量名。
4. **`enable_thinking` 默认关闭（即空 `{}`）**：AMD 免费模型是 Flash 系列，默认无思考模式，故默认 `EXTRA_BODY={}` 让 TTFT 只反映前缀 prefill。若用 Qwen 系列且想显式关闭思考，在 `.env` 设 `EXTRA_BODY={"enable_thinking": false}` 即可（该参数仅 Qwen3.x 识别，其它模型可能忽略或报错，按需填写）。
5. **cache% 指标「尽力而为」，且 TTFT 会被「模型预热」混淆**：原书用 Kimi K2.6，其 API 直接返回 `cached_tokens`。AMD 部分模型/端点**可能不返回缓存命中指标**（但其用量控制台会展示 KV Hit% 可作交叉验证）。本实验以 **TTFT（冷 vs 热）为通用信号**（任何 OpenAI 兼容服务商都能测），若服务商返回 `cached_tokens` 则一并打印佐证——拿不到也不影响结论。但要注意：即使**完全没有前缀缓存**，第二次请求也会因模型/GPU 刚被跑热而快约 1.3x，所以「热比冷快」不等于「命中缓存」。脚本已用 `correct` 的加速比作基准、并设 2.0x 地板来区分「真实命中」与「仅预热」。
6. **缓存是否生效取决于服务商而非本脚本**：若服务商未对该模型/请求开启自动前缀缓存，则 `correct` 也可能不见提速——这本身是一个有价值的结论（说明你需要「缓存感知的推理服务层」）。脚本会在汇总表里标注「⚠️ 未见明显提速」。
7. **网络与限流**：AMD 免费公共端点限速为每 Key 30 次/分钟、每 IP 120 次/分钟、并发 8；免费端点常驻在线、不消耗 credits。本实验 8 场景 ×（1 冷 + 3 热）≈ 32 次调用，在额度内，但同一 Key 别连跑太快。遇 429 限流稍等重试即可。
8. **依赖与版本**：`openai>=1.40`、`python-dotenv>=1.0`；Python 3.10+。`my-test/.venv` 需先 `pip install -r experiments/ch02/exp2-3/requirements.txt`。

## 五、运行

```bash
# 1) 安装依赖（首次）
pip install -r experiments/ch02/exp2-3/requirements.txt

# 2) 在仓库根目录创建 .env（参考 .env.example，变量名固定）
#    OPENAI_API_KEY=rc-你的令牌
#    OPENAI_MODEL=DeepSeek-V4-Flash
#    OPENAI_BASE_URL=https://developer.amd.com.cn/radeon/api/v1

# 3) 跑全部 8 个场景（默认 DeepSeek-V4-Flash）
python experiments/ch02/exp2-3/main.py

# 4) 常用参数
python experiments/ch02/exp2-3/main.py --model Qwen3.8-27B   # 换模型（Qwen3.8-27B 为 Dedicated，消耗 credits）
python experiments/ch02/exp2-3/main.py --only correct        # 只跑某场景
python experiments/ch02/exp2-3/main.py --warm-repeats 3      # 热请求重复次数（取最快）
```

## 六、8 个场景说明（含 1 个负对照）

| 场景 | 演示的错误 / 意图 | 变化点位置 | 实测判定（本地 `qwen3:0.6b`） |
| :-- | :-- | :-- | :-- |
| `correct` | 无（基线） | — | ✅ 完整复用（779/780） |
| `dynamic_system` | 系统提示最前塞时间戳 | 最前 | ❌ 未复用（41/805） |
| `shuffled_tools` | 工具描述进 system 并打乱 | 中段（工具块） | ⚠️ 部分复用（228/1002） |
| `dynamic_profile` | 用户状态塞进 system 最前 | 最前 | ❌ 未复用（42/804） |
| `drop_oldest` | 丢弃最早一轮历史 | 中段（历史块） | ⚠️ 部分复用（560/783） |
| `append_new` | 仅在末尾追加新一轮 | 末尾 | ✅ 完整复用（750/819） |
| `text_format` | 历史压成纯文本 + 最新轮插最前 | 最前 | ❌ 未复用（36/443） |
| `totally_different` | 负对照：完全无关的 system/历史/问题 | 最前（全局） | ❌ 未复用（36/782，≈0） |

> `cached` 列 = 热请求中从 KV 缓存复用的 token 数 / 该请求 prompt 总 token 数。判定以 `cached` 为主信号（见第八节踩坑说明：TTFT 在本地思考模型上不可靠）。

## 七、实测现象与解读（本地 `qwen3:0.6b` · Ollama 连续前缀匹配）

> 下面数字是本机实测（8 场景全跑一遍的真实值），不是理论预期。原书「5 个错误模式都该失效」的说法过于笼统，真实结果比它更有信息量。

### 7.1 一张表看懂「位置决定一切」
| 场景 | 热 cached / prompt | 判定 | 一句话 |
| :-- | :-- | :-- | :-- |
| `correct` | 779 / 780 | ✅ 完整复用 | 前缀全固定，整段跳过 prefill |
| `append_new` | 750 / 819 | ✅ 完整复用 | 只在**末尾**追加，前缀头不变 → 照样命中 |
| `shuffled_tools` | 228 / 1002 | ⚠️ 部分复用 | 变化点在**中段**（工具块），头部仍命中 |
| `drop_oldest` | 560 / 783 | ⚠️ 部分复用 | 历史块中段改写，系统提示头仍命中 |
| `dynamic_system` | 41 / 805 | ❌ 未复用 | 时间戳塞在**最前** → 前缀从第 0 个 token 分叉 |
| `dynamic_profile` | 42 / 804 | ❌ 未复用 | 用户状态塞在**最前** → 整段断裂 |
| `text_format` | 36 / 443 | ❌ 未复用 | 扁平化后最新轮插最前 → 头部改写 → 断裂 |
| `totally_different` | 36 / 782 | ❌ 未复用 | 负对照，仅共享 RUN_TAG（≈0） |

### 7.2 三个结论
1. **真正的杀手是「把易变内容塞到 prompt 最前面」**。`dynamic_system`（时间戳）、`dynamic_profile`（用户状态）、`text_format`（扁平化后最新轮插最前）都把差异放在最前 → 前缀从第 0 个 token 就分叉 → 几乎 0 复用。这正是 2-3 想教的核心反模式。
2. **改在尾部/中段是被容忍的**。`append_new` 只在末尾加一轮 → 几乎全命中；`shuffled_tools`/`drop_oldest` 在中段分叉 → 头部仍部分命中（⚠️）。所以「上下文有变化」≠「缓存失效」，要看变化点**在哪里**。
3. **负对照 `totally_different` ≈ 0**，证明 Ollama 的缓存是**按内容匹配**的，不是固定块——我们的测量是有效的，不是端点强行塞了个常量。

### 7.3 为什么最终用本地 Ollama，而不是云端 API
- 我们最初用 AMD Radeon Cloud（`DeepSeek-V4-Flash`），结果**全部 8 个场景的 `cached` 都恒为 768（= 3×256 块）**——因为 AMD 是「块级哈希」缓存：每个 256-token 块独立算哈希，某块内容曾经缓存过就复用，**跟它前面变没变无关**。系统提示+历史那几个稳定块永远命中，无论你怎么改前缀里的点。信号被彻底抹平，根本分不出对错模式。
- 本地 Ollama/llama.cpp 是「从开头连续最长前缀匹配」：首差异 token 处断开即整体失效 → 正好呈现教科书式的「命中/失效」对比，且免费、无 429、模型你本来就有。
- **机制与模型参数规模无关**：0.6B 还是 27B，缓存能否生效只取决于「前缀是否稳定 + 服务端是否开启 prefix caching」。

### 7.4 关键陷阱：TTFT 冷热差 ≠ 缓存命中
哪怕**完全没有**前缀缓存，第二次请求也会因为模型/GPU 刚被跑热而快 ~1.3x。所以「热比冷快」不能直接当作「命中缓存」。本实验以 `cached_tokens` 为主信号（服务端直接回报复用 token 数），TTFT 受思考/排队噪声影响只作旁证——在本地 Qwen3（默认开 thinking）上 TTFT 常抓不到，更凸显 `cached_tokens` 的可靠。详见第八节踩坑说明。

## 八、踩坑备注

- 未配 `OPENAI_API_KEY` 时脚本会直接报错退出并提示去建 `.env`，不会发起请求。
- `stream_options={"include_usage": True}` 才能在流式响应末段拿到 `usage`（含可能的 `cached_tokens`）。
- 若 `EXTRA_BODY` 中 `enable_thinking` 不被该模型接受，去掉或换成 `reasoning_effort` 即可（见 `.env.example` 注释）。
- 中文路径下若用 shell `cd` 进入实验目录失败，用绝对路径直接调用 `python <绝对路径>/main.py`，或在 Python 里 `os.chdir` 后再跑。

## 九、公众号记录要点（写作线索）

- 一句话钩子：为什么你的 Agent「每次都像第一次见」？因为上下文前缀每天都在偷偷变。
- 用 `correct` vs `dynamic_system` 的 TTFT 对比图，最直观。
- 强调「缓存只认前缀一致」：把时间戳/随机 ID 放最前是头号反模式。
- 落点：上下文工程的性价比——固定前缀 = 省钱 + 降延迟，是部署 Agent 第一件该做的事。

## 十、参考

- 原书实验：`bojieli/ai-agent-book` → `chapter2/kv-cache`
- AMD Radeon Cloud · Token Factory（获取 Key 与模型列表）：https://developer.amd.com.cn/radeon/tokenfactory
- AMD Radeon Cloud API 文档（OpenAI 兼容）：https://amd-aim.github.io/radeon-cloud-docs/
- 可用模型列表（`GET /v1/models`，需鉴权）：https://developer.amd.com.cn/radeon/api/v1/models
