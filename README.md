# langgraphApple —— 苹果病虫害问答智能体（LangGraph + PostgreSQL 持久化）

在 `Apple/` 目录已有能力（ResNet50 病害分类、U-Net 病害分割、Milvus 病虫害知识库）的基础上，  
用 **LangGraph** 构建一个支持**多轮对话**与**对话状态持久化**的问答智能体，  
状态检查点（checkpoint）全部落到 **PostgreSQL**。

数据库连接：`postgresql://langgraph_user:123456@localhost:5432/langgraph_db?sslmode=disable`

---

## 1. 目录结构与复用关系

```
langgraphApple/
├── config.py             # 集中配置（复用 Apple/ 的 config 组织方式，新增 DB_URL）
├── state.py              # ★ 对话状态定义 AgentState（多轮上下文的核心）
├── services.py           # ★ 复用 Apple 的模型代码（分类 ResNet50 + 分割 U-Net）
├── knowledge.py          # ★ 复用 apple_agent.ipynb 的 Milvus 向量检索（text-embedding-v3）
├── tools.py              # 把上述能力封装成 3 个 LangChain Tool
├── graph.py              # ★ LangGraph 图构建 + PostgresSaver 持久化
├── app.py                # FastAPI 服务（/chat、/history、/threads、/health）
├── init_db.py            # 初始化 PostgreSQL 检查点表
├── ingest_knowledge.py   # 把 Apple/apple.txt 灌入 Milvus
├── demo_cli.py           # 命令行多轮对话 Demo
├── selftest.py           # 组件自检
├── docker-compose.pg.yml # 一条命令拉起 PostgreSQL
├── requirements.txt
└── .env / .env.example
```

### 复用了 Apple/ 的什么

| 本模块                       | 复用来源                                  | 复用方式                                                                                                                                                                                         |
| ------------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `services.classify_image` | `Apple/apple_pest_classify_api.py`    | 复用 ResNet50 结构 + 同款预处理（Resize256→CenterCrop224→Normalize）+ 置信度阈值策略                                                                                                                           |
| `services.segment_image`  | `Apple/seg_api.py`、`Apple/predict.py` | **直接 import**：`Apple/model/unet_resnet.py` 的 `Unet`、`Apple/utils/utils.py` 的 `cvtColor/preprocess_input/resize_image`，推理与占比统计流程一致                                                            |
| `knowledge.*`             | `Apple/apple_agent.ipynb`             | 复用 Milvus（`rag_tutorial/Appledocs`、1024 维、COSINE）与 `DashScopeEmbeddings`，知识来源 `Apple/apple.txt`。⚠️ 嵌入模型已由 notebook 的 `Pro/BAAI/bge-m3`（DashScope 上已不存在）改为 **`text-embedding-v3`**（同为 1024 维） |
| `config.py`               | `Apple/` 集中配置风格                       | 统一管理路径/端口/模型/Milvus/DB                                                                                                                                                                       |

> `services.py` 通过 `sys.path.insert(0, APPLE_DIR)` 让 `model`、`utils` 成为可导入的顶层包，  
> 因此 **Apple 的模型网络与预处理代码是"零拷贝"被复用的**，Apple 侧改动会同步生效。

### ⚠️ 修正了 Apple 侧的一个类别顺序 Bug

`Apple/apple_pest_classify_api.py` 写的是 `CLASS_NAMES = ['healthy','rust','scab']`，  
但训练数据目录是 `Apple/train|test/{Scab, healthy, rust}`，`torchvision.ImageFolder`  
按**文件夹名字母序**编码标签，真实映射应为 `Scab=0 / healthy=1 / rust=2`。

在 `Apple/test` 抽样 29 张实测：

| 类别顺序                           | 准确率       |
| ------------------------------ | --------- |
| `[Scab, healthy, rust]`（本实现采用） | **79.3%** |
| `[healthy, rust, scab]`（原脚本）   | 13.8%     |

因此 `config.CLASS_NAMES = ["Scab", "healthy", "rust"]`。**建议同步修正 Apple 侧脚本。**

---

## 2. 快速开始

```bash
# 0) 环境：使用已装好依赖的 conda 环境 langchain1-2study
#    （含 langgraph 1.1.2 / langgraph-checkpoint-postgres / psycopg / torch / pymilvus / dashscope）

# 1) 启动 PostgreSQL（持久化用）
docker compose -f docker-compose.pg.yml up -d

# 2) 创建 LangGraph 检查点表
python init_db.py

# 3) （可选）启动 Milvus 并灌入知识库，用于农业知识检索
docker compose -f ../../Milvus/compose/docker-compose.yml up -d
python ingest_knowledge.py

# 4) 自检
python selftest.py

# 5) 持久化端到端验证（多轮 + 进程重启 + 查库）
python verify_persistence.py

# 6) 起服务
python app.py          # http://localhost:8002/docs
# 或命令行多轮体验
python demo_cli.py demo-1 --image ../Apple/test/rust/xxx.jpg
```

> **实机验证结果（已完成）**：Docker 启动后，`langgraph-postgres` 与 Milvus  
> 已全部拉起，上述命令 0~5 步全部跑通：
>
> - `selftest.py` → **6/6 通过**；
> - `verify_persistence.py` → **A/B/C 三项全部通过**：多轮对话跨轮复用图片、  
>   销毁内存对象后从 PostgreSQL 恢复完整历史（8 条消息）、`checkpoints` 表中  
>   该会话已落 15 个检查点；
> - `ingest_knowledge.py` → 知识库写入 Milvus 成功，检索可命中，  
>   第三轮"怎么防治"已能基于知识库内容作答。

---

## 3. 智能体的对话流程

图是一张标准 **ReAct 循环**：

```
        ┌─────────┐  最后一条 AI 消息含 tool_calls  ┌─────────┐
        │  agent  │ ───────────────────────────────▶ │  tools  │
        └─────────┘                                  └─────────┘
             ▲                                            │
             └────────────── ToolMessage 结果回填 ─────────┘
             │
        无 tool_calls
             ▼
            END
```

1. **START → `agent`**  
   `agent` 节点把消息拼成固定三段交给绑定了工具的 LLM：
   ```
   [ 静态 system prompt ] + [ 持久化的历史消息 ] + [ 本轮实时上下文 ]
   ```
   - **静态 system prompt**（`graph.SYSTEM_PROMPT`）：只描述角色、工具用途、工作与输出要求，  
     **不含任何逐轮变化的内容**；
   - **历史消息**：从 PostgreSQL 恢复的全部对话；
   - **本轮实时上下文**：由 `_build_live_context()` 生成，告知模型"本会话有没有图片、预分类结果是什么"，  
     **追加在末尾**，且不写回 `state["messages"]`（不污染持久化历史）。
   > **为什么实时上下文不能放进 system prompt？**  
   > 通义千问的 Context Cache 自动开启，但要求**前缀重复 ≥1024 token** 才命中，  
   > 规则是"重复内容在前、变化内容在后"。system prompt 位于序列最前面，  
   > 一旦它每轮都变，最长公共前缀会从第一个 token 就分叉，**后面整段历史的缓存会一起作废**。  
   > 挪到末尾后，`system + 历史` 构成的最长公共前缀在每轮之间完全稳定，缓存才有机会命中。
2. **条件边 `tools_condition`**  
   看 `agent` 输出的 `AIMessage` 是否带 `tool_calls`：
   - 有 → 走 `tools` 节点（`langgraph.prebuilt.ToolNode` 自动执行工具）；
   - 没有 → 走 `END`，该条 AI 消息就是最终回答。
3. **`tools` → `agent`**  
   工具结果作为 `ToolMessage` 追加进消息列表，回到 `agent` 继续推理，直到不再需要工具。

三个工具的分工（对应 Apple 的三项能力）：

| 工具                       | 触发场景           | 底层能力              |
| ------------------------ | -------------- | ----------------- |
| `identify_apple_disease` | "这是什么病 / 健康吗"  | Apple ResNet50 分类 |
| `analyze_apple_disease`  | "严重吗 / 病斑占比多少" | Apple U-Net 分割    |
| `search_apple_knowledge` | "怎么防治 / 用什么药"  | Apple Milvus RAG  |

**两个图片工具不接收路径参数**，而是用 `InjectedState` 从对话状态里读 `image_path`：

```python
@tool
def identify_apple_disease(state: Annotated[AgentState, InjectedState]) -> str:
    image_path = (state or {}).get("image_path")   # ← 从状态取，而不是让 LLM 传
```

这样 LLM 只需决定"要不要看图"，不必知道服务端临时文件路径；  
且因为状态被持久化，**第一轮传图、第二轮只问"严重吗"也能命中同一张图**。

---

## 4. 状态管理方式

状态定义在 `state.py`，是一个带 reducer 的 `TypedDict`：

```python
class AgentState(TypedDict, total=False):
    messages:       Annotated[list, add_messages]   # 累积多轮消息
    thread_id:      Optional[str]                   # 会话 ID
    user_id:        Optional[str]                   # 用户维度隔离
    image_path:     Optional[str]                   # 当前会话的图片（跨轮复用）
    classification: Optional[dict]                  # 预分类结果缓存
    retrieved:      Optional[list[dict]]            # 检索片段
```

两种字段语义：

- **`messages` 配了 `add_messages` reducer** → 新消息**追加**到历史，而不是覆盖。这是多轮记忆的基础。
- **其余字段没有 reducer** → "最后一次写入生效"，且**本轮不传该字段时就保持上一轮的值**。  
  所以 `app.py` 只在用户本轮真的上传图片时才写 `image_path`，  
  后续轮次不传即自动沿用——这就是"跨轮复用同一张图"的实现原理。

每轮调用只需传入"增量"（本轮用户消息 + 可选的新图片），LangGraph 会自动把历史状态合并进来：

```python
graph.invoke(
    {"messages": [HumanMessage("严重吗？")]},          # 只传增量
    config={"configurable": {"thread_id": "demo-1"}},  # 指定会话
)
```

---

## 5. 持久化机制（PostgreSQL）

编译图时传入 checkpointer 即开启持久化：

```python
pool = ConnectionPool(conninfo=DB_URL, min_size=1, max_size=10,
                      kwargs={"autocommit": True, "prepare_threshold": 0})
checkpointer = PostgresSaver(pool)
checkpointer.setup()             # 幂等建表
app = builder.compile(checkpointer=checkpointer)
```

工作机制：

1. **`thread_id` = 会话主键。** 每次调用都带上  
   `config={"configurable": {"thread_id": ...}}`，同一个 `thread_id` 就是同一段对话。
2. **每个 super-step 落盘。** LangGraph 在**每个节点执行完成后**把整份 `AgentState`  
   序列化写入 PostgreSQL，而不是等整轮结束才写。因此即使中途崩溃，也能从最近检查点恢复。
3. **调用前自动加载。** 用同一个 `thread_id` 再次 `invoke` 时，  
   checkpointer 先按 `thread_id` 读出最近的状态快照恢复历史，再合并本轮输入。
4. **表结构**（由 `setup()` 自动创建，`init_db.py` 可单独初始化）：
   | 表                       | 作用                                               |
   | ----------------------- | ------------------------------------------------ |
   | `checkpoints`           | 每个 super-step 的状态快照（按 thread_id + checkpoint_id） |
   | `checkpoint_blobs`      | 状态中的大字段（如消息内容）                                   |
   | `checkpoint_writes`     | 各节点的写入记录，用于恢复/重放                                 |
   | `checkpoint_migrations` | 迁移版本                                             |
5. **可读性。** `/history/{thread_id}` 会从 PostgreSQL 恢复到内存后按角色还原成消息列表；  
   `/threads` 直接查 `checkpoints` 表列出全部会话。

验证持久化的推荐方式：起一次 `demo_cli.py demo-1` 聊两句，退出进程；  
再 `demo_cli.py demo-1` 进来问"刚才那张图是什么病"，智能体仍能答上——  
说明状态确实是从 PostgreSQL 恢复的，而不是存在内存里。

---

## 6. 接口说明

### POST /chat

```json
{
  "question": "这张叶子是什么病？",
  "thread_id": "demo-1",          // 不传则新建会话
  "user_id": "u-001",
  "image_base64": "<可选，本轮上传的图片>"
}
```

返回：

```json
{
  "status": "success",
  "thread_id": "demo-1",
  "answer": "……",
  "class": "rust", "confidence": 0.9541, "all_probs": {...},
  "metadata": {"has_image": true}
}
```

### 其他

- `GET /history/{thread_id}` —— 会话历史消息（含工具调用记录）
- `GET /threads` —— 全部会话列表（含检查点数量）
- `GET /health` —— 健康检查（DB 连接、会话数、Milvus 状态）

---

## 7. 容错设计

- **模型懒加载 + 单例**：进程内只加载一次，分类/分割模型缺失不影响服务启动，只让对应工具返回错误提示。
- **Milvus 降级**：连不上时 `search_apple_knowledge` 返回"未检索到"，智能体仍可凭自身知识回答。
- **PostgreSQL 快速失败**：连接池 `timeout=10`，连不上时抛出带修复指引的明确错误，不会静默卡死。

---

## 8. 响应速度优化（已落地）

延迟实测归因（`qwen-plus`、CPU 推理）表明瓶颈**几乎全在大模型生成**：

| 环节                    | 实测耗时               |
| --------------------- | ------------------ |
| LLM 生成一份 1900 字"防治方案" | **53.1s**（≈36 字/秒） |
| 分割 U-Net（CPU）         | 1.9s               |
| 分类 ResNet50（CPU）      | 0.5s               |
| RAG 检索（含 embedding）   | 0.8s               |

即 `响应时间 ≈ 输出字数 ÷ 36`，本地模型与检索加起来不到 6 秒。因此优先动"生成层 + 输入层"。

### 已落地的两项改动

**① 输出长度约束（生成层）** —— `SYSTEM_PROMPT` 末尾新增「输出要求」段：

- 默认控制在 **200 字以内**，仅当用户明确要求"详细/全面/分点说明"时才展开；
- 用短句和 `- ` 要点罗列，**禁用 Markdown 表格 / 标题层级 / 代码块**；
- 不复述工具返回的原始数值与字段名。

**② 前缀缓存修复（输入层）** —— 把 `image_hint` / `cls_hint` 从 system prompt 移到消息**末尾**，  
由 `_build_live_context()` 生成。system prompt 变成完全静态常量，`system + 历史` 的公共前缀  
每轮之间字节级一致，Context Cache 才有机会命中（命中部分按输入价 20% 计费且免重复计算）。

### 实测效果

| 指标                | 改前              | 改后               |
| ----------------- | --------------- | ---------------- |
| 三轮对话总耗时           | 单轮即 53.1s       | **19.1s（3 轮合计）** |
| 平均单轮              | 53.1s           | **6.4s**         |
| 平均回答长度            | ≈1900 字         | **170 字**        |
| system prompt 稳定性 | 每轮变化            | **三轮字节完全一致**     |
| 前缀缓存可用性           | 第 1 个 token 即分叉 | **公共前缀 100% 稳定** |

三轮对话中工具调用（识别 → 分割 → 知识检索）与跨轮状态复用（第 2 轮不传图仍复用第 1 轮的图）  
均正常，`selftest.py` 6/6、`verify_persistence.py` A/B/C 全部通过。

### 下一步可继续做的（未落地）

| 动作                                           | 层级  | 预期                         |
| -------------------------------------------- | --- | -------------------------- |
| 开流式输出（后端 SSE + 前端逐段渲染）                       | 感知层 | 感知延迟 53s → 0.7s（首字仅 0.72s） |
| 换 `qwen-flash` / `qwen-turbo`                | 生成层 | 同长度快 2–3 倍                 |
| `lifespan` 里预热模型与 Milvus 连接                  | 工程层 | 省掉首位用户的 ≈3.8s 冷启动          |
| 图片工具复用 `state["classification"]`，避免重复推理      | 编排层 | 每轮省 ≈0.5s                  |
| `pre_model_hook` 裁剪历史 + `recursion_limit` 封顶 | 输入层 | 长对话不劣化                     |

### 已启用 GPU（`torch 2.6.0+cu124`）

本机为 **RTX 3050 Laptop（4GB）**，`langchain1-2study` 环境已换成 CUDA 版 torch，  
`torch.cuda.is_available() = True`，`services.py` 无需任何改动（本来就是  
`torch.device("cuda" if torch.cuda.is_available() else "cpu")`）：

| 环节                     | CPU   | GPU 预热后   | 首次（含 CUDA 预热） |
| ---------------------- | ----- | --------- | ------------- |
| 分类 ResNet50            | 0.50s | **0.17s** | 0.46s         |
| 分割 U-Net（5312×2988 大图） | 1.90s | **0.31s** | 0.45s         |
| 两模型冷加载                 | 7.5s  | **2.9s**  | —             |

CUDA 环境自检：`torch 2.6.0+cu124` / `torchvision 0.21.0+cu124` / cuDNN 90100 / compute capability 8.6，  
模型推理与 `torch.matmul` 均已实测在 `cuda` 上执行，双模型常驻显存约 **171 MiB / 4096 MiB**。

> ⚠️ **4GB 显存的并发约束**：模型占用很小，但**不要同时跑两个加载模型的 Python 进程**  
> （例如服务已在 8002 运行，又另开一个终端跑 `selftest.py`），两个 CUDA 上下文叠加会触发  
> `CUDA error: out of memory`。串行执行一切正常。
>
> `requirements.txt` 顶部已加 `--extra-index-url https://download.pytorch.org/whl/cu124`，


> 直接 `pip install -r requirements.txt` 即可保持 GPU 版（**不要加 `--force-reinstall`**）。  
> CPU-only 机器请删掉该行，并把 torch/torchvision 换成不带 `+cu124` 的同版本号。
>
> 依赖状态：`pip check` 仅剩一条 `unstructured-inference` 要求 `torch>=2.10` 的告警 ——  
> Windows + Python 3.13 的 cu124 索引最高只提供 torch 2.6.0，属**不可避免的取舍**，  
> 且本项目不依赖该包。`numpy` 已钉在 `2.4.4`（而非 2.5.x），以保证 `numba` 可用、环境零冲突。

---

## 9. 为什么只需要一个 8002 端口

原架构（见 `D:/python学习/start_all.bat`）是**三个独立进程**，智能体作为客户端用 `requests.post` 调另外两个：

| 端口   | 文件                                                    | 作用                       | 运行环境                 |
| ---- | ----------------------------------------------------- | ------------------------ | -------------------- |
| 8003 | `D:/python学习/Apple/seg_api.py`                        | 分割 `POST /segment_apple` | `u-net`（cu118 torch） |
| 8000 | `D:/code/langchain1.2Study/Apple/apple_pest_model.py` | 分类 `POST /predict_apple` | `langchain1-2study`  |
| 8002 | `D:/code/langchain1.2Study/Apple/agent_api.py`        | LangChain 智能体            | `langchain1-2study`  |

> 另外 `D:/python学习/Apple/apple_pest_classify_api.py` 是**更早一版**独立分类服务（注释里写 `--port 8001`），  
> 已被 8000 取代，属历史遗留 —— 所以容易记成"8001/8002/8003"。

`langgraphApple` 只保留 8002，原因是**复用方式从「跨进程 HTTP 调用」改成了「同进程直接 import」**：

```python
sys.path.insert(0, APPLE_DIR)
from model.unet_resnet import Unet                       # 分割模型
from utils.utils import cvtColor, preprocess_input, resize_image
```

前提是 `langchain1-2study` 里 torch / torchvision / opencv / pymilvus 已齐备，  
一个进程就能装下"智能体 + 分类 + 分割 + RAG"。

**收益**：3 个进程 → 1 个；省掉 2 次 HTTP 往返与图片重复传输（原来同一张图要分别 POST 给 8000 和 8003，  
每次都要 base64 序列化再落盘）；启动只需一条命令；分类结果可直接写进 LangGraph 状态被后续轮次复用。

**代价**：单点故障 —— 模型 OOM 或卡死会拖垮整个服务（旧架构下只会挂掉 8003，智能体仍能降级返回"分割失败"）；  
冷启动更重（两个模型首次加载 ≈8.4s）；无法独立扩缩容，也不能给分类和分割用不同版本的 torch。

> ⚠️ 旧的 `agent_api.py` **同样监听 8002**，与 `app.py` 不能同时启动。
