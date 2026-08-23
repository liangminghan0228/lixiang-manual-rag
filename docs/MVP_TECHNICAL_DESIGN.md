# 理想汽车手册 RAG MVP 技术方案（历史基线）

> 当前实现已经超过本 MVP 范围。请优先阅读 [当前项目实现架构](CURRENT_IMPLEMENTATION_ARCHITECTURE.md)；本文用于回看最小闭环如何演进。

## 1. 文档定位

本文是当前 MVP 的唯一开发依据，目标是用最少模块跑通可验证的 Dense RAG 闭环。完整模块化架构和生产能力均不属于本阶段交付范围。

MVP 要回答三个问题：

1. 能否稳定地把理想汽车手册转换成可追踪的 chunk？
2. BGE-M3 + Qdrant 能否召回支持问题的正文？
3. LLM 能否只依据已召回正文生成带来源的答案？

## 2. 范围

### 2.1 本期实现

- 一个固定的理想汽车手册快照；
- HTML 目录和 topic 页面采集；
- 标题/段落结构解析；
- Heading Chunking；
- 本地 BGE-M3 Dense Embedding；
- Qdrant Cosine Top-K 检索；
- OpenRouter 或 Mock Generator；
- `/v1/retrieve`、`/v1/chat`、`/health`；
- 答案引用、无证据拒答和分阶段耗时；
- 单一50题数据集；MVP阶段用`--limit`执行检索F1@5和MRR@10。

### 2.2 本期不实现

- Reranker、BM25、Hybrid Retrieval；
- Query Rewrite、Multi-Query、Agent；
- OCR、VLM 和图片正文理解；
- 多车型、多版本选择和用户权限；
- 异步任务、消息队列、缓存；
- Prometheus、Grafana、Locust 和百万向量压测；
- Web 前端。

## 3. 固定技术选型

| 层 | MVP 选型 | 说明 |
|---|---|---|
| 运行环境 | Python 3.11 + uv | 避免使用系统 Python 3.9 |
| API | FastAPI + Pydantic | 参数校验和 HTTP 接口 |
| 配置 | pydantic-settings + YAML | `.env` 保存密钥，YAML 保存实验参数 |
| 抓取 | httpx | 静态 HTML，无需浏览器渲染 |
| 解析 | BeautifulSoup4 + lxml | 解析目录、topic 和正文结构 |
| Embedding | FlagEmbedding + `BAAI/bge-m3` | 本地生成文档和问题向量 |
| 向量库 | Qdrant Server + qdrant-client | Docker 单节点运行 |
| LLM | OpenRouter OpenAI-compatible API | 无 Key 时使用 Mock Generator |
| 测试 | pytest + pytest-asyncio | 单元测试和最小集成测试 |
| 代码质量 | ruff | 格式化与静态检查 |

第一版不使用 LangChain。所有步骤显式实现，方便观察数据结构和各阶段耗时。

## 4. MVP 架构

完整图见 [MVP 架构](MVP_ARCHITECTURE.md)，运行链路固定为：

```text
离线：
index.html/topic.html
  -> crawler
  -> parser
  -> HeadingChunker
  -> BGE-M3 embed_documents
  -> Qdrant upsert

在线：
question
  -> BGE-M3 embed_query
  -> Qdrant Top-5
  -> 证据阈值与 Top-3
  -> OpenRouter/Mock
  -> answer + citations + timings_ms
```

只保留五个可替换接口：`Chunker`、`Embedder`、`VectorStore`、`Retriever`、`Generator`。接口和默认实现放在同一业务模块，不建设独立插件框架。

## 5. 模块职责

### 5.1 Ingestion

`crawler.py`：

- 请求不带查询参数的 `index.html`；
- 从 `ul#manual-nav` 递归解析目录；
- 从 `a[data-content]` 生成 topic URL 并去重；
- 并发 2、最大 2 请求/秒、超时 20 秒、最多重试 3 次；
- 保存原始 HTML、内容哈希和 `manifest.jsonl`；
- 支持失败记录和断点续抓。

`parser.py`：

- 读取 `main article[role="article"]`；
- 保留标题层级、段落、列表、表格和 warning/caution；
- 图片只保存 URL、alt 和章节，不下载、不做 OCR；
- 输出标准 `Document`，不直接生成向量。

`chunker.py`：

- 优先按 topic 内标题切分；
- 目标 400～600 中文字；
- 超长章节按段落切分，重叠 80 字；
- 列表和警告块不可从中间切断；
- chunk 前拼接车型、breadcrumb 和 section title。

`service.py`：串联采集、解析、分块、Embedding 和 Qdrant Upsert，并输出统计摘要。

### 5.2 Retrieval

`embedder.py`：统一提供 `embed_documents()` 和 `embed_query()`，两条链路必须使用同一模型及归一化配置。

`vector_store.py`：封装 collection 创建、payload index、批量 upsert、查询和健康检查；其他模块不能直接 import `QdrantClient`。

`dense.py`：组合 Embedder 与 VectorStore，返回 Top-K `SearchResult`。第一版不做查询改写、融合和重排。

### 5.3 Generation

`service.py`：

- 接收 DenseRetriever 返回的候选；
- 根据评测集校准后的 `min_score` 过滤；
- 最多选择 Top-3 正文；
- 没有合格正文时直接拒答；
- 构造带编号证据的 Prompt；
- 返回答案、引用和阶段耗时。

Generator 不能自行查询 Qdrant，只能读取 ChatService 传入的正文。

### 5.4 API

FastAPI 只负责请求/响应转换，不包含抓取、检索或 Prompt 逻辑。手册入库通过 CLI 执行，不提供上传 API。

## 6. 核心数据结构

```python
class Document(BaseModel):
    document_id: str
    title: str
    breadcrumb: list[str]
    source_url: str
    sections: list[Section]
    metadata: dict[str, Any]

class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    section_path: list[str]
    source_url: str
    metadata: dict[str, Any]

class SearchResult(BaseModel):
    chunk: Chunk
    score: float
    rank: int

class Citation(BaseModel):
    index: int
    chunk_id: str
    title: str
    source_url: str

class Answer(BaseModel):
    text: str
    citations: list[Citation]
    timings_ms: dict[str, float]
```

`chunk_id`：

```text
UUID5(snapshot_id + topic_id + section_path + chunk_text_hash)
```

它同时用作 Qdrant point ID 的来源，保证重复入库幂等。

## 7. Qdrant 设计

Collection：

```text
lixiang_mvp_bge_m3_v1
```

Collection 配置：

- 一个 Dense Vector；
- vector size 从 Embedder 实际输出读取并校验；
- distance 使用 Cosine；
- 不与其他 Embedding 模型共用 collection。

Point：

```json
{
  "id": "stable-point-id",
  "vector": [0.01, -0.02],
  "payload": {
    "chunk_id": "...",
    "snapshot_id": "20250916141802",
    "manual_id": "i8_SS3_MAX_2025-002",
    "vehicle_model": "理想i8",
    "topic_id": "topic-2025-7386992B",
    "title": "安全驾驶",
    "section_path": ["安全驾驶", "正确的坐姿"],
    "text": "正文",
    "source_url": "https://...",
    "content_hash": "sha256:..."
  }
}
```

只为 `snapshot_id`、`manual_id`、`vehicle_model`、`topic_id` 创建 keyword payload index。`text` 和 URL 只返回，不创建索引。

## 8. 接口契约

### 8.1 检索

```http
POST /v1/retrieve
Content-Type: application/json

{
  "question": "驾驶前需要检查什么？",
  "top_k": 5
}
```

响应：

```json
{
  "results": [
    {
      "chunk": {
        "chunk_id": "...",
        "text": "...",
        "title": "安全驾驶",
        "source_url": "https://..."
      },
      "score": 0.78,
      "rank": 1
    }
  ],
  "timings_ms": {
    "embedding": 12.4,
    "qdrant": 4.8,
    "total": 18.1
  }
}
```

### 8.2 问答

```http
POST /v1/chat
Content-Type: application/json

{
  "question": "驾驶前需要检查什么？"
}
```

响应必须包括 `text`、`citations`、`evidence`、`timings_ms` 和 `refused`。回答中的 `[1]` 必须能映射到响应中的第一个 citation。

### 8.3 健康检查

```http
GET /health
```

返回 API、Embedding 是否加载、Qdrant 是否可访问、Generator 是否为 mock。

## 9. 配置

当前配置文件：[mvp.yaml](../configs/mvp.yaml)。环境变量：

```env
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openrouter/free
QDRANT_URL=http://localhost:6333
```

优先级：环境变量覆盖 YAML。API Key 为空时自动装配 Mock Generator，保证采集和检索开发不被外部 API 阻塞。

`min_score`不预先写死。当前统一评测不再保留独立阈值校准脚本：检索使用F1@5和MRR@10，生成使用Ragas忠实度、答案相关性、完整性，不可回答问题单独统计拒答正确率。

## 10. 项目目录

```text
app/
  models.py
  settings.py
  wiring.py
  ingestion/
    crawler.py
    parser.py
    chunker.py
    service.py
  retrieval/
    embedder.py
    vector_store.py
    base.py
    dense.py
  generation/
    base.py
    openrouter.py
    mock.py
    service.py
  api/
    main.py
    schemas.py
configs/mvp.yaml
data/
  raw/
  normalized/
  eval/rag_eval_v2.jsonl
tests/
  unit/
  integration/
scripts/evaluate.py
compose.yaml
pyproject.toml
```

## 11. 开发顺序

### M0：工程初始化

- `uv` 创建 Python 3.11 项目；
- 添加依赖、ruff、pytest；
- 创建 Qdrant `compose.yaml`；
- 实现配置加载和领域模型。

### M1：采集与解析

- 实现 crawler、manifest、parser 和 HeadingChunker；
- 用保存的 HTML fixture 完成单元测试；
- 产出 raw 与 normalized 数据。

### M2：Embedding 与检索

- 实现本地 BGE-M3；
- 创建 Qdrant collection 和 payload index；
- 完成幂等入库和 `/v1/retrieve`。

### M3：问答闭环

- 实现 Mock/OpenRouter Generator；
- 完成证据阈值、Prompt、引用和 `/v1/chat`；
- 加入无证据拒答。

### M4：冒烟评测

- 从唯一50题数据集选择少量问题快速验证；
- 输出F1@5、MRR@10和阶段耗时；
- 根据结果校准 `min_score`。

## 12. 验收标准

MVP 完成必须同时满足：

1. 同一快照重复入库，Qdrant point 数量不增加；
2. 抓取失败、空正文和重复 topic 均有统计；
3. `/v1/retrieve` 返回正文、相似度、rank 和原始 URL；
4. `/v1/chat` 的关键结论带引用，引用能打开原 topic；
5. 无合格证据时不调用真实 LLM，直接拒答；
6. 统一50题数据集可重复执行并输出F1@5、MRR@10；
7. 响应分别记录 Embedding、Qdrant、LLM 和总耗时；
8. 单元测试不依赖网络、Qdrant 或 OpenRouter。

MVP 完成不代表达到生产性能或回答准确率目标，只代表最小闭环、证据边界和评测入口已经建立。

## 13. MVP 之后

只有当离线结果暴露明确问题时再扩展：

- 专有名词召回差：增加 BM25/Hybrid；
- Top-K 中正确证据排序靠后：增加 Reranker；
- 无依据回答仍多：加强 EvidenceSelector 和回答评测；
- 延迟或吞吐成为问题：增加 Locust、Prometheus 和 Grafana；
- 文本无法回答图片问题：再增加图片下载、OCR/VLM。

后续完整演进方案见 [完整技术方案](TECHNICAL_DESIGN.md) 和 [完整模块化架构](MODULAR_ARCHITECTURE.md)。
