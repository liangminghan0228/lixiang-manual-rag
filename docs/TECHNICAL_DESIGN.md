# 理想汽车手册中文 RAG 完整技术方案

> 本文保留完整目标、选型和验收口径。主要工程能力已于 2026-08-16 实现；当前代码结构、实测指标和剩余边界以 [当前项目实现架构](CURRENT_IMPLEMENTATION_ARCHITECTURE.md) 为准，MVP 文档仅作为历史基线。

## 0. 实现状态摘要

| 完整方案能力 | 当前状态 |
|---|---|
| 官方车型目录发现与批量采集 | 已实现；当前目录发现39份手册、7,645个唯一 topic |
| 结构解析、标题分块、冻结快照 | 单车型已产出196 documents / 848 chunks；全车型任务可断点执行 |
| BGE-M3 + Qdrant Dense 检索与 payload filter | 已实现并通过真实 Qdrant 集成测试 |
| BM25、Hybrid RRF、BGE Reranker | 已实现，提供独立实验配置与离线报告 |
| EvidenceSelector、chunk 级引用、引用修复/拒答 | 已实现 |
| 唯一50题宽类型评测集 | 已生成；42条自动参考答案待人工复核 |
| 检索F1/MRR + Ragas生成评测 | 已实现统一入口；完整真实Judge实验待执行 |
| 增量抓取、跨手册 Embedding 缓存、陈旧 point 删除、Run Manifest | 已实现；首轮跨年款缓存命中率75.4% |
| Locust、Prometheus、Grafana、隔离合成数据工具 | 已实现并完成 5 并发短基线 |
| 单次请求 RAG Trace 调试台 | 已实现；SSE 展示 Embedding 摘要、召回、精排、证据、LLM 和引用链路 |
| 固定真实 LLM 的答案评测、百万点容量结论 | 工具已具备，正式实验尚未执行 |

## 1. 项目目标

从零完成一条可解释、可评测的中文 RAG 链路：

```text
理想汽车公开手册
  -> 静态 HTML 采集与版本快照
  -> 结构化解析与分块
  -> BGE-M3 向量化
  -> Qdrant 语义召回
  -> BGE Reranker 精排
  -> OpenRouter LLM 基于证据回答
  -> 离线质量评测与并发压测
```

项目的学习重点是分块、向量检索、重排、证据约束、质量评测和性能定位。第一版不引入 LangChain、Agent、消息队列、多租户权限和复杂前端。

模块边界、接口契约、配置装配和对比实验隔离规则见 [模块化架构设计](MODULAR_ARCHITECTURE.md)。本文中的技术选型是默认基线实现，而不是流水线对具体厂商的硬依赖。

最小版本的历史设计见 [最小 RAG Demo 架构](MVP_ARCHITECTURE.md)；当前实现已包含 Reranker、Hybrid Retrieval、监控和并发压测入口。

## 2. 数据源核验与获取方案

### 2.1 已核验的页面结构

目标页面：

```text
https://manuals.lixiang.com/zh-cn/W022025ULTRA/20250916141802/index.html?content=topic-2025-7386992B.html
```

2026-08-15 实际检查结果：

- 页面通过普通 HTTP GET 即可获取，不依赖浏览器执行 JavaScript。
- `?content=...` 只指定首次展示的 topic；采集入口应使用不带查询参数的 `index.html`。
- 页面标题为“理想i8”，目录元数据 `data-number` 为 `i8_SS3_MAX_2025-002`。目录名和 URL 路径可能不一致，因此车型、版本应以页面元数据为准并保留原始 URL。
- 目录位于 `ul#manual-nav`，叶子节点通过 `a[data-content]` 指向同目录下的 topic HTML。
- 当前 HTML 中有 294 个 `data-content` 引用、196 个唯一 topic 文件。数量只是当前快照，不应写死。
- topic 正文位于 `main article[role="article"]`；子章节使用嵌套 `article.topic` 和 `h1`～`h6`。
- 正文包含段落、列表、表格、提示/警告、站内交叉链接，以及 `img/*.webp` 图片。

### 2.2 采集流程

```text
index.html
  -> 解析车型和手册版本
  -> 遍历目录并生成 breadcrumb
  -> 将 data-content 转换成绝对 URL
  -> 按绝对 URL 去重
  -> 低并发抓取 topic HTML
  -> 保存原始快照和 manifest
  -> 解析结构化 topic JSONL
```

采集器固定采用 `httpx + BeautifulSoup4 + lxml`：

1. 请求无查询参数的 `index.html`，记录 `source_url`、抓取时间、`ETag`、`Last-Modified`、内容 SHA-256、车型与版本。
2. 递归解析 `ul#manual-nav`。目录文字取 `.nav-text`，正文文件取 `a[data-content]`，沿父节点生成 breadcrumb。
3. 对绝对 topic URL 去重后抓取。默认并发 2、全局不超过 2 请求/秒、超时 20 秒、最多重试 3 次，并采用指数退避。
4. 每成功抓取一页立即写入 `manifest.jsonl`，支持断点续抓；失败写入状态和错误，不静默跳过。
5. 增量更新优先发送 `If-None-Match`/`If-Modified-Since`；内容哈希未变化时不重新向量化。
6. 第一阶段不下载图片，只保留图片绝对 URL、alt 和所在章节；第二阶段如需要图文问答，再下载图片并引入 OCR/VLM。

原始数据布局：

```text
data/raw/{manual_id}/{snapshot_id}/
  index.html
  topics/{topic_file}.html
  manifest.jsonl
  metadata.json
```

合规边界：该站点的 `/robots.txt` 当前返回 404，这不等于自动获得批量复制授权。项目仅用于学习时仍应低频访问、标明来源、不公开再分发原始手册；扩大抓取范围或商业使用前需核对网站条款和版权授权。不得绕过登录、鉴权、验证码或访问控制。

### 2.3 正文解析

每个 topic 输出一条结构化记录：

```json
{
  "manual_id": "i8_SS3_MAX_2025-002",
  "snapshot_id": "20250916141802",
  "topic_id": "topic-2025-7386992B",
  "title": "安全驾驶",
  "breadcrumb": ["用车场景", "出行准备", "安全驾驶"],
  "source_url": "https://.../topic-2025-7386992B.html",
  "sections": [],
  "content_hash": "sha256:..."
}
```

解析规则：

- 保留标题层级、段落、列表顺序、表格文本、站内链接和图片引用。
- 将 `note warning/caution` 转换成带类型的文本块，不能丢失警告语义。
- 站内链接转换为绝对 URL，并尽可能关联到对应 `topic_id`。
- 页面外壳、菜单、搜索框、样式和脚本不进入知识正文。
- 解析后检查 topic 数、空正文数、重复 URL 数和解析失败数；失败率非零时生成报告。

## 3. 默认基线技术选型

| 模块 | 固定选型 | 说明 |
|---|---|---|
| 语言/API | Python 3.11、FastAPI、Pydantic | 单体服务，便于学习和分段计时 |
| HTTP/解析 | httpx、BeautifulSoup4、lxml | 抓取静态目录和 topic HTML |
| Embedding | `BAAI/bge-m3`、FlagEmbedding | 中文语义向量；文档与问题使用同一模型 |
| 向量数据库 | Qdrant、qdrant-client | Docker 单节点；Cosine 距离与 payload 过滤 |
| Reranker | `BAAI/bge-reranker-v2-m3` | 将召回 Top-10 精排为 Top-3 |
| LLM | OpenRouter OpenAI-compatible API | 开发阶段可用 `openrouter/free`；正式评测必须固定模型 |
| 配置 | pydantic-settings、`.env` | 密钥不进入代码和 Git |
| 指标 | prometheus-client | 记录分阶段延迟、错误率和请求量 |
| 压测 | Locust | 分离检索压测与完整问答压测 |
| 部署 | Docker Compose | `api + qdrant + prometheus + grafana` |

第一版不使用 LangChain：采集、分块、检索和提示词链路均显式实现，便于理解数据如何流动和定位耗时。流水线只依赖 Protocol，表中的具体实现通过配置和显式 Registry 装配。

## 4. 分块与 Qdrant 数据设计

### 4.1 分块规则

- 优先以 topic 内的标题章节为自然分块边界。
- 每块目标 400～600 中文字；小于 150 字且同属一个父标题的相邻块可合并。
- 超长章节按段落继续切分，重叠 80 字；列表、表格、警告块不从中间截断。
- chunk 文本前拼接车型、breadcrumb 和当前标题，提升脱离上下文后的可检索性。
- `chunk_id = SHA256(manual_id + snapshot_id + topic_id + section_path + text_hash)`，保证可追踪和幂等写入。

### 4.2 Qdrant collection

collection 名称：`lixiang_manual_chunks_v1`。

每个 point 保存一个 dense vector，payload 为：

```json
{
  "chunk_id": "...",
  "manual_id": "i8_SS3_MAX_2025-002",
  "snapshot_id": "20250916141802",
  "vehicle_model": "理想i8",
  "topic_id": "topic-2025-7386992B",
  "title": "安全驾驶",
  "section_path": ["安全驾驶", "正确的坐姿"],
  "breadcrumb": ["用车场景", "出行准备", "安全驾驶"],
  "text": "...",
  "source_url": "https://...",
  "content_hash": "sha256:..."
}
```

创建 payload index：`manual_id`、`snapshot_id`、`vehicle_model`、`topic_id`。问题涉及具体车型或手册版本时先做 payload 过滤，再进行向量召回，避免不同版本内容混答。

## 5. 问答链路

```text
问题
  -> 可选车型/版本过滤
  -> BGE-M3 Query Embedding
  -> Qdrant Cosine Top-10
  -> 相似度阈值过滤
  -> BGE Reranker Top-3
  -> 读取完整 chunk 与来源元数据
  -> OpenRouter LLM 生成答案和引用
```

LLM 约束：

- 只能依据传入的 Top-3 正文回答。
- 每个关键结论使用 `[1]`、`[2]` 引用 chunk。
- 证据不足或冲突时明确拒答或说明冲突。
- 返回 `answer`、`citations`、`retrieved_chunks` 和分阶段 `timings_ms`。

目录命中、标题相似和向量分数只表示候选相关，不能直接作为最终事实证据；最终回答必须绑定实际传入 LLM 的正文 chunk。

最小 API：

| 接口 | 用途 |
|---|---|
| `POST /v1/manuals/import` | 创建手册采集与索引任务 |
| `POST /v1/retrieve` | 仅执行召回和重排，供调试与压测 |
| `POST /v1/chat` | 完整 RAG 问答 |
| `GET /health` | API、模型和 Qdrant 健康检查 |
| `GET /metrics` | Prometheus 指标 |

## 6. 离线质量评测

只维护一份50题数据集，覆盖单chunk、同topic多chunk、跨topic、跨车型/年款和不可回答问题，并用标签覆盖事实、操作、安全、条件、参数、对比和总结场景。每条记录：

```json
{
  "id": "q001",
  "user_input": "驾驶前需要检查什么？",
  "reference": "驾驶前应检查……",
  "reference_contexts": ["手册原文……"],
  "gold_chunk_ids": ["..."],
  "answerable": true,
  "retrieval_filters": {"manual_keys": ["W022025ULTRA"]},
  "tags": ["operation", "safety"]
}
```

一次运行同时评估两层：

1. 检索层：基于gold chunk ID计算Precision@5、Recall@5、F1@5和MRR@10。
2. 生成层：Ragas `Faithfulness`评估忠实度，`AnswerRelevancy`评估答案相关性，`FactualCorrectness(mode="recall")`评估完整性；不可回答问题单独计算拒答正确率。

Ragas Judge和Generator必须固定模型，报告保存数据集版本、组件ID、模型、逐题结果和错误。自动生成的42条参考答案经人工复核后才能设正式验收阈值；任何指标都必须附数据集规模、版本和判分规则。

对比实验只保留三组：

- 向量 Top-5 基线。
- 向量 Top-10 + Reranker Top-3。
- 不同 chunk 大小的消融实验。

## 7. 性能评测

单份手册当前只有约 196 个唯一 topic，通常不足以暴露向量数据库在生产规模下的瓶颈。因此性能评测分为两类：

### 7.1 真实语料端到端测试

- 使用真实手册验证采集、Embedding、检索、重排和回答的实际延迟。
- `/v1/retrieve`：1、10、50、100 并发。
- `/v1/chat`：1、5、10 并发，并固定 LLM 模型。
- 每轮预热 2 分钟、正式运行 10 分钟，记录 P50/P95/P99、QPS、错误率。

### 7.2 扩容压力测试

- 使用相同向量维度生成明确标记为 synthetic 的 1 万、10 万、100 万 point 数据集，只用于容量和 ANN 性能测试。
- 不把复制文本或合成向量的结果解释为真实业务检索质量。
- 分别记录 Qdrant 查询延迟、API 排队时间、CPU、内存、磁盘、索引构建时间和召回率/近似误差。

必须分别计时：

```text
embedding_ms
qdrant_search_ms
rerank_ms
llm_first_token_ms
llm_total_ms
total_ms
```

`openrouter/free` 会随机选择免费模型且存在共享限流和可用性波动，只适合联调，不能用于可复现的端到端性能结论。正式压测采用以下两种模式：

- `mock LLM`：评估 API、检索和序列化开销。
- 固定的付费模型或本地模型：评估真实端到端延迟，报告中单列外部 LLM 耗时。

## 8. 项目结构

```text
lixiang-manual-rag/
  app/
    domain/          # Pydantic 领域对象
    ports/           # 可替换模块的 Protocol
    pipelines/       # ingestion、retrieval、chat 编排
    bootstrap/       # 配置、Registry、Factory
    adapters/        # 数据源、解析、分块、模型和存储实现
    api/             # FastAPI 路由，只调用 pipeline
  configs/
    baseline.yaml
    experiments/     # 对比实验配置
  data/
    raw/             # 原始快照，不提交 Git
    normalized/      # topic/chunk JSONL，不提交 Git
    eval/            # eval.jsonl
  scripts/evaluate.py # 单次检索 + 生成 + Ragas统一评测
  loadtest/          # locustfile.py、synthetic_loader.py
  infra/             # compose.yaml、prometheus.yml
  docs/
    TECHNICAL_DESIGN.md
    MODULAR_ARCHITECTURE.md
  .gitignore
  pyproject.toml
  README.md
```

## 9. 实施顺序与完成标准

### 阶段一：采集与解析

- 完成目录递归、URL 去重、断点续抓和 HTML 解析。
- 生成 raw snapshot、manifest 和 normalized topic JSONL。
- 能报告目录数、唯一 topic 数、成功数、失败数和空正文数。

### 阶段二：向量检索

- 实现结构化分块、BGE-M3 Embedding、Qdrant 幂等写入和 Top-K 查询。
- `/v1/retrieve` 返回分数、正文和可打开的原始来源。

### 阶段三：证据问答

- 加入 reranker 和 OpenRouter。
- `/v1/chat` 输出带引用答案；无正文证据时拒答。

### 阶段四：质量评测

- 建立唯一的50题宽类型冻结数据集。
- 输出检索F1@5、MRR@10及Ragas忠实度、答案相关性、完整性报告。
- 快速测试使用同一数据集的`--limit`，不维护第二份smoke数据。

### 阶段五：性能评测

- 完成真实语料与 synthetic 扩容两套压测。
- 报告各阶段 P95/P99、吞吐、错误率和资源占用，并明确测试环境与边界。

项目完成的核心标准不是“模型能回答”，而是能够从一个回答反查到 Qdrant chunk、topic 页面、手册版本和原始 URL，同时能够用固定评测集和可复现压测解释质量与性能变化。
