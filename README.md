# 理想汽车手册 RAG

从零实现的中文 RAG 学习项目：抓取理想汽车公开用户手册，用本地 BGE-M3、可替换的检索与向量存储组件完成召回和精排，通过 OpenRouter 输出严格引用答案，并用分层离线集、Locust 和 Prometheus 分析质量与性能。当前默认向量存储实现为 Qdrant。

当前实现是单体、模块化 RAG，不包含 Agent、多租户权限和消息队列。自然语言入口为 FastAPI `/v1/chat`；另提供单次请求 RAG Trace 调试台，可实时查看 Embedding 摘要、召回、精排、证据筛选、LLM 输入输出和引用校验。Swagger 位于 <http://127.0.0.1:8000/docs>。

## 快速开始

要求：Python 3.11、[uv](https://docs.astral.sh/uv/)、Docker Desktop。

```bash
# 在本地创建 .env，并按需配置 OPENROUTER_API_KEY 等环境变量
uv sync --group dev
docker compose up -d
uv run python -m app.ingestion.service --config configs/mvp.yaml
APP_CONFIG=configs/mvp.yaml uv run uvicorn app.api.main:app --reload
```

服务启动后打开 <http://127.0.0.1:8000/trace-console> 使用 Trace 调试台。调试台先通过 `POST /v1/traces` 创建请求，再从 `GET /v1/traces/{trace_id}/events` 接收 SSE 事件；`GET /v1/traces/{trace_id}` 用于刷新和断线后的快照恢复。

`OPENROUTER_API_KEY` 未配置时，按配置可回退到 Mock Generator；真实评测应在 `.env` 固定 `OPENROUTER_MODEL`，不要长期使用会动态路由的 `openrouter/free`。BGE 模型采用懒加载，第一次检索会明显慢于热请求。

## API

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/v1/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"question":"驾驶前需要检查什么？","top_k":5,"filters":{"snapshot_id":"20250916141802"}}'

curl -X POST http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"驾驶前需要检查什么？"}'
```

接口还包括 Trace 创建/事件/快照、异步入库 `POST /v1/manuals/import`、任务查询 `GET /v1/manuals/import/{job_id}` 与 Prometheus `/metrics`。过滤字段支持 `manual_id`、`snapshot_id`、`vehicle_model`、`topic_ids`。

## 离线质量评测

```bash
uv run python -m scripts.evaluate_smoke \
  --config configs/experiments/dense.yaml \
  --dataset data/eval/full_v1.jsonl \
  --top-k 10 --also-k 5 \
  --output reports/evaluation/dense-full-v1.json

uv run python -m scripts.evaluate_answers --limit 10
```

`full_v1.jsonl` 共 100 条：55 个单 chunk、20 个同 topic 多 chunk、10 个跨 topic、15 个不可回答问题。85 个可回答样本的证据结构已校验，但答案内容仍标记为 `content_review_required`；人工复核前，报告只用于调试和横向实验，不能作为正式准确率。

已提供 Dense、BM25、Hybrid、Dense + Reranker、Hybrid + Reranker 配置。更换 Chunker 或 Embedding 必须使用新 collection；只切 Retriever/Reranker/Generator 可以复用现有向量。

## 全车型手册采集

车型下拉从官网目录 `https://manuals.lixiang.com/carlmodels_zh-cn_officialwebsite.json` 动态加载，代码不写死车型。当前目录有39份手册版本、合计7,645个唯一 topic。每个版本使用 `manual_key/snapshot_id` 隔离文件，统一写入独立 collection `lixiang_all_manuals_bge_m3_v1`：

不同年款存在大量相同正文。入库时用“去掉车型/版本前缀后的正文 + Embedding 组件 ID”作为持久化缓存键，跨手册复用向量；Qdrant payload 和引用仍保留完整车型信息。首轮实测中，ONE 2021 的 558 个 chunk 有 421 个命中 ONE 2020 的向量缓存（75.4%），只需新计算 137 个。

```bash
# 全部目录条目；支持断点续跑
uv run python -m app.ingestion.batch_service --config configs/all-models.yaml

# 开发时只处理匹配车型或前 N 份
uv run python -m app.ingestion.batch_service \
  --config configs/all-models.yaml --include 'i8|i6' --limit 2

# 使用全车型索引启动 API，并查看目录/处理状态
APP_CONFIG=configs/all-models.yaml uv run uvicorn app.api.main:app --reload
curl http://127.0.0.1:8000/v1/manuals
```

检索时建议传入 `manual_keys` 或 `manual_names`，防止不同年款和配置款串用：

```json
{
  "question": "如何进行冬季行驶前检查？",
  "filters": {"manual_keys": ["W022025ULTRA"]}
}
```

## 并发与监控

```bash
# 先启动 API；0 表示只压检索，设置 0~1 可混入 /v1/chat
RAG_LOADTEST_CHAT_RATIO=0 uv run locust -f loadtest/locustfile.py \
  --headless -u 5 -r 1 -t 20s --host http://127.0.0.1:8000

# 启动 Prometheus 与 Grafana；API 仍在宿主机 8000 端口运行
docker compose --profile observability up -d
```

Prometheus: <http://127.0.0.1:9090>；Grafana: <http://127.0.0.1:3000>，本地默认账号 `admin/admin`。合成数据脚本只允许写入 `synthetic_` 前缀 collection：

```bash
uv run python -m loadtest.synthetic_loader \
  --collection synthetic_bge_10k --points 10000 --dimension 1024
```

## 验证

```bash
uv run ruff check app tests scripts loadtest
uv run ruff format --check app tests scripts loadtest
uv run pytest -q
RUN_QDRANT_INTEGRATION=1 uv run pytest tests/integration/test_qdrant.py -q
docker compose --profile observability config --quiet
```

## 文档

- [当前实现架构](docs/CURRENT_IMPLEMENTATION_ARCHITECTURE.md)
- [完整技术方案与实现状态](docs/TECHNICAL_DESIGN.md)
- [模块替换与实验隔离规则](docs/MODULAR_ARCHITECTURE.md)
- [MVP 历史基线](docs/MVP_TECHNICAL_DESIGN.md)
- [本机短压测记录](reports/performance/README.md)
