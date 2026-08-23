# 最终实验设计

## 1. 实验范围

本阶段只比较两个维度，其他变量固定：

| 维度 | 实验组 | 控制变量 |
|---|---|---|
| 查询优化 | Identity、Normalize、Rewrite、Expansion、Multi-Query、HyDE、Decomposition | 同一数据集、Chunker、Embedding、Retriever、Reranker、EvidenceSelector、Generator 和评测指标 |
| RAG Strategy | Self-RAG、Agentic RAG、GraphRAG | Vanilla 作为控制组；索引、基础配置、EvidenceSelector、Generator 和评测指标固定 |

固定项：当前 `HeadingChunker`、`DiversifiedEvidenceSelector`、当前 Generator/Prompt，以及现有指标。快速实验使用从 `rag_eval_v2.jsonl` 选出的30题分层子集；暂不比较分块、EvidenceSelector、LLM 或 Prompt。

## 2. 查询优化定义

- **Identity**：原问题直接检索。
- **Normalize**：大小写、Unicode、标点和配置别名归一化。
- **Rewrite**：调用固定的查询规划 LLM，将问题改写成一个检索查询。
- **Expansion**：按配置词典补充领域同义词/提示词，仍产生一个检索查询。
- **Multi-Query**：由固定 LLM 生成多个查询，分别检索后用 RRF 融合。
- **HyDE**：由固定 LLM 生成假设性答案文本，以该文本检索。
- **Decomposition**：将复杂问题拆成多个子查询，分别检索后用 RRF 融合。

这些方式是互相独立的实验配置，不在同一个配置中叠加。当前查询规划模型固定为 `nvidia/nemotron-3.5-content-safety:free`。

## 3. RAG Strategy 定义

- **Vanilla**：一次检索、一次证据选择、一次生成，作为控制组。
- **Self-RAG**：控制器判断证据是否足够，允许一次查询重写/补检索，再生成并做支持性检查。
- **Agentic RAG**：控制器在最多 `max_steps` 步内决定继续检索或生成。
- **GraphRAG**：当前实现是基于文档章节/主题邻接关系的 document-structure graph baseline；不是实体抽取、社区发现或全局摘要型 Entity/Community GraphRAG。

Self-RAG、Agentic RAG 的 Controller、最终答案 Generator 和 Ragas Judge 同样固定为 `nvidia/nemotron-3.5-content-safety:free`。该模型定位为内容安全分类模型，因此正式运行前仍需通过结构化 JSON、问答生成与 Ragas Judge 兼容性冒烟测试。

## 4. 运行方式

先验证配置，不加载模型：

```bash
cd lixiang-manual-rag
uv run python -m scripts.build_eval_subset \
  --spec configs/eval-subsets/rag_eval_v2_30.yaml
uv run python -m scripts.run_experiments \
  --suite configs/suites/query-optimization.yaml --dry-run
uv run python -m scripts.run_experiments \
  --suite configs/suites/rag-strategies.yaml --dry-run
```

正式运行：

```bash
uv run python -m scripts.run_experiments \
  --suite configs/suites/query-optimization.yaml
uv run python -m scripts.run_experiments \
  --suite configs/suites/rag-strategies.yaml
```

查询优化套件默认 `retrieval_only: true`，只比较召回指标；Strategy 套件执行完整问答评测。两个套件默认使用30题分层子集：10个单Chunk、5个同主题多Chunk、6个跨主题、4个跨手册和5个不可回答问题。结果位于 `reports/experiments/{suite_id}/{run_id}/`，每个配置单独保存报告和 manifest。

## 5. 公平性与边界

两套套件均继承 `configs/experiments/hybrid_reranker.yaml`，因此默认使用相同索引、数据和指标。查询优化套件固定 Hybrid 与 Reranker，只改变查询计划及其必要的融合；Strategy 套件允许策略改变检索编排，其中 document-structure GraphRAG 会使用图检索代替基础 Hybrid。额外控制步骤、停止原因和组件 ID 会写入 Trace 或评测报告。

本阶段不改变数据集、分块和指标实现，也不声称 GraphRAG 已实现实体图谱或社区摘要能力。
