# RAG 对比平台扩展性改造实施方案

> 状态：最终实验范围已实现。分块方式和现有评测指标保持不变。

## 1. 范围与控制变量

平台当前比较：

- 查询优化：Identity、Normalize、Rewrite、Expansion、Multi-Query、HyDE、Decomposition；
- RAG Strategy：Self-RAG、Agentic RAG、GraphRAG，Vanilla 仅作为控制组。

EvidenceSelector 固定现有 `DiversifiedEvidenceSelector`，最终答案的 LLM/Prompt 固定当前实现，暂不做这两项对比。Chunker、索引和指标计算均保持现状；快速实验统一使用同一30题分层子集，查询优化套件固定 Retriever/Reranker，Strategy 套件允许策略改变检索编排。

## 2. 已实施改造

| 改造项 | 当前实现 |
|---|---|
| 查询计划 | `QueryProcessor -> QueryPlan`，支持单查询、多查询及 RRF 融合 |
| 查询组件 | `app/retrieval/query.py` 中注册七种 QueryProcessor；LLM 方法通过统一 planner 注入 |
| RAG 编排 | `RagStrategy` 契约；Vanilla、Self-RAG、Agentic RAG、document-structure GraphRAG 已装配 |
| 配置继承 | YAML `extends` 支持从共同基线派生实验，避免复制检索配置 |
| 批量评测 | `scripts.run_experiments` 按 suite 串行执行统一 `scripts.evaluate`，报告隔离 |
| 可观测性 | 报告记录 QueryProcessor、RAG Strategy 与答案耗时；Trace 记录策略步骤和停止原因 |

## 3. 扩展规则

### QueryProcessor

实现 `process() -> QueryPlan`，在 `QUERY_PROCESSORS` 注册，并为新组件添加独立 YAML。多查询必须声明真实的融合策略（当前为 RRF），不能把多查询伪装成 `single_query`。

### RAG Strategy

实现 `RagStrategy` 的 `retrieve`、`answer` 和 `answer_from_outcome` 契约；复用 `RetrievalOutcome`、`EvidenceBundle` 和引用校验语义。额外检索/LLM 轮次必须记录步骤、耗时和错误，不得绕过统一评测入口。

### 新 Retriever/Reranker/Generator

实现既有 Protocol，在 Registry 注册 Factory，Factory 显式接收 settings 和依赖，并补充装配测试与实验 YAML。主流程不直接创建 Qdrant、Embedding 或 LLM 客户端。

## 4. 实验配置与执行

配置目录：

```text
configs/suites/query-optimization.yaml  # 7 个查询优化实验，retrieval_only
configs/suites/rag-strategies.yaml      # Vanilla + 3 个 Strategy，完整评测
configs/experiments/query/*.yaml
configs/experiments/rag/*.yaml
configs/eval-subsets/rag_eval_v2_30.yaml
```

```bash
uv run python -m scripts.build_eval_subset \
  --spec configs/eval-subsets/rag_eval_v2_30.yaml

uv run python -m scripts.run_experiments \
  --suite configs/suites/query-optimization.yaml --dry-run
uv run python -m scripts.run_experiments \
  --suite configs/suites/rag-strategies.yaml --dry-run

uv run python -m scripts.run_experiments \
  --suite configs/suites/query-optimization.yaml
uv run python -m scripts.run_experiments \
  --suite configs/suites/rag-strategies.yaml
```

Identity、Normalize、Expansion 不依赖查询 LLM；其余实验需要 OpenRouter key。当前 `QUERY_OPTIMIZER_MODEL`、`OPENROUTER_MODEL`、`RAG_CONTROLLER_MODEL` 和 `RAGAS_JUDGE_MODEL` 统一固定为 `nvidia/nemotron-3.5-content-safety:free`，不使用动态 `openrouter/free`。由于它是内容安全分类模型，正式评测前必须先验证查询 JSON、回答生成和 Judge 兼容性。

## 5. 验收标准

- 七个查询优化配置和四个 Strategy 配置均可 dry-run；
- 实验只改变声明的组件，其余配置从 Hybrid Reranker 基线继承；
- 快速实验统一使用30题分层子集，最终确认可切回50题全集；指标实现保持不变；
- Vanilla 保持历史检索→证据→生成→引用校验行为；
- GraphRAG 报告明确标注为 document-structure graph baseline，不宣称 Entity/Community GraphRAG；
- 单元测试、Ruff、格式检查通过。
