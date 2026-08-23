from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ragas
from ragas import EvaluationDataset

from app.evaluation import (
    RETRIEVAL_F1_K,
    RETRIEVAL_MRR_K,
    EvaluationCase,
    GenerationCaseMetrics,
    RetrievalCaseMetrics,
    aggregate_generation_metrics,
    aggregate_retrieval_metrics,
    build_ragas_judge,
    evaluate_retrieval_case,
)
from app.generation.mock import MockGenerator
from app.settings import PROJECT_ROOT
from app.wiring import build_container


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_cases(path: Path, limit: int) -> list[EvaluationCase]:
    cases = [
        EvaluationCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("evaluation dataset contains duplicate case ids")
    return cases[:limit] if limit else cases


def _score(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def render_markdown(report: dict[str, Any]) -> str:
    retrieval = report["retrieval"]["overall"]
    generation = report["generation"]
    lines = [
        "# RAG Evaluation Report",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Cases: {report['evaluated_cases']}/{report['dataset_cases']}",
        f"- Experiment: `{report['experiment_id']}`",
        f"- Ragas: `{report['ragas']['status']}` (`{report['ragas']['version']}`)",
        "",
        "## Overall",
        "",
        (
            "| Retrieval F1@5 | MRR@10 | Faithfulness | Answer relevancy "
            "| Completeness | Refusal accuracy |"
        ),
        "|---:|---:|---:|---:|---:|---:|",
        (
            f"| {_score(retrieval.get('f1_at_5'))} "
            f"| {_score(retrieval.get('mrr_at_10'))} "
            f"| {_score(generation.get('faithfulness'))} "
            f"| {_score(generation.get('answer_relevancy'))} "
            f"| {_score(generation.get('completeness'))} "
            f"| {_score(generation.get('refusal_accuracy'))} |"
        ),
        "",
        "## Retrieval by question type",
        "",
        "| Type | Count | Precision@5 | Recall@5 | F1@5 | MRR@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for question_type, values in report["retrieval"]["by_question_type"].items():
        lines.append(
            f"| {question_type} | {values['count']} | {_score(values['precision_at_5'])} "
            f"| {_score(values['recall_at_5'])} | {_score(values['f1_at_5'])} "
            f"| {_score(values['mrr_at_10'])} |"
        )
    if report["errors"]:
        lines.extend(["", "## Errors", "", f"- Failed cases: {report['errors']}"])
    lines.append("")
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = resolve_path(args.dataset)
    cases = load_cases(dataset_path, args.limit)
    container = build_container(args.config)
    max_k = RETRIEVAL_MRR_K
    judge = None
    judge_model = args.judge_model or container.runtime.ragas_judge_model
    judge_base_url = args.judge_base_url or container.runtime.ragas_judge_base_url
    if not args.retrieval_only:
        api_key = container.runtime.ragas_judge_api_key or container.runtime.openrouter_api_key
        if not judge_model:
            raise ValueError("RAGAS_JUDGE_MODEL or --judge-model is required")
        if judge_model == "openrouter/free":
            raise ValueError("Ragas requires a fixed judge model, not openrouter/free")
        if not api_key:
            raise ValueError("RAGAS_JUDGE_API_KEY or OPENROUTER_API_KEY is required")
        if isinstance(container.generator, MockGenerator):
            raise ValueError("generation evaluation requires a configured non-mock generator")
        if container.settings.generation.model == "openrouter/free":
            raise ValueError("generation evaluation requires a fixed OPENROUTER_MODEL")
        judge = build_ragas_judge(
            model=judge_model,
            api_key=api_key,
            base_url=judge_base_url,
            embedder=container.embedder,
            cache_dir=resolve_path(args.ragas_cache_dir),
            relevancy_strictness=args.relevancy_strictness,
        )

    retrieval_metrics: list[RetrievalCaseMetrics] = []
    generation_metrics: list[GenerationCaseMetrics] = []
    ragas_samples = []
    details: list[dict[str, Any]] = []
    error_count = 0
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.id}: {case.user_input}")
        detail: dict[str, Any] = {
            "id": case.id,
            "question_type": case.question_type,
            "answerable": case.answerable,
            "tags": case.tags,
        }
        try:
            outcome = await container.chat_service.retrieve(
                case.user_input,
                max_k,
                case.retrieval_filters,
            )
            detail["retrieval_timings_ms"] = dict(outcome.timings_ms)
            detail["retrieved"] = [
                {
                    "chunk_id": result.chunk.chunk_id,
                    "rank": result.rank,
                    "score": result.score,
                }
                for result in outcome.results[:max_k]
            ]
            if case.answerable:
                metric = evaluate_retrieval_case(
                    case,
                    outcome.results,
                )
                retrieval_metrics.append(metric)
                detail["retrieval_metrics"] = metric.model_dump(mode="json")
            if judge is not None:
                answer = await container.chat_service.answer_from_outcome(
                    case.user_input,
                    outcome,
                    case.retrieval_filters,
                )
                sample, metric = await judge.score(case, answer)
                ragas_samples.append(sample)
                generation_metrics.append(metric)
                detail["generation_metrics"] = metric.model_dump(mode="json")
                detail["answer"] = {
                    "text": answer.text,
                    "refused": answer.refused,
                    "citation_validated": answer.citation_validated,
                    "citation_chunk_ids": [item.chunk_id for item in answer.citations],
                    "evidence_chunk_ids": [item.chunk.chunk_id for item in answer.evidence],
                    "timings_ms": answer.timings_ms,
                }
        except Exception as exc:  # noqa: BLE001 - preserve per-case evaluation failures
            error_count += 1
            detail["error"] = f"{type(exc).__name__}: {exc}"
            if args.fail_fast:
                raise
        details.append(detail)

    if ragas_samples:
        evaluation_dataset = EvaluationDataset(samples=ragas_samples)
        evaluation_dataset.validate_samples(evaluation_dataset.samples)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment_id": container.settings.experiment.id,
        "dataset": str(dataset_path),
        "dataset_cases": len(load_cases(dataset_path, 0)),
        "evaluated_cases": len(cases),
        "question_type_counts": dict(Counter(case.question_type for case in cases)),
        "label_status_counts": dict(Counter(case.label_status for case in cases)),
        "components": {
            "embedder": container.embedder.component_id,
            "query_processor": container.query_processor.component_id,
            "retriever": container.retriever.component_id,
            "reranker": container.reranker.component_id,
            "generator": container.generator.component_id,
            "rag_strategy": container.rag_strategy.component_id,
        },
        "metric_contract": {
            "retrieval": [f"f1@{RETRIEVAL_F1_K}", f"mrr@{RETRIEVAL_MRR_K}"],
            "generation": [
                "ragas_faithfulness",
                "ragas_answer_relevancy",
                "ragas_factual_correctness_recall_as_completeness",
                "refusal_accuracy",
            ],
        },
        "retrieval": aggregate_retrieval_metrics(retrieval_metrics),
        "generation": aggregate_generation_metrics(generation_metrics),
        "ragas": {
            "version": ragas.__version__,
            "status": (
                "skipped_retrieval_only"
                if judge is None
                else ("completed" if error_count == 0 else "completed_with_errors")
            ),
            "judge_model": judge_model if judge is not None else None,
            "judge_base_url": judge_base_url if judge is not None else None,
            "relevancy_strictness": args.relevancy_strictness if judge is not None else None,
        },
        "errors": error_count,
        "details": details,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and generation with Ragas")
    parser.add_argument("--config", default="configs/all-models.yaml")
    parser.add_argument("--dataset", default="data/eval/rag_eval_v2.jsonl")
    parser.add_argument("--output", default="reports/evaluation/rag-eval-v2.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--judge-model", default="")
    parser.add_argument(
        "--judge-base-url",
        default="",
    )
    parser.add_argument("--ragas-cache-dir", default="artifacts/ragas-cache")
    parser.add_argument("--relevancy-strictness", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be non-negative")
    if args.relevancy_strictness <= 0:
        parser.error("relevancy-strictness must be positive")
    report = asyncio.run(run(args))
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    summary = {key: report[key] for key in ("retrieval", "generation", "ragas", "errors")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
