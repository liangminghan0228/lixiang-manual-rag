from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.evaluation import (
    EvaluationCase,
    aggregate_retrieval_metrics,
    evaluate_retrieval_case,
    suggest_min_score,
)
from app.settings import PROJECT_ROOT
from app.wiring import build_container


def load_cases(path: Path) -> list[EvaluationCase]:
    return [
        EvaluationCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--config", default="configs/experiments/dense.yaml")
    parser.add_argument("--dataset", default="data/eval/smoke.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--also-k", type=int, nargs="*", default=[5])
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    dataset_path = resolve_path(args.dataset)
    cases = load_cases(dataset_path)
    container = build_container(args.config)
    ks = sorted({args.top_k, *args.also_k})
    max_k = max(ks)
    metrics_by_k: dict[int, list[object]] = {k: [] for k in ks}
    labeled_scores: list[tuple[bool, float | None]] = []
    details: list[dict[str, object]] = []

    for case in cases:
        outcome = container.retriever.retrieve(case.question, max_k)
        top_score = outcome.results[0].score if outcome.results else None
        labeled_scores.append((case.answerable, top_score))
        case_metrics: dict[str, object] = {}
        if case.answerable:
            for k in ks:
                metric = evaluate_retrieval_case(case, outcome.results, k)
                metrics_by_k[k].append(metric)
                case_metrics[str(k)] = metric.model_dump(mode="json")
        details.append(
            {
                "id": case.id,
                "question_type": case.question_type,
                "answerable": case.answerable,
                "label_status": case.label_status,
                "top_score": top_score,
                "timings_ms": outcome.timings_ms,
                "metrics": case_metrics,
            }
        )

    threshold = suggest_min_score(labeled_scores)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment_id": container.settings.experiment.id,
        "dataset": str(dataset_path),
        "cases": len(cases),
        "label_status_counts": {
            status: sum(case.label_status == status for case in cases)
            for status in sorted({case.label_status for case in cases})
        },
        "metrics_by_k": {
            str(k): aggregate_retrieval_metrics(items) for k, items in metrics_by_k.items()
        },
        "threshold_calibration": (
            {
                "suggested_min_score": threshold.threshold,
                "answerable_accept_rate": threshold.true_positive_rate,
                "unanswerable_reject_rate": threshold.true_negative_rate,
                "balanced_accuracy": threshold.balanced_accuracy,
                "note": "Only calibrates the evidence gate; it is not answer accuracy.",
            }
            if threshold
            else None
        ),
        "details": details,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
