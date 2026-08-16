from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.evaluation import EvaluationCase, evaluate_answer_case
from app.settings import PROJECT_ROOT
from app.wiring import build_container


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate answer points and citations")
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--dataset", default="data/eval/full_v1.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="reports/evaluation/answers.json")
    args = parser.parse_args()

    cases = [
        EvaluationCase.model_validate_json(line)
        for line in resolve_path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        cases = cases[: args.limit]
    container = build_container(args.config)
    details: list[dict[str, object]] = []
    for case in cases:
        answer = container.chat_service.chat(case.question)
        details.append(
            {
                "id": case.id,
                "question_type": case.question_type,
                "label_status": case.label_status,
                "metrics": evaluate_answer_case(case, answer),
                "answer": answer.model_dump(mode="json"),
            }
        )

    metric_names = {
        key
        for item in details
        for key in item["metrics"]  # type: ignore[union-attr]
    }
    aggregate = (
        {
            name: sum(float(item["metrics"].get(name, 0)) for item in details) / len(details)  # type: ignore[union-attr]
            for name in sorted(metric_names)
        }
        if details
        else {}
    )
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment_id": container.settings.experiment.id,
        "cases": len(cases),
        "aggregate": aggregate,
        "unsupported_claim_rate": None,
        "unsupported_claim_note": (
            "Requires a reviewed claim-level judge set; not inferred heuristically."
        ),
        "details": details,
    }
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
