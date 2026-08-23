from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from app.settings import PROJECT_ROOT


class EvalSubsetSpec(BaseModel):
    id: str
    source: str
    output: str
    selected_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_selected_ids(self) -> EvalSubsetSpec:
        if len(set(self.selected_ids)) != len(self.selected_ids):
            raise ValueError("selected_ids must be unique")
        return self


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def build_subset(spec_path: str | Path) -> dict[str, Any]:
    selected_spec = _project_path(spec_path)
    spec = EvalSubsetSpec.model_validate(
        yaml.safe_load(selected_spec.read_text(encoding="utf-8")) or {}
    )
    source = _project_path(spec.source)
    output = _project_path(spec.output)
    rows = _read_jsonl(source)
    by_id = {str(row.get("id")): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"source dataset contains duplicate ids: {source}")
    missing = [case_id for case_id in spec.selected_ids if case_id not in by_id]
    if missing:
        raise ValueError(f"selected ids are missing from source: {', '.join(missing)}")
    selected = [by_id[case_id] for case_id in spec.selected_ids]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    summary = {
        "subset_id": spec.id,
        "source": str(source),
        "output": str(output),
        "count": len(selected),
        "question_types": dict(Counter(row.get("question_type") for row in selected)),
        "answerable": dict(Counter(str(bool(row.get("answerable"))).lower() for row in selected)),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reproducible evaluation subset")
    parser.add_argument(
        "--spec",
        default="configs/eval-subsets/rag_eval_v2_30.yaml",
    )
    args = parser.parse_args()
    print(json.dumps(build_subset(args.spec), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
