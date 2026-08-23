from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.build_eval_subset import EvalSubsetSpec, build_subset


def test_subset_rejects_duplicate_ids() -> None:
    try:
        EvalSubsetSpec(id="subset", source="source", output="output", selected_ids=["a", "a"])
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate ids should be rejected")


def test_build_subset_uses_explicit_id_order(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {"id": "a", "question_type": "single_chunk", "answerable": True},
                {"id": "b", "question_type": "unanswerable", "answerable": False},
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output.jsonl"
    spec = tmp_path / "subset.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "id": "subset",
                "source": str(source),
                "output": str(output),
                "selected_ids": ["b", "a"],
            }
        ),
        encoding="utf-8",
    )

    summary = build_subset(spec)
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert [row["id"] for row in rows] == ["b", "a"]
    assert summary["count"] == 2
    assert summary["question_types"] == {"unanswerable": 1, "single_chunk": 1}
