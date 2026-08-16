from __future__ import annotations

import json

from app.evaluation import suggest_min_score
from app.settings import PROJECT_ROOT, load_settings


def test_mvp_settings_load() -> None:
    settings, _ = load_settings("configs/mvp.yaml")
    assert settings.embedding.model == "BAAI/bge-m3"
    assert settings.vector_store.collection == "lixiang_mvp_bge_m3_v1"
    assert settings.data.raw_dir.is_absolute()


def test_smoke_dataset_has_twenty_cases_and_unanswerables() -> None:
    path = PROJECT_ROOT / "data" / "eval" / "smoke.jsonl"
    cases = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(cases) == 20
    assert sum(not case["answerable"] for case in cases) >= 5


def test_threshold_calibration_separates_answerable_cases() -> None:
    metrics = suggest_min_score([(True, 0.82), (True, 0.73), (False, 0.22), (False, 0.31)])

    assert metrics is not None
    assert 0.31 < metrics.threshold < 0.73
    assert metrics.balanced_accuracy == 1.0
