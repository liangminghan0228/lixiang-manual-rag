from __future__ import annotations

from app.settings import RuntimeEnvironment, load_settings


def test_mvp_settings_load() -> None:
    settings, _ = load_settings("configs/mvp.yaml")
    assert settings.embedding.model == "BAAI/bge-m3"
    assert settings.vector_store.collection == "lixiang_mvp_bge_m3_v1"
    assert settings.data.raw_dir.is_absolute()


def test_runtime_environment_reads_ragas_settings(monkeypatch) -> None:
    monkeypatch.setenv("RAGAS_JUDGE_API_KEY", "judge-key")
    monkeypatch.setenv("RAGAS_JUDGE_MODEL", "provider/fixed-judge")
    monkeypatch.setenv("RAGAS_JUDGE_BASE_URL", "https://judge.example/v1")

    runtime = RuntimeEnvironment(_env_file=None)

    assert runtime.ragas_judge_api_key == "judge-key"
    assert runtime.ragas_judge_model == "provider/fixed-judge"
    assert runtime.ragas_judge_base_url == "https://judge.example/v1"
