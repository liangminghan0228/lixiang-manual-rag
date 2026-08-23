from __future__ import annotations

from pathlib import Path

import yaml

from app.settings import RuntimeEnvironment, load_settings
from scripts.run_experiments import load_suite


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


def test_experiment_config_can_extend_a_base_config(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "data": {"source_url": "https://example.com", "snapshot_id": "snapshot"},
                "embedding": {"provider": "hash_mock"},
                "vector_store": {"provider": "in_memory"},
                "retrieval": {"provider": "dense", "top_k": 5, "candidate_top_k": 10},
                "generation": {"provider": "mock"},
            }
        ),
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        yaml.safe_dump(
            {
                "extends": str(base),
                "experiment": {"id": "child-v1"},
                "retrieval": {"query_processor": {"provider": "multi_query", "max_queries": 3}},
            }
        ),
        encoding="utf-8",
    )

    settings, _ = load_settings(child, apply_runtime_overrides=False)

    assert settings.retrieval.provider == "dense"
    assert settings.retrieval.query_processor.provider == "multi_query"
    assert settings.retrieval.query_processor.max_queries == 3
    assert settings.experiment.id == "child-v1"


def test_config_extends_is_relative_to_the_child_file(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "data": {"source_url": "https://example.com", "snapshot_id": "snapshot"},
                "embedding": {"provider": "hash_mock"},
                "vector_store": {"provider": "in_memory"},
                "retrieval": {"provider": "dense"},
                "generation": {"provider": "mock"},
                "experiment": {"id": "inherited-v1"},
            }
        ),
        encoding="utf-8",
    )
    child = tmp_path / "nested" / "child.yaml"
    child.parent.mkdir()
    child.write_text("extends: ../base.yaml\n", encoding="utf-8")

    settings, _ = load_settings(child, apply_runtime_overrides=False)

    assert settings.experiment.id == "inherited-v1"


def test_final_query_suite_changes_only_the_query_processor() -> None:
    suite = load_suite("configs/suites/query-optimization.yaml")
    loaded = [load_settings(path, apply_runtime_overrides=False)[0] for path in suite.configs]

    assert [item.retrieval.query_processor.provider for item in loaded] == [
        "identity",
        "normalize",
        "expansion",
        "rewrite",
        "multi_query",
        "hyde",
        "decomposition",
    ]
    assert suite.dataset == "data/eval/rag_eval_v2_30.jsonl"
    fixed = {
        (
            item.data.chunker,
            item.data.target_chars,
            item.data.overlap_chars,
            item.embedding.provider,
            item.vector_store.collection,
            item.retrieval.provider,
            item.retrieval.reranker.provider,
            item.retrieval.evidence_selector,
            item.generation.provider,
            item.rag.strategy,
        )
        for item in loaded
    }
    assert fixed == {
        (
            "heading",
            500,
            80,
            "bge_m3_local",
            "lixiang_all_manuals_bge_m3_v1",
            "hybrid",
            "bge_local",
            "diversified",
            "mock",
            "vanilla",
        )
    }


def test_final_rag_suite_has_one_control_and_three_target_strategies() -> None:
    suite = load_suite("configs/suites/rag-strategies.yaml")
    loaded = [load_settings(path, apply_runtime_overrides=False)[0] for path in suite.configs]

    assert [item.rag.strategy for item in loaded] == [
        "vanilla",
        "self_rag",
        "agentic_rag",
        "graph_rag",
    ]
    assert suite.dataset == "data/eval/rag_eval_v2_30.jsonl"
    assert {item.retrieval.query_processor.provider for item in loaded} == {"identity"}
    assert {item.retrieval.evidence_selector for item in loaded} == {"diversified"}
    assert {(item.generation.provider, item.generation.model) for item in loaded} == {
        ("openrouter", "nvidia/nemotron-3.5-content-safety:free")
    }
