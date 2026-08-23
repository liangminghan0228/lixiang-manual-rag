from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.wiring import build_container


def _write_config(tmp_path: Path, updates: dict[str, object]) -> str:
    base: dict[str, object] = {
        "data": {
            "source_url": "https://example.com",
            "snapshot_id": "test-snapshot",
        },
        "embedding": {"provider": "hash_mock"},
        "vector_store": {"provider": "in_memory"},
        "retrieval": {
            "provider": "dense",
            "top_k": 5,
            "candidate_top_k": 10,
            "evidence_top_k": 3,
        },
        "generation": {
            "provider": "mock",
            "model": "mock",
            "mock_when_key_missing": True,
        },
    }

    for section, values in updates.items():
        if section in base and isinstance(base[section], dict) and isinstance(values, dict):
            base[section].update(values)
        else:
            base[section] = values

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return str(path)


def test_wired_retriever_variants_are_registered_by_name(tmp_path) -> None:
    dense = build_container(_write_config(tmp_path, {"retrieval": {"provider": "dense"}}))
    assert dense.candidate_retriever.component_id.startswith("dense:")
    assert dense.retriever.candidate_retriever is dense.candidate_retriever
    assert dense.retriever.query_processor is dense.query_processor
    assert dense.retriever.reranker is dense.reranker

    bm25 = build_container(_write_config(tmp_path, {"retrieval": {"provider": "bm25"}}))
    assert bm25.candidate_retriever.component_id.startswith("bm25-")

    hybrid = build_container(_write_config(tmp_path, {"retrieval": {"provider": "hybrid"}}))
    assert hybrid.candidate_retriever.component_id == "hybrid-rrf-60-v1"


def test_wired_generator_variants_are_registered_by_name(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    mock_container = build_container(
        _write_config(
            tmp_path,
            {"generation": {"provider": "mock", "model": "mock"}},
        )
    )
    assert mock_container.generator.component_id == "mock-generator-v1"

    openrouter_mocked = build_container(
        _write_config(
            tmp_path,
            {
                "generation": {
                    "provider": "openrouter",
                    "mock_when_key_missing": True,
                    "model": "openrouter/free",
                }
            },
        )
    )
    assert openrouter_mocked.generator.component_id == "mock-generator-v1"


def test_openrouter_generates_real_generator_with_runtime_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/free")
    openrouter = build_container(
        _write_config(
            tmp_path,
            {
                "generation": {
                    "provider": "openrouter",
                    "mock_when_key_missing": False,
                    "model": "openrouter/free",
                }
            },
        )
    )
    assert openrouter.generator.component_id == "openrouter:openrouter/free"


def test_unknown_retriever_provider_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported retriever: unknown"):
        build_container(_write_config(tmp_path, {"retrieval": {"provider": "unknown"}}))


def test_unknown_generator_provider_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported generator: unknown"):
        build_container(_write_config(tmp_path, {"generation": {"provider": "unknown"}}))


def test_openrouter_without_key_raises_when_mock_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"):
        build_container(
            _write_config(
                tmp_path,
                {
                    "generation": {
                        "provider": "openrouter",
                        "model": "openrouter/free",
                        "mock_when_key_missing": False,
                    }
                },
            )
        )


def test_query_processor_variants_are_config_driven(tmp_path, monkeypatch) -> None:
    normalized = build_container(
        _write_config(
            tmp_path,
            {"retrieval": {"query_processor": {"provider": "normalize"}}},
        )
    )
    assert normalized.query_processor.component_id == "normalizing-v1"

    monkeypatch.setenv("QUERY_OPTIMIZER_API_KEY", "query-key")
    monkeypatch.setenv("QUERY_OPTIMIZER_MODEL", "provider/fixed-query-model")
    rewritten = build_container(
        _write_config(
            tmp_path,
            {"retrieval": {"query_processor": {"provider": "rewrite"}}},
        )
    )
    assert "provider/fixed-query-model" in rewritten.query_processor.component_id


def test_model_backed_query_processor_requires_fixed_model(tmp_path, monkeypatch) -> None:
    # Empty environment values must override any developer-local .env file.
    monkeypatch.setenv("QUERY_OPTIMIZER_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    with pytest.raises(ValueError, match="QUERY_OPTIMIZER_API_KEY"):
        build_container(
            _write_config(
                tmp_path,
                {"retrieval": {"query_processor": {"provider": "multi_query"}}},
            )
        )


def test_rag_strategy_variants_are_config_driven(tmp_path, monkeypatch) -> None:
    graph = build_container(_write_config(tmp_path, {"rag": {"strategy": "graph_rag"}}))
    assert graph.rag_strategy.component_id == "graph-rag:document-structure:v1"

    monkeypatch.setenv("OPENROUTER_API_KEY", "controller-key")
    monkeypatch.setenv("RAG_CONTROLLER_MODEL", "provider/fixed-controller")
    self_rag = build_container(_write_config(tmp_path, {"rag": {"strategy": "self_rag"}}))
    assert self_rag.rag_strategy.component_id.startswith("self_rag:openrouter-controller:")


def test_unknown_rag_strategy_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported RAG strategy"):
        build_container(_write_config(tmp_path, {"rag": {"strategy": "unknown"}}))
