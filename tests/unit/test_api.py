from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.wiring import build_container


def test_api_health_retrieve_and_chat(sample_chunk) -> None:
    container = build_container("configs/test.yaml")
    vectors = container.embedder.embed_documents([sample_chunk.text])
    container.vector_store.upsert([sample_chunk], vectors)

    with TestClient(create_app(container)) as client:
        health = client.get("/health")
        retrieve = client.post(
            "/v1/retrieve",
            json={
                "question": "驾驶前检查车灯",
                "filters": {"topic_ids": ["topic-safe"]},
            },
        )
        chat = client.post("/v1/chat", json={"question": "驾驶前需要检查什么？"})
        metrics = client.get("/metrics")

    assert health.status_code == 200
    assert health.json()["generator_mock"] is True
    assert retrieve.status_code == 200
    assert retrieve.json()["results"][0]["chunk"]["topic_id"] == "topic-safe"
    assert chat.status_code == 200
    assert chat.json()["citations"][0]["source_url"].startswith("https://")
    assert metrics.status_code == 200
    assert "rag_http_requests_total" in metrics.text


def test_api_validates_empty_question() -> None:
    container = build_container("configs/test.yaml")
    with TestClient(create_app(container)) as client:
        response = client.post("/v1/retrieve", json={"question": ""})
    assert response.status_code == 422


def test_api_lists_discovered_manuals(tmp_path) -> None:
    container = build_container("configs/test.yaml")
    container.settings.data.raw_dir = tmp_path / "raw"
    container.settings.experiment.manifest_dir = tmp_path / "reports"
    container.settings.data.raw_dir.mkdir()
    (container.settings.data.raw_dir / "catalog.json").write_text(
        json.dumps(
            [
                {
                    "catalog_id": 494,
                    "name": "理想i8",
                    "url": (
                        "https://manuals.lixiang.com/zh-cn/W022025ULTRA/20260422091942/index.html"
                    ),
                    "publish_date": "2026-04-22T09:19:42",
                    "version": "8.5.0",
                    "manual_key": "W022025ULTRA",
                    "snapshot_id": "20260422091942",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with TestClient(create_app(container)) as client:
        response = client.get("/v1/manuals")

    assert response.status_code == 200
    assert response.json()[0]["catalog"]["manual_key"] == "W022025ULTRA"
    assert response.json()[0]["status"] == "discovered"


def test_trace_api_streams_full_rag_chain(sample_chunk) -> None:
    container = build_container("configs/test.yaml")
    vectors = container.embedder.embed_documents([sample_chunk.text])
    container.vector_store.upsert([sample_chunk], vectors)

    with TestClient(create_app(container)) as client:
        created = client.post(
            "/v1/traces",
            json={"question": "驾驶前需要检查什么？", "trace_level": "sampled"},
        )
        assert created.status_code == 202
        trace_id = created.json()["trace_id"]

        snapshot = None
        for _ in range(100):
            snapshot = client.get(f"/v1/traces/{trace_id}").json()
            if snapshot["status"] != "running":
                break
            time.sleep(0.01)

        assert snapshot is not None
        assert snapshot["status"] == "completed"
        event_names = [event["event"] for event in snapshot["events"]]
        assert event_names == [
            "request.received",
            "request.normalized",
            "embedding.started",
            "embedding.completed",
            "retrieval.dense.completed",
            "rerank.completed",
            "retrieval.pipeline.completed",
            "evidence.completed",
            "llm.input",
            "llm.output",
            "citation.validated",
            "response.completed",
        ]
        embedding = next(
            event for event in snapshot["events"] if event["event"] == "embedding.completed"
        )
        assert embedding["payload"]["dimension"] == 64
        assert embedding["payload"]["preview"]

        streamed = client.get(f"/v1/traces/{trace_id}/events")
        assert streamed.status_code == 200
        assert "event: embedding.completed" in streamed.text
        assert "event: response.completed" in streamed.text

        console = client.get("/trace-console")
        assert console.status_code == 200
        assert "RAG Request Trace" in console.text
