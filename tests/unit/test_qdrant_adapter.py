from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient

from app.models import Chunk
from app.retrieval.vector_store import QdrantVectorStore
from app.settings import VectorStoreSettings


def test_qdrant_adapter_collection_upsert_search_and_count() -> None:
    store = QdrantVectorStore(
        VectorStoreSettings(url="memory", collection="adapter-test"),
        client=QdrantClient(":memory:"),
    )
    chunk = Chunk(
        chunk_id=str(uuid.uuid4()),
        document_id="doc",
        manual_id="manual",
        snapshot_id="snapshot",
        vehicle_model="i8",
        topic_id="topic-safe",
        title="安全驾驶",
        text="驾驶前检查车辆。",
        section_path=["安全驾驶"],
        source_url="https://example.com",
        content_hash="hash",
    )
    vector = [1.0, 0.0, 0.0, 0.0]

    with pytest.warns(UserWarning, match="Payload indexes have no effect"):
        store.ensure_collection(len(vector))
    store.upsert([chunk], [vector])
    store.upsert([chunk], [vector])
    result = store.search(vector, 1)[0]

    assert store.count() == 1
    assert result.score == 1.0
    assert result.chunk.topic_id == "topic-safe"
