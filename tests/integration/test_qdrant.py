from __future__ import annotations

import os
import uuid

import pytest

from app.models import Chunk
from app.retrieval.vector_store import QdrantVectorStore
from app.settings import VectorStoreSettings

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_QDRANT_INTEGRATION") != "1",
    reason="set RUN_QDRANT_INTEGRATION=1 with Qdrant running",
)


def test_qdrant_upsert_and_search() -> None:
    collection = f"lixiang_mvp_test_{uuid.uuid4().hex}"
    store = QdrantVectorStore(
        VectorStoreSettings(url="http://localhost:6333", collection=collection)
    )
    chunk = Chunk(
        chunk_id=str(uuid.uuid4()),
        document_id="doc",
        manual_id="manual",
        snapshot_id="snapshot",
        vehicle_model="test",
        topic_id="topic",
        title="title",
        text="text",
        section_path=["section"],
        source_url="https://example.com",
        content_hash="hash",
    )
    vector = [1.0, 0.0, 0.0, 0.0]
    try:
        store.ensure_collection(len(vector))
        store.upsert([chunk], [vector])

        assert store.count() == 1
        assert store.search(vector, 1)[0].chunk.chunk_id == chunk.chunk_id
    finally:
        if store.client.collection_exists(collection):
            store.client.delete_collection(collection)
