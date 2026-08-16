from __future__ import annotations

from app.ingestion.embedding_cache import EmbeddingCache


def test_embedding_cache_round_trip(tmp_path) -> None:
    cache = EmbeddingCache(tmp_path / "vectors.sqlite3")
    cache.put_many({"model:hash": [0.25, -0.5, 1.0]})

    assert cache.get_many(["missing", "model:hash"]) == {"model:hash": [0.25, -0.5, 1.0]}
