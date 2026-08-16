from __future__ import annotations

import sqlite3
from array import array
from pathlib import Path


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        result: dict[str, list[float]] = {}
        with self._connect() as connection:
            for start in range(0, len(keys), 500):
                batch = keys[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT cache_key, dimension, vector FROM embeddings "  # noqa: S608
                    f"WHERE cache_key IN ({placeholders})",
                    batch,
                )
                for cache_key, dimension, payload in rows:
                    values = array("f")
                    values.frombytes(payload)
                    if len(values) == dimension:
                        result[str(cache_key)] = values.tolist()
        return result

    def put_many(self, values: dict[str, list[float]]) -> None:
        if not values:
            return
        rows = []
        for cache_key, vector in values.items():
            payload = array("f", vector).tobytes()
            rows.append((cache_key, len(vector), payload))
        with self._connect() as connection:
            connection.executemany(
                "INSERT OR IGNORE INTO embeddings(cache_key, dimension, vector) VALUES (?, ?, ?)",
                rows,
            )
