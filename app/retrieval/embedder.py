from __future__ import annotations

import hashlib
import math
import re
import threading
from typing import Any, Protocol

from app.settings import EmbeddingSettings


class Embedder(Protocol):
    @property
    def component_id(self) -> str: ...

    @property
    def is_loaded(self) -> bool: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class BgeM3Embedder:
    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._lock = threading.Lock()

    @property
    def component_id(self) -> str:
        return f"bge-m3:{self.settings.model}"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    from FlagEmbedding import BGEM3FlagModel
                except ImportError as exc:  # pragma: no cover - depends on optional runtime
                    raise RuntimeError(
                        "FlagEmbedding is not installed; run `uv sync` before using BGE-M3"
                    ) from exc
                self._model = BGEM3FlagModel(
                    self.settings.model,
                    use_fp16=self.settings.use_fp16,
                )
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        with self._lock:
            output = model.encode(
                texts,
                batch_size=self.settings.batch_size,
                max_length=self.settings.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        dense = output["dense_vecs"]
        return dense.tolist() if hasattr(dense, "tolist") else [list(item) for item in dense]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        vectors = self._encode([text])
        if not vectors:
            raise ValueError("query must not be empty")
        return vectors[0]


class DeterministicHashEmbedder:
    """Small deterministic embedder for tests and offline API development."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    @property
    def component_id(self) -> str:
        return f"hash-mock-{self.dimension}-v1"

    @property
    def is_loaded(self) -> bool:
        return True

    def _embed(self, text: str) -> list[float]:
        values = [0.0] * self.dimension
        normalized = re.sub(r"\s+", "", text.lower())
        chinese_or_chars = [
            normalized[index : index + 2] for index in range(max(len(normalized) - 1, 0))
        ]
        words = re.findall(r"[a-z0-9]+", text.lower())
        tokens = [*chinese_or_chars, *words]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            values[index] += sign
        if not any(values):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            for index, byte in enumerate(digest):
                values[index % self.dimension] += (byte - 127.5) / 127.5
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("query must not be empty")
        return self._embed(text)
