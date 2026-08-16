from __future__ import annotations

import math
import uuid
from typing import Protocol

from qdrant_client import QdrantClient, models

from app.models import Chunk, RetrievalFilters, SearchResult
from app.settings import VectorStoreSettings


class VectorStore(Protocol):
    @property
    def component_id(self) -> str: ...

    def ensure_collection(self, vector_size: int) -> None: ...

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(
        self,
        vector: list[float],
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[SearchResult]: ...

    def all_chunks(self, filters: RetrievalFilters | None = None) -> list[Chunk]: ...

    def existing_content_hashes(self, chunks: list[Chunk]) -> dict[str, str]: ...

    def delete_chunks(self, chunk_ids: list[str]) -> None: ...

    def count(self) -> int: ...

    def health(self) -> bool: ...


def _chunk_payload(chunk: Chunk) -> dict[str, object]:
    return chunk.model_dump(mode="json")


def _chunk_from_payload(payload: dict[str, object]) -> Chunk:
    return Chunk.model_validate(payload)


class QdrantVectorStore:
    PAYLOAD_INDEX_FIELDS = (
        "snapshot_id",
        "manual_id",
        "vehicle_model",
        "manual_key",
        "manual_name",
        "topic_id",
    )

    def __init__(
        self,
        settings: VectorStoreSettings,
        client: QdrantClient | None = None,
    ) -> None:
        self.settings = settings
        self.client = client or QdrantClient(
            url=settings.url,
            timeout=settings.timeout_seconds,
        )

    @property
    def component_id(self) -> str:
        return f"qdrant:{self.settings.collection}"

    def ensure_collection(self, vector_size: int) -> None:
        if not self.client.collection_exists(self.settings.collection):
            self.client.create_collection(
                collection_name=self.settings.collection,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            for field in self.PAYLOAD_INDEX_FIELDS:
                self.client.create_payload_index(
                    collection_name=self.settings.collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            return

        info = self.client.get_collection(self.settings.collection)
        vector_config = info.config.params.vectors
        existing_size = getattr(vector_config, "size", None)
        if existing_size is not None and existing_size != vector_size:
            raise ValueError(
                f"collection vector size mismatch: expected {existing_size}, got {vector_size}"
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        points = [
            models.PointStruct(
                id=str(uuid.UUID(chunk.chunk_id)),
                vector=vector,
                payload=_chunk_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if points:
            self.client.upsert(
                collection_name=self.settings.collection,
                points=points,
                wait=True,
            )

    @staticmethod
    def _filter(filters: RetrievalFilters | None) -> models.Filter | None:
        if filters is None:
            return None
        conditions: list[models.FieldCondition] = []
        for field, value in (
            ("manual_id", filters.manual_id),
            ("snapshot_id", filters.snapshot_id),
            ("vehicle_model", filters.vehicle_model),
        ):
            if value:
                conditions.append(
                    models.FieldCondition(key=field, match=models.MatchValue(value=value))
                )
        if filters.topic_ids:
            conditions.append(
                models.FieldCondition(
                    key="topic_id",
                    match=models.MatchAny(any=filters.topic_ids),
                )
            )
        for field, values in (
            ("manual_key", filters.manual_keys),
            ("manual_name", filters.manual_names),
        ):
            if values:
                conditions.append(
                    models.FieldCondition(key=field, match=models.MatchAny(any=values))
                )
        return models.Filter(must=conditions) if conditions else None

    def search(
        self,
        vector: list[float],
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[SearchResult]:
        response = self.client.query_points(
            collection_name=self.settings.collection,
            query=vector,
            query_filter=self._filter(filters),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        results: list[SearchResult] = []
        for rank, point in enumerate(response.points, start=1):
            payload = dict(point.payload or {})
            results.append(
                SearchResult(
                    chunk=_chunk_from_payload(payload),
                    score=float(point.score),
                    rank=rank,
                    recall_score=float(point.score),
                    retriever_id=self.component_id,
                )
            )
        return results

    def all_chunks(self, filters: RetrievalFilters | None = None) -> list[Chunk]:
        chunks: list[Chunk] = []
        offset: models.ExtendedPointId | None = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.settings.collection,
                scroll_filter=self._filter(filters),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            chunks.extend(_chunk_from_payload(dict(record.payload or {})) for record in records)
            if offset is None:
                return chunks

    def existing_content_hashes(self, chunks: list[Chunk]) -> dict[str, str]:
        if not chunks or not self.client.collection_exists(self.settings.collection):
            return {}
        result: dict[str, str] = {}
        for start in range(0, len(chunks), 256):
            batch = chunks[start : start + 256]
            records = self.client.retrieve(
                collection_name=self.settings.collection,
                ids=[str(uuid.UUID(chunk.chunk_id)) for chunk in batch],
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                payload = dict(record.payload or {})
                chunk_id = str(payload.get("chunk_id", record.id))
                result[chunk_id] = str(payload.get("content_hash", ""))
        return result

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids or not self.client.collection_exists(self.settings.collection):
            return
        self.client.delete(
            collection_name=self.settings.collection,
            points_selector=models.PointIdsList(
                points=[str(uuid.UUID(chunk_id)) for chunk_id in chunk_ids]
            ),
            wait=True,
        )

    def count(self) -> int:
        if not self.client.collection_exists(self.settings.collection):
            return 0
        return int(
            self.client.count(
                collection_name=self.settings.collection,
                exact=True,
            ).count
        )

    def health(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:  # noqa: BLE001 - health must return a boolean
            return False


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.vector_size: int | None = None
        self.points: dict[str, tuple[list[float], Chunk]] = {}

    @property
    def component_id(self) -> str:
        return "in-memory-v1"

    def ensure_collection(self, vector_size: int) -> None:
        if self.vector_size is not None and self.vector_size != vector_size:
            raise ValueError("vector size mismatch")
        self.vector_size = vector_size

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.ensure_collection(len(vector))
            self.points[chunk.chunk_id] = (vector, chunk)

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

    @staticmethod
    def _matches(chunk: Chunk, filters: RetrievalFilters | None) -> bool:
        if filters is None:
            return True
        return not (
            (filters.manual_id and chunk.manual_id != filters.manual_id)
            or (filters.snapshot_id and chunk.snapshot_id != filters.snapshot_id)
            or (filters.vehicle_model and chunk.vehicle_model != filters.vehicle_model)
            or (filters.manual_keys and chunk.manual_key not in filters.manual_keys)
            or (filters.manual_names and chunk.manual_name not in filters.manual_names)
            or (filters.topic_ids and chunk.topic_id not in filters.topic_ids)
        )

    def search(
        self,
        vector: list[float],
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[SearchResult]:
        if self.vector_size is not None and len(vector) != self.vector_size:
            raise ValueError("vector size mismatch")
        ranked = sorted(
            (
                (self._cosine(vector, stored_vector), chunk)
                for stored_vector, chunk in self.points.values()
                if self._matches(chunk, filters)
            ),
            key=lambda item: item[0],
            reverse=True,
        )[:top_k]
        return [
            SearchResult(
                chunk=chunk,
                score=score,
                rank=rank,
                recall_score=score,
                retriever_id=self.component_id,
            )
            for rank, (score, chunk) in enumerate(ranked, start=1)
        ]

    def all_chunks(self, filters: RetrievalFilters | None = None) -> list[Chunk]:
        return [chunk for _, chunk in self.points.values() if self._matches(chunk, filters)]

    def existing_content_hashes(self, chunks: list[Chunk]) -> dict[str, str]:
        return {
            chunk.chunk_id: stored_chunk.content_hash
            for chunk in chunks
            if (stored := self.points.get(chunk.chunk_id)) is not None
            for stored_chunk in [stored[1]]
        }

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            self.points.pop(chunk_id, None)

    def count(self) -> int:
        return len(self.points)

    def health(self) -> bool:
        return True
