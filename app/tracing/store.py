from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from app.tracing.models import EventStatus, TraceEvent, TraceSnapshot, TraceStatus


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class _TraceState:
    trace_id: str
    status: TraceStatus = "running"
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    events: list[TraceEvent] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None


class InMemoryTraceStore:
    def __init__(
        self,
        *,
        ttl_seconds: int = 1800,
        max_traces: int = 100,
        max_events_per_trace: int = 200,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_traces = max_traces
        self.max_events_per_trace = max_events_per_trace
        self._traces: dict[str, _TraceState] = {}
        self._lock = threading.Lock()

    def create(self) -> TraceSnapshot:
        with self._lock:
            self._cleanup_locked()
            while len(self._traces) >= self.max_traces:
                oldest = min(self._traces.values(), key=lambda item: item.created_at)
                del self._traces[oldest.trace_id]
            trace_id = uuid.uuid4().hex
            state = _TraceState(trace_id=trace_id)
            self._traces[trace_id] = state
            return self._snapshot_locked(state)

    def append(
        self,
        trace_id: str,
        *,
        stage: str,
        event: str,
        status: EventStatus = "completed",
        elapsed_ms: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> TraceEvent:
        with self._lock:
            state = self._require_locked(trace_id)
            trace_event = TraceEvent(
                trace_id=trace_id,
                seq=len(state.events) + 1,
                stage=stage,
                event=event,
                status=status,
                elapsed_ms=elapsed_ms,
                payload=payload or {},
            )
            if len(state.events) >= self.max_events_per_trace:
                raise RuntimeError(f"trace event limit exceeded: {trace_id}")
            state.events.append(trace_event)
            state.updated_at = trace_event.timestamp
            return trace_event

    def complete(self, trace_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            state = self._require_locked(trace_id)
            state.status = "completed"
            state.result = result
            state.updated_at = _now()

    def fail(self, trace_id: str, error: str) -> None:
        with self._lock:
            state = self._require_locked(trace_id)
            state.status = "failed"
            state.error = error
            state.updated_at = _now()

    def get(self, trace_id: str) -> TraceSnapshot | None:
        with self._lock:
            self._cleanup_locked()
            state = self._traces.get(trace_id)
            return self._snapshot_locked(state) if state is not None else None

    def events_after(self, trace_id: str, seq: int) -> tuple[list[TraceEvent], TraceStatus]:
        with self._lock:
            state = self._require_locked(trace_id)
            return [event for event in state.events if event.seq > seq], state.status

    def _require_locked(self, trace_id: str) -> _TraceState:
        try:
            return self._traces[trace_id]
        except KeyError as exc:
            raise KeyError(f"trace not found: {trace_id}") from exc

    @staticmethod
    def _snapshot_locked(state: _TraceState) -> TraceSnapshot:
        return TraceSnapshot(
            trace_id=state.trace_id,
            status=state.status,
            created_at=state.created_at,
            updated_at=state.updated_at,
            events=list(state.events),
            result=state.result,
            error=state.error,
        )

    def _cleanup_locked(self) -> None:
        expires_before = _now() - timedelta(seconds=self.ttl_seconds)
        expired = [
            trace_id
            for trace_id, state in self._traces.items()
            if state.updated_at < expires_before
        ]
        for trace_id in expired:
            del self._traces[trace_id]
