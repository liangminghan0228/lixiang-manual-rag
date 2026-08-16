from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

from app.tracing.models import EventStatus
from app.tracing.store import InMemoryTraceStore


class TraceRecorder:
    def __init__(
        self,
        trace_id: str,
        store: InMemoryTraceStore,
        *,
        level: str,
        options: dict[str, int] | None = None,
    ) -> None:
        self.trace_id = trace_id
        self.store = store
        self.level = level
        self.options = options or {}

    def emit(
        self,
        stage: str,
        event: str,
        *,
        status: EventStatus = "completed",
        elapsed_ms: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.store.append(
            self.trace_id,
            stage=stage,
            event=event,
            status=status,
            elapsed_ms=elapsed_ms,
            payload=payload,
        )


_CURRENT_TRACE: ContextVar[TraceRecorder | None] = ContextVar(
    "rag_current_trace",
    default=None,
)


def set_trace(recorder: TraceRecorder) -> Token[TraceRecorder | None]:
    return _CURRENT_TRACE.set(recorder)


def reset_trace(token: Token[TraceRecorder | None]) -> None:
    _CURRENT_TRACE.reset(token)


def current_trace() -> TraceRecorder | None:
    return _CURRENT_TRACE.get()


def trace_option(name: str, default: int) -> int:
    recorder = current_trace()
    return recorder.options.get(name, default) if recorder is not None else default


def trace_level() -> str:
    recorder = current_trace()
    return recorder.level if recorder is not None else "summary"


def emit_trace(
    stage: str,
    event: str,
    *,
    status: EventStatus = "completed",
    elapsed_ms: float | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    recorder = current_trace()
    if recorder is not None:
        recorder.emit(
            stage,
            event,
            status=status,
            elapsed_ms=elapsed_ms,
            payload=payload,
        )
