from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

TraceStatus = Literal["running", "completed", "failed"]
EventStatus = Literal["started", "completed", "failed"]


class TraceEvent(BaseModel):
    trace_id: str
    seq: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    stage: str
    event: str
    status: EventStatus = "completed"
    elapsed_ms: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class TraceSnapshot(BaseModel):
    trace_id: str
    status: TraceStatus
    created_at: datetime
    updated_at: datetime
    events: list[TraceEvent]
    result: dict[str, Any] | None = None
    error: str | None = None
