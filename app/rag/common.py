from __future__ import annotations

import time
from typing import Any

from app.models import Answer, RetrievalOutcome
from app.rag.base import REFUSAL_TEXT
from app.rag.controller import (
    HeuristicRagController,
    invoke_controller,
)
from app.rag.vanilla import VanillaRagStrategy
from app.tracing import emit_trace


class StrategyBase:
    strategy_name = "advanced"

    def __init__(self, vanilla: VanillaRagStrategy, controller: Any | None = None) -> None:
        self.vanilla = vanilla
        self.controller = controller or HeuristicRagController()

    @property
    def component_id(self) -> str:
        controller_id = getattr(self.controller, "component_id", type(self.controller).__name__)
        return f"{self.strategy_name}:{controller_id}:v1"

    async def _controller(self, names: tuple[str, ...], *args: Any) -> Any:
        for name in names:
            if hasattr(self.controller, name):
                return await invoke_controller(self.controller, name, *args)
        return None

    def _trace(self, event: str, **payload: Any) -> None:
        emit_trace("strategy", event, payload={"strategy": self.strategy_name, **payload})

    @staticmethod
    def _refusal(outcome: RetrievalOutcome, started: float, evidence=None) -> Answer:
        timings = dict(outcome.timings_ms)
        timings["llm"] = 0.0
        timings["total"] = round((time.perf_counter() - started) * 1000, 3)
        return Answer(
            text=REFUSAL_TEXT,
            citations=[],
            evidence=evidence or [],
            timings_ms=timings,
            refused=True,
            citation_validated=True,
        )
