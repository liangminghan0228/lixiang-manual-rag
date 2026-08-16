from __future__ import annotations

from app.models import SearchResult
from app.tracing import emit_trace


class MockGenerator:
    @property
    def component_id(self) -> str:
        return "mock-generator-v1"

    async def generate(self, question: str, contexts: list[SearchResult]) -> str:
        emit_trace(
            "llm",
            "llm.input",
            status="started",
            payload={
                "operation": "generate",
                "provider": "mock",
                "model": self.component_id,
                "question": question,
                "evidence_chunk_ids": [item.chunk.chunk_id for item in contexts],
            },
        )
        if not contexts:
            return "知识库中未找到足够依据。"
        first = contexts[0].chunk.text
        body = first.split("正文：", 1)[-1].strip()
        summary = body[:220].rstrip()
        if len(body) > len(summary):
            summary += "……"
        output = f"根据知识库正文，{summary} [1]"
        emit_trace(
            "llm",
            "llm.output",
            payload={"operation": "generate", "model": self.component_id, "text": output},
        )
        return output

    async def repair_citations(
        self,
        question: str,
        draft: str,
        contexts: list[SearchResult],
    ) -> str:
        emit_trace(
            "llm",
            "llm.input",
            status="started",
            payload={
                "operation": "repair_citations",
                "provider": "mock",
                "model": self.component_id,
                "question": question,
                "draft": draft,
                "evidence_chunk_ids": [item.chunk.chunk_id for item in contexts],
            },
        )
        output = draft if "[1]" in draft or not contexts else f"{draft.rstrip()} [1]"
        emit_trace(
            "llm",
            "llm.output",
            payload={
                "operation": "repair_citations",
                "model": self.component_id,
                "text": output,
            },
        )
        return output
