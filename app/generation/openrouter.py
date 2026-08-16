from __future__ import annotations

from openai import AsyncOpenAI

from app.models import SearchResult
from app.settings import GenerationSettings
from app.tracing import emit_trace
from app.tracing.recorder import trace_level

SYSTEM_PROMPT = """你是中文汽车用户手册问答助手。
规则：
1. 只能依据用户消息中提供的证据回答，不使用外部知识。
2. 每个关键结论必须使用 [1]、[2] 形式引用对应证据。
3. 证据不足、冲突或无法回答时，明确回答“知识库中未找到足够依据”。
4. 不得编造操作步骤、数值、车型配置或引用。
5. 回答简洁、准确，优先保留安全警告和适用条件。
"""


class OpenRouterGenerator:
    def __init__(self, settings: GenerationSettings, api_key: str) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=settings.timeout_seconds,
        )

    @property
    def component_id(self) -> str:
        return f"openrouter:{self.settings.model}"

    @staticmethod
    def _evidence(contexts: list[SearchResult]) -> str:
        return "\n\n".join(
            (
                f"[{index}] 标题：{result.chunk.title}\n"
                f"路径：{' > '.join(result.chunk.section_path)}\n"
                f"正文：{result.chunk.text}\n"
                f"来源：{result.chunk.source_url}"
            )
            for index, result in enumerate(contexts, start=1)
        )

    async def _complete(self, user_content: str, *, operation: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        input_payload = {
            "operation": operation,
            "provider": "openrouter",
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "message_chars": [len(message["content"]) for message in messages],
        }
        if trace_level() != "summary":
            input_payload["messages"] = messages
        emit_trace(
            "llm",
            "llm.input",
            status="started",
            payload=input_payload,
        )
        response = await self.client.chat.completions.create(
            model=self.settings.model,
            temperature=self.settings.temperature,
            messages=messages,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenRouter returned an empty answer")
        output = content.strip()
        usage = response.usage.model_dump(mode="json") if response.usage else None
        emit_trace(
            "llm",
            "llm.output",
            payload={
                "operation": operation,
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason,
                "usage": usage,
                "text": output,
            },
        )
        return output

    async def generate(self, question: str, contexts: list[SearchResult]) -> str:
        return await self._complete(
            f"问题：{question}\n\n证据：\n{self._evidence(contexts)}",
            operation="generate",
        )

    async def repair_citations(
        self,
        question: str,
        draft: str,
        contexts: list[SearchResult],
    ) -> str:
        return await self._complete(
            "请修复下面草稿的引用格式，不增加任何新事实。"
            "每个事实性结论后必须标注一个或多个有效引用编号，如 [1] 或 [1][2]；"
            "只能使用给定证据编号。只输出修复后的答案。\n\n"
            f"问题：{question}\n\n草稿：{draft}\n\n证据：\n{self._evidence(contexts)}",
            operation="repair_citations",
        )
