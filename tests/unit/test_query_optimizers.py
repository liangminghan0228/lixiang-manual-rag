import json

import pytest

from app.models import RetrievalFilters
from app.retrieval.query import (
    DecompositionQueryProcessor,
    ExpansionQueryProcessor,
    HyDEQueryProcessor,
    LLMRewriteQueryProcessor,
    MultiQueryProcessor,
    NormalizingQueryProcessor,
)


class FakePlanner:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def plan(self, question, *, strategy):
        self.calls.append((question, strategy))
        return json.dumps(self.value, ensure_ascii=False)


def test_normalizing_and_expansion_preserve_filters():
    filters = RetrievalFilters(topic_ids=["safe"])
    assert NormalizingQueryProcessor().process("  车灯，  检查！ ", filters).text == "车灯 检查"
    plan = ExpansionQueryProcessor({"灯": ["照明灯", "灯光"]}).process("检查 灯", filters)
    assert plan.filters == filters
    assert plan.queries == ["检查 灯 照明灯 灯光"]
    assert plan.fusion_strategy == "single_query"


@pytest.mark.parametrize(
    ("processor", "expected_strategy", "expected_queries"),
    [
        (LLMRewriteQueryProcessor(FakePlanner({"query": "改写"})), "single_query", ["改写"]),
        (
            HyDEQueryProcessor(FakePlanner({"hypothetical_document": "假设文档"})),
            "single_query",
            ["假设文档"],
        ),
        (MultiQueryProcessor(FakePlanner({"queries": ["一", "二"]})), "rrf", ["一", "二"]),
        (
            DecompositionQueryProcessor(FakePlanner({"queries": ["子问题一", "子问题二"]})),
            "rrf",
            ["子问题一", "子问题二"],
        ),
    ],
)
def test_llm_processors_use_structured_plan(processor, expected_strategy, expected_queries):
    plan = processor.process("原始问题", RetrievalFilters(manual_id="m"))
    assert plan.fusion_strategy == expected_strategy
    assert plan.queries == expected_queries
    assert plan.filters.manual_id == "m"
    assert plan.metadata["planner_elapsed_ms"] >= 0


def test_llm_malformed_output_is_explicit_error():
    class Bad:
        def plan(self, question, *, strategy):
            return "not-json"

    with pytest.raises(ValueError, match="query planner failed"):
        LLMRewriteQueryProcessor(Bad()).process("问题")
