from __future__ import annotations

from pathlib import Path

from app.ingestion.crawler import parse_manual_index
from app.ingestion.parser import LiXiangHtmlParser

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_parse_manual_index_deduplicates_topics() -> None:
    html = (FIXTURES / "index.html").read_text(encoding="utf-8")
    manual_id, vehicle_model, topics, directory_refs = parse_manual_index(
        html,
        "https://example.com/manual/index.html",
    )

    assert manual_id == "TEST-MANUAL-001"
    assert vehicle_model == "理想测试车型"
    assert directory_refs == 3
    assert len(topics) == 2
    assert topics[0].breadcrumbs == ["用车场景", "出行准备", "安全驾驶"]
    assert topics[0].source_url == "https://example.com/manual/topic-safe.html"


def test_parser_preserves_warning_and_image(topic_ref) -> None:
    html = (FIXTURES / "topic-safe.html").read_bytes()
    document = LiXiangHtmlParser().parse(
        html,
        topic_ref,
        manual_id="TEST-MANUAL-001",
        snapshot_id="test-snapshot",
        vehicle_model="理想测试车型",
    )

    assert document is not None
    assert document.sections[0].path == ["安全驾驶", "行车检查"]
    assert any(block.startswith("[警告]") for block in document.sections[0].blocks)
    assert document.sections[0].images[0].url == "https://example.com/manual/img/safe.webp"


def test_parser_returns_none_without_article(topic_ref) -> None:
    document = LiXiangHtmlParser().parse(
        b"<html><body>empty</body></html>",
        topic_ref,
        manual_id="TEST",
        snapshot_id="test",
        vehicle_model="test",
    )
    assert document is None
