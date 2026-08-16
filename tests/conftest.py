from __future__ import annotations

from pathlib import Path

import pytest

from app.models import Chunk, Document, Section, TopicRef

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def topic_ref() -> TopicRef:
    return TopicRef(
        topic_id="topic-safe",
        title="安全驾驶",
        breadcrumbs=["用车场景", "出行准备", "安全驾驶"],
        source_url="https://example.com/manual/topic-safe.html",
        source_file="topic-safe.html",
    )


@pytest.fixture
def sample_document() -> Document:
    return Document(
        document_id="TEST:test-snapshot:topic-safe",
        manual_id="TEST",
        snapshot_id="test-snapshot",
        vehicle_model="理想测试车型",
        topic_id="topic-safe",
        title="安全驾驶",
        breadcrumb=["用车场景", "出行准备", "安全驾驶"],
        source_url="https://example.com/manual/topic-safe.html",
        sections=[
            Section(
                title="行车检查",
                level=2,
                path=["安全驾驶", "行车检查"],
                blocks=[
                    "每次驾驶前，请确认车灯正常工作并检查车辆周边环境。",
                    "- 确认所有车窗清晰。",
                    "[警告] 请勿穿高跟鞋或拖鞋驾驶车辆。",
                ],
            )
        ],
        content_hash="document-hash",
    )


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(
        chunk_id="30f4edc4-724a-5c58-bdd6-53be92c66a3a",
        document_id="TEST:test-snapshot:topic-safe",
        manual_id="TEST",
        snapshot_id="test-snapshot",
        vehicle_model="理想测试车型",
        topic_id="topic-safe",
        title="安全驾驶",
        text="车型：理想测试车型\n路径：安全驾驶 > 行车检查\n正文：驾驶前检查车灯和车辆周边。",
        section_path=["安全驾驶", "行车检查"],
        source_url="https://example.com/manual/topic-safe.html",
        content_hash="chunk-hash",
    )
