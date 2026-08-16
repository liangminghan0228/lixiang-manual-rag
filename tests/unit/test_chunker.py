from __future__ import annotations

from app.ingestion.chunker import HeadingChunker


def test_chunker_is_deterministic(sample_document) -> None:
    chunker = HeadingChunker(target_chars=120, overlap_chars=20)
    first = chunker.split(sample_document)
    second = chunker.split(sample_document)

    assert first
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].vehicle_model == "理想测试车型"
    assert "路径：" in first[0].text
    assert first[0].metadata["chunker"] == "heading-120-20-v1"


def test_chunker_rejects_invalid_overlap() -> None:
    try:
        HeadingChunker(target_chars=100, overlap_chars=100)
    except ValueError as exc:
        assert "overlap_chars" in str(exc)
    else:
        raise AssertionError("expected invalid overlap to fail")
