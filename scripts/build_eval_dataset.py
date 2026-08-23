from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from app.evaluation import EvaluationCase
from app.models import Chunk, RetrievalFilters
from app.settings import PROJECT_ROOT

QUESTION_COUNTS = {
    "single_chunk": 18,
    "multi_chunk_same_topic": 8,
    "multi_topic": 10,
    "cross_manual": 6,
    "unanswerable": 8,
}

UNANSWERABLE_QUESTIONS = [
    "这辆车的终身免费保养政策是什么？",
    "下一次软件更新的准确发布日期是什么？",
    "车辆保险明年的具体费用是多少？",
    "附近维修门店今天几点关门？",
    "车主上个月一共行驶了多少公里？",
    "我的订单预计哪天交付？",
    "未来五年的充电价格分别是多少？",
    "尚未发布的下一代车型会新增哪些配置？",
]

TAG_PATTERNS = {
    "safety": r"安全|警告|危险|碰撞|气囊|儿童",
    "operation": r"开启|关闭|使用|操作|设置|调节|解锁|锁止",
    "driving": r"驾驶|行驶|制动|泊车|方向盘|辅助驾驶",
    "energy": r"充电|电池|电量|续航|能耗",
    "maintenance": r"保养|维护|检查|轮胎|清洁|故障",
    "comfort": r"座椅|空调|后视镜|车窗|娱乐|音响",
}

GENERIC_SECTIONS = {"前言", "目录", "概述", "说明", "注意事项"}


def _content_tag(chunk: Chunk) -> str:
    text = " ".join([chunk.title, *chunk.section_path, chunk.text])
    return next(
        (tag for tag, pattern in TAG_PATTERNS.items() if re.search(pattern, text)),
        "general",
    )


def _body(chunk: Chunk, limit: int = 700) -> str:
    body = chunk.text.split("正文：", 1)[-1].strip()
    if len(body) <= limit:
        return body
    candidate = body[:limit]
    boundary = max(candidate.rfind("。"), candidate.rfind("；"), candidate.rfind("\n"))
    return candidate[: boundary + 1].strip() if boundary >= limit // 2 else candidate.rstrip() + "…"


def _section(chunk: Chunk) -> str:
    return chunk.section_path[-1] if chunk.section_path else chunk.title


def _manual(chunk: Chunk) -> str:
    return chunk.manual_name or chunk.vehicle_model


def _reference(chunks: list[Chunk]) -> str:
    return "\n".join(f"{_manual(chunk)}《{_section(chunk)}》：{_body(chunk)}" for chunk in chunks)


def _case(
    case_id: str,
    question: str,
    question_type: str,
    chunks: list[Chunk],
    *tags: str,
) -> EvaluationCase:
    manual_keys = list(dict.fromkeys(chunk.manual_key for chunk in chunks if chunk.manual_key))
    return EvaluationCase(
        id=case_id,
        user_input=question,
        question_type=question_type,
        reference=_reference(chunks),
        reference_contexts=[chunk.text for chunk in chunks],
        gold_chunk_ids=[chunk.chunk_id for chunk in chunks],
        retrieval_filters=RetrievalFilters(manual_keys=manual_keys),
        tags=list(dict.fromkeys([*tags, *(_content_tag(chunk) for chunk in chunks)])),
        label_status="generated_reference_review_required",
    )


def _load_chunks(root: Path) -> list[Chunk]:
    paths = sorted(root.glob("*/*/chunks.jsonl"))
    if not paths:
        raise RuntimeError(f"no per-manual chunks found under {root}")
    chunks: dict[str, Chunk] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                chunk = Chunk.model_validate_json(line)
                chunks.setdefault(chunk.chunk_id, chunk)
    return sorted(
        (
            chunk
            for chunk in chunks.values()
            if chunk.manual_key
            and chunk.section_path
            and _section(chunk) not in GENERIC_SECTIONS
            and len(_body(chunk)) >= 80
        ),
        key=lambda item: (
            item.manual_key or "",
            item.topic_id,
            "/".join(item.section_path),
            item.chunk_id,
        ),
    )


def _take_diverse_singles(chunks: list[Chunk], count: int, used: set[str]) -> list[Chunk]:
    buckets: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        buckets[_content_tag(chunk)].append(chunk)
    selected: list[Chunk] = []
    seen_topics: set[tuple[str | None, str]] = set()
    tags = sorted(buckets)
    while len(selected) < count:
        made_progress = False
        for tag in tags:
            while buckets[tag]:
                chunk = buckets[tag].pop(0)
                topic_key = (chunk.manual_key, chunk.topic_id)
                if chunk.chunk_id in used or topic_key in seen_topics:
                    continue
                selected.append(chunk)
                used.add(chunk.chunk_id)
                seen_topics.add(topic_key)
                made_progress = True
                break
            if len(selected) == count:
                break
        if not made_progress:
            raise RuntimeError(f"could only select {len(selected)}/{count} diverse single cases")
    return selected


def _same_topic_pairs(chunks: list[Chunk], count: int, used: set[str]) -> list[list[Chunk]]:
    grouped: dict[tuple[str | None, str], list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[(chunk.manual_key, chunk.topic_id)].append(chunk)
    pairs: list[list[Chunk]] = []
    for items in grouped.values():
        candidates = [chunk for chunk in items if chunk.chunk_id not in used]
        if len(candidates) < 2 or _section(candidates[0]) == _section(candidates[1]):
            continue
        pair = candidates[:2]
        pairs.append(pair)
        used.update(chunk.chunk_id for chunk in pair)
        if len(pairs) == count:
            return pairs
    raise RuntimeError(f"could only select {len(pairs)}/{count} same-topic pairs")


def _multi_topic_pairs(chunks: list[Chunk], count: int, used: set[str]) -> list[list[Chunk]]:
    by_manual: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        if chunk.manual_key:
            by_manual[chunk.manual_key].append(chunk)
    pairs: list[list[Chunk]] = []
    for manual_key in sorted(by_manual):
        candidates = [chunk for chunk in by_manual[manual_key] if chunk.chunk_id not in used]
        representatives: list[Chunk] = []
        seen_topics: set[str] = set()
        for chunk in candidates:
            if chunk.topic_id not in seen_topics:
                representatives.append(chunk)
                seen_topics.add(chunk.topic_id)
        if len(representatives) < 2:
            continue
        left = representatives[0]
        right = next(
            (item for item in representatives[1:] if _content_tag(item) != _content_tag(left)),
            representatives[1],
        )
        pair = [left, right]
        pairs.append(pair)
        used.update(chunk.chunk_id for chunk in pair)
        if len(pairs) == count:
            return pairs
    raise RuntimeError(f"could only select {len(pairs)}/{count} multi-topic pairs")


def _cross_manual_pairs(chunks: list[Chunk], count: int, used: set[str]) -> list[list[Chunk]]:
    by_section: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_section[_section(chunk)].append(chunk)
    pairs: list[list[Chunk]] = []
    used_sections: set[str] = set()
    for section, items in sorted(by_section.items()):
        if section in used_sections or len(section) < 3:
            continue
        available = [chunk for chunk in items if chunk.chunk_id not in used]
        left = next(iter(available), None)
        if left is None:
            continue
        right = next(
            (chunk for chunk in available[1:] if chunk.manual_key != left.manual_key),
            None,
        )
        if right is None:
            continue
        pair = [left, right]
        pairs.append(pair)
        used_sections.add(section)
        used.update(chunk.chunk_id for chunk in pair)
        if len(pairs) == count:
            return pairs
    raise RuntimeError(f"could only select {len(pairs)}/{count} cross-manual pairs")


def build_cases(chunks: list[Chunk]) -> list[EvaluationCase]:
    used: set[str] = set()
    records: list[EvaluationCase] = []
    singles = _take_diverse_singles(chunks, QUESTION_COUNTS["single_chunk"], used)
    for index, chunk in enumerate(singles, start=1):
        records.append(
            _case(
                f"single-{index:03d}",
                f"在{_manual(chunk)}中，{_section(chunk)}有哪些说明和注意事项？",
                "single_chunk",
                [chunk],
                "single_evidence",
            )
        )

    same_topic = _same_topic_pairs(chunks, QUESTION_COUNTS["multi_chunk_same_topic"], used)
    for index, (left, right) in enumerate(same_topic, start=1):
        records.append(
            _case(
                f"same-topic-{index:03d}",
                f"在{_manual(left)}中，请综合说明{_section(left)}和{_section(right)}两方面内容。",
                "multi_chunk_same_topic",
                [left, right],
                "multi_evidence",
            )
        )

    multi_topic = _multi_topic_pairs(chunks, QUESTION_COUNTS["multi_topic"], used)
    for index, (left, right) in enumerate(multi_topic, start=1):
        records.append(
            _case(
                f"multi-topic-{index:03d}",
                f"结合{_manual(left)}手册，{_section(left)}和{_section(right)}分别需要注意什么？",
                "multi_topic",
                [left, right],
                "cross_topic",
                "multi_evidence",
            )
        )

    cross_manual = _cross_manual_pairs(chunks, QUESTION_COUNTS["cross_manual"], used)
    for index, (left, right) in enumerate(cross_manual, start=1):
        records.append(
            _case(
                f"cross-manual-{index:03d}",
                f"对比{_manual(left)}与{_manual(right)}手册中的{_section(left)}说明，两者分别怎么规定？",
                "cross_manual",
                [left, right],
                "comparison",
                "multi_manual",
            )
        )

    manuals = list(dict.fromkeys(chunk.manual_key for chunk in chunks if chunk.manual_key))
    for index, question in enumerate(UNANSWERABLE_QUESTIONS, start=1):
        records.append(
            EvaluationCase(
                id=f"unanswerable-{index:03d}",
                user_input=question,
                question_type="unanswerable",
                answerable=False,
                reference="知识库中未找到足够依据。",
                retrieval_filters=RetrievalFilters(
                    manual_keys=[manuals[(index - 1) % len(manuals)]]
                ),
                tags=["unanswerable"],
                label_status="reviewed_unanswerable",
            )
        )

    expected = sum(QUESTION_COUNTS.values())
    if len(records) != expected:
        raise RuntimeError(f"expected {expected} cases, got {len(records)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the single 50-case RAG evaluation set")
    parser.add_argument("--chunks-root", default="data/normalized")
    parser.add_argument("--output", default="data/eval/rag_eval_v2.jsonl")
    args = parser.parse_args()
    chunks_root = Path(args.chunks_root)
    if not chunks_root.is_absolute():
        chunks_root = PROJECT_ROOT / chunks_root
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    cases = build_cases(_load_chunks(chunks_root))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(case.model_dump_json() + "\n" for case in cases),
        encoding="utf-8",
    )
    print(f"wrote {len(cases)} cases to {output}")


if __name__ == "__main__":
    main()
