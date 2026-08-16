from __future__ import annotations

import argparse
import json
from collections import defaultdict

from app.models import Chunk
from app.settings import PROJECT_ROOT

UNANSWERABLE_QUESTIONS = [
    "这辆车的终身免费保养政策是什么？",
    "2028 款车型新增了哪些配置？",
    "车辆保险每年具体多少钱？",
    "附近维修门店今天几点关门？",
    "如何把车辆改装成七座？",
    "车辆在月球环境下能行驶多久？",
    "下一次软件更新的准确发布日期是什么？",
    "二手车三年后的成交价格是多少？",
    "怎样关闭所有法律要求的安全功能？",
    "车主上个月一共行驶了多少公里？",
    "当前车辆剩余电量是多少？",
    "我的订单预计哪天交付？",
    "发生事故后保险公司会赔多少钱？",
    "未来五年的充电价格是多少？",
    "这辆车与尚未发布车型的销量谁更高？",
]


def evidence(group_id: str, chunk: Chunk) -> dict[str, object]:
    return {
        "id": group_id,
        "acceptable_chunk_ids": [chunk.chunk_id],
        "acceptable_topic_ids": [],
    }


def case(
    case_id: str,
    question: str,
    question_type: str,
    chunks: list[Chunk],
) -> dict[str, object]:
    groups = [evidence(f"g{index}", chunk) for index, chunk in enumerate(chunks, start=1)]
    points = [
        {
            "id": f"p{index}",
            "text": "覆盖对应章节要点",
            "keywords": [chunk.section_path[-1] if chunk.section_path else chunk.title],
            "required_evidence_groups": [f"g{index}"],
        }
        for index, chunk in enumerate(chunks, start=1)
    ]
    return {
        "id": case_id,
        "question": question,
        "question_type": question_type,
        "answerable": True,
        "required_evidence_groups": groups,
        "answer_points": points,
        "label_status": "structure_verified_content_review_required",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic 100-case eval draft")
    parser.add_argument("--chunks", default="data/normalized/20250916141802/chunks.jsonl")
    parser.add_argument("--output", default="data/eval/full_v1.jsonl")
    args = parser.parse_args()
    chunks_path = PROJECT_ROOT / args.chunks
    chunks = [
        Chunk.model_validate_json(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    usable = [chunk for chunk in chunks if chunk.section_path and len(chunk.text) >= 80]
    by_topic: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in usable:
        by_topic[chunk.topic_id].append(chunk)

    records: list[dict[str, object]] = []
    for index, chunk in enumerate(usable[:55], start=1):
        section = chunk.section_path[-1]
        records.append(
            case(
                f"single-{index:03d}",
                f"手册中关于“{section}”是怎么说明的？",
                "single_chunk",
                [chunk],
            )
        )

    same_topic_pairs = [items[:2] for items in by_topic.values() if len(items) >= 2][:20]
    for index, pair in enumerate(same_topic_pairs, start=1):
        left, right = pair
        records.append(
            case(
                f"same-topic-{index:03d}",
                f"请综合说明“{left.section_path[-1]}”和“{right.section_path[-1]}”两方面要求。",
                "multi_chunk_same_topic",
                pair,
            )
        )

    representatives = [items[0] for items in by_topic.values() if items]
    for index in range(10):
        left = representatives[index * 2]
        right = representatives[index * 2 + 1]
        records.append(
            case(
                f"multi-topic-{index + 1:03d}",
                f"跨章节比较“{left.section_path[-1]}”与“{right.section_path[-1]}”分别需要注意什么？",
                "multi_topic",
                [left, right],
            )
        )

    for index, question in enumerate(UNANSWERABLE_QUESTIONS, start=1):
        records.append(
            {
                "id": f"unanswerable-{index:03d}",
                "question": question,
                "question_type": "unanswerable",
                "answerable": False,
                "required_evidence_groups": [],
                "answer_points": [],
                "label_status": "reviewed_unanswerable",
            }
        )

    if len(records) != 100:
        raise RuntimeError(f"expected 100 cases, got {len(records)}")
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"wrote {len(records)} cases to {output}")


if __name__ == "__main__":
    main()
