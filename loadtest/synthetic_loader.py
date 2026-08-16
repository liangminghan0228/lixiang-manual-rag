from __future__ import annotations

import argparse
import random
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models


def main() -> None:
    parser = argparse.ArgumentParser(description="Load isolated synthetic Qdrant points")
    parser.add_argument("--url", default="http://localhost:6333")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--points", type=int, choices=[10_000, 100_000, 1_000_000], required=True)
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if not args.collection.startswith("synthetic_"):
        raise SystemExit("collection must start with 'synthetic_' to protect business data")

    client = QdrantClient(url=args.url, timeout=60)
    if not client.collection_exists(args.collection):
        client.create_collection(
            collection_name=args.collection,
            vectors_config=models.VectorParams(
                size=args.dimension,
                distance=models.Distance.COSINE,
            ),
        )
    rng = random.Random(20250816)
    for start in range(0, args.points, args.batch_size):
        end = min(start + args.batch_size, args.points)
        batch = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, f"{args.collection}:{index}")),
                vector=[rng.uniform(-1, 1) for _ in range(args.dimension)],
                payload={"synthetic": True, "ordinal": index, "topic_id": f"topic-{index % 1000}"},
            )
            for index in range(start, end)
        ]
        client.upsert(collection_name=args.collection, points=batch, wait=True)
        print(f"loaded {end}/{args.points}", flush=True)


if __name__ == "__main__":
    main()
