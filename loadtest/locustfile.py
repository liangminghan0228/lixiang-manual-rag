from __future__ import annotations

import json
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(os.getenv("RAG_LOADTEST_DATASET", PROJECT_ROOT / "data/eval/rag_eval_v2.jsonl"))
QUESTIONS = [
    json.loads(line)["user_input"]
    for line in DATASET.read_text(encoding="utf-8").splitlines()
    if line.strip()
]


class RagUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def rag_request(self) -> None:
        chat_ratio = float(os.getenv("RAG_LOADTEST_CHAT_RATIO", "0"))
        if random.random() < chat_ratio:  # noqa: S311
            self._chat()
        else:
            self._retrieve()

    def _retrieve(self) -> None:
        with self.client.post(
            "/v1/retrieve",
            json={"question": random.choice(QUESTIONS), "top_k": 10},  # noqa: S311
            name="POST /v1/retrieve",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")

    def _chat(self) -> None:
        with self.client.post(
            "/v1/chat",
            json={"question": random.choice(QUESTIONS)},  # noqa: S311
            name="POST /v1/chat",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
