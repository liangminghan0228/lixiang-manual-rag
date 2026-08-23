from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from app.factories import (
    CHUNKERS,
    EMBEDDERS,
    EVIDENCE_SELECTORS,
    GENERATORS,
    QUERY_PROCESSORS,
    RAG_STRATEGIES,
    RERANKERS,
    RETRIEVERS,
    VECTOR_STORES,
    require_factory,
)
from app.settings import PROJECT_ROOT, load_config_data, load_settings

SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MAX_CAPTURE_CHARS = 32_768


class ExperimentSuite(BaseModel):
    id: str
    configs: list[str] = Field(min_length=1)
    dataset: str = "data/eval/rag_eval_v2.jsonl"
    output_root: str = "reports/experiments"
    retrieval_only: bool = True
    limit: int = Field(default=0, ge=0)
    continue_on_error: bool = True

    @model_validator(mode="after")
    def validate_identity(self) -> ExperimentSuite:
        if not SAFE_ID_PATTERN.fullmatch(self.id):
            raise ValueError("suite id may only contain letters, numbers, '.', '_' and '-'")
        if len(set(self.configs)) != len(self.configs):
            raise ValueError("suite configs must be unique")
        return self


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_suite(path: str | Path) -> ExperimentSuite:
    selected = resolve_project_path(path)
    raw = yaml.safe_load(selected.read_text(encoding="utf-8")) or {}
    return ExperimentSuite.model_validate(raw)


def load_experiment_id(config_path: str | Path) -> str:
    selected = resolve_project_path(config_path)
    raw = load_config_data(selected)
    experiment_id = str((raw.get("experiment") or {}).get("id") or "").strip()
    if not experiment_id:
        raise ValueError(f"missing experiment.id in {selected}")
    if not SAFE_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError(f"unsafe experiment.id in {selected}: {experiment_id}")
    return experiment_id


def validate_experiment_config(config_path: Path) -> None:
    """Validate config inheritance and registry names without constructing components."""
    settings, _ = load_settings(config_path, apply_runtime_overrides=False)
    require_factory(CHUNKERS, settings.data.chunker, "chunker")
    require_factory(EMBEDDERS, settings.embedding.provider, "embedder")
    require_factory(VECTOR_STORES, settings.vector_store.provider, "vector store")
    require_factory(RETRIEVERS, settings.retrieval.provider, "retriever")
    require_factory(
        QUERY_PROCESSORS,
        settings.retrieval.query_processor.provider,
        "query processor",
    )
    require_factory(RERANKERS, settings.retrieval.reranker.provider, "reranker")
    require_factory(EVIDENCE_SELECTORS, settings.retrieval.evidence_selector, "evidence selector")
    require_factory(GENERATORS, settings.generation.provider, "generator")
    require_factory(RAG_STRATEGIES, settings.rag.strategy, "RAG strategy")


def build_evaluate_command(
    *,
    config_path: Path,
    dataset_path: Path,
    output_path: Path,
    retrieval_only: bool,
    limit: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.evaluate",
        "--config",
        str(config_path),
        "--dataset",
        str(dataset_path),
        "--output",
        str(output_path),
    ]
    if retrieval_only:
        command.append("--retrieval-only")
    if limit:
        command.extend(["--limit", str(limit)])
    return command


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _bounded_output(value: str | None) -> str:
    text = value or ""
    if len(text) <= MAX_CAPTURE_CHARS:
        return text
    omitted = len(text) - MAX_CAPTURE_CHARS
    return f"{text[:MAX_CAPTURE_CHARS]}\n...[truncated {omitted} characters]"


def _report_error_count(path: Path) -> int | None:
    """Read evaluate's per-case error count when the report was generated."""
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        required = {"experiment_id", "components", "retrieval", "generation", "errors", "details"}
        if not isinstance(report, dict) or not required.issubset(report):
            return None
        errors = report["errors"]
        if isinstance(errors, bool) or not isinstance(errors, int) or errors < 0:
            return None
        return errors
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def run_suite(
    suite_path: str | Path,
    *,
    dry_run: bool = False,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    suite_file = resolve_project_path(suite_path)
    suite = load_suite(suite_file)
    configs = [resolve_project_path(item) for item in suite.configs]
    experiment_ids = [load_experiment_id(path) for path in configs]
    if len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError("experiment.id values must be unique within a suite")

    generated_at = datetime.now(UTC)
    run_id = generated_at.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = resolve_project_path(suite.output_root) / suite.id / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "suite-manifest.json"
    dataset_path = resolve_project_path(suite.dataset)
    manifest: dict[str, Any] = {
        "suite_id": suite.id,
        "run_id": run_id,
        "generated_at": generated_at.isoformat(),
        "suite_file": str(suite_file),
        "dataset": str(dataset_path),
        "retrieval_only": suite.retrieval_only,
        "limit": suite.limit,
        "status": "planned" if dry_run else "running",
        "runs": [],
    }

    for config_path, experiment_id in zip(configs, experiment_ids, strict=True):
        if dry_run:
            validate_experiment_config(config_path)
        output_path = run_dir / f"{experiment_id}.json"
        command = build_evaluate_command(
            config_path=config_path,
            dataset_path=dataset_path,
            output_path=output_path,
            retrieval_only=suite.retrieval_only,
            limit=suite.limit,
        )
        item: dict[str, Any] = {
            "experiment_id": experiment_id,
            "config": str(config_path),
            "output": str(output_path),
            "command": command,
            "status": "planned" if dry_run else "running",
        }
        manifest["runs"].append(item)
        _write_manifest(manifest_path, manifest)
        if dry_run:
            continue

        started = time.perf_counter()
        try:
            completed = command_runner(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001 - persist runner failures in the manifest
            item.update(
                {
                    "status": "failed",
                    "returncode": None,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                    "stdout": "",
                    "stderr": "",
                }
            )
            _write_manifest(manifest_path, manifest)
            if not suite.continue_on_error:
                break
            continue
        report_errors = _report_error_count(output_path)
        report_valid = report_errors is not None
        item.update(
            {
                "status": (
                    "completed"
                    if completed.returncode == 0 and report_valid and report_errors == 0
                    else "failed"
                ),
                "returncode": completed.returncode,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "stdout": _bounded_output(completed.stdout),
                "stderr": _bounded_output(completed.stderr),
            }
        )
        if report_errors is not None:
            item["evaluation_errors"] = report_errors
        _write_manifest(manifest_path, manifest)
        if item["status"] == "failed" and not suite.continue_on_error:
            break

    statuses = [item["status"] for item in manifest["runs"]]
    if dry_run:
        manifest["status"] = "planned"
    elif statuses and all(status == "completed" for status in statuses):
        manifest["status"] = "completed"
    elif "failed" in statuses:
        manifest["status"] = "failed"
    else:
        manifest["status"] = "incomplete"
    _write_manifest(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an isolated matrix of RAG experiments")
    parser.add_argument("--suite", default="configs/experiment-suite.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = run_suite(args.suite, dry_run=args.dry_run)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if manifest["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
