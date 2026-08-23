from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from app.settings import PROJECT_ROOT
from scripts.run_experiments import (
    MAX_CAPTURE_CHARS,
    ExperimentSuite,
    build_evaluate_command,
    load_experiment_id,
    run_suite,
)


def _write_valid_report(path: Path, *, errors: int = 0) -> None:
    path.write_text(
        json.dumps(
            {
                "experiment_id": "test-v1",
                "components": {},
                "retrieval": {},
                "generation": {},
                "errors": errors,
                "details": [],
            }
        ),
        encoding="utf-8",
    )


def test_suite_rejects_duplicate_configs() -> None:
    try:
        ExperimentSuite(id="suite-v1", configs=["a.yaml", "a.yaml"])
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate configs should be rejected")


def test_load_experiment_id(tmp_path: Path) -> None:
    config = tmp_path / "dense.yaml"
    config.write_text("experiment:\n  id: dense-v1\n", encoding="utf-8")

    assert load_experiment_id(config) == "dense-v1"


def test_build_command_keeps_current_evaluation_contract(tmp_path: Path) -> None:
    command = build_evaluate_command(
        config_path=tmp_path / "dense.yaml",
        dataset_path=tmp_path / "rag_eval_v2.jsonl",
        output_path=tmp_path / "dense-v1.json",
        retrieval_only=True,
        limit=5,
    )

    assert command[1:3] == ["-m", "scripts.evaluate"]
    assert "--retrieval-only" in command
    assert command[-2:] == ["--limit", "5"]


def test_run_suite_writes_isolated_outputs_and_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_a = tmp_path / "a.yaml"
    config_b = tmp_path / "b.yaml"
    config_a.write_text("experiment:\n  id: dense-v1\n", encoding="utf-8")
    config_b.write_text("experiment:\n  id: hybrid-v1\n", encoding="utf-8")
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "id": "baselines-v1",
                "configs": [str(config_a), str(config_b)],
                "dataset": str(tmp_path / "dataset.jsonl"),
                "output_root": str(tmp_path / "reports"),
                "retrieval_only": True,
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_runner(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        _write_valid_report(output)
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    manifest = run_suite(suite_path, command_runner=fake_runner)

    assert manifest["status"] == "completed"
    assert [item["experiment_id"] for item in manifest["runs"]] == [
        "dense-v1",
        "hybrid-v1",
    ]
    assert len(calls) == 2
    assert manifest["runs"][0]["output"] != manifest["runs"][1]["output"]
    manifest_path = Path(manifest["runs"][0]["output"]).parent / "suite-manifest.json"
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "completed"


def test_dry_run_does_not_execute_commands(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"extends: {(PROJECT_ROOT / 'configs/local-smoke.yaml').as_posix()}\n"
        "experiment:\n  id: dense-v1\n",
        encoding="utf-8",
    )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        yaml.safe_dump(
            {
                "id": "dry-run-v1",
                "configs": [str(config)],
                "output_root": str(tmp_path / "reports"),
            }
        ),
        encoding="utf-8",
    )

    manifest = run_suite(suite, dry_run=True)

    assert manifest["status"] == "planned"
    assert manifest["runs"][0]["status"] == "planned"


def test_report_errors_mark_run_and_suite_failed(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("experiment:\n  id: dense-v1\n", encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        yaml.safe_dump(
            {
                "id": "evaluation-errors-v1",
                "configs": [str(config)],
                "output_root": str(tmp_path / "reports"),
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command, **kwargs):
        del kwargs
        output = Path(command[command.index("--output") + 1])
        _write_valid_report(output, errors=1)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    manifest = run_suite(suite, command_runner=fake_runner)

    assert manifest["status"] == "failed"
    assert manifest["runs"][0]["status"] == "failed"
    assert manifest["runs"][0]["evaluation_errors"] == 1


def test_missing_evaluation_report_marks_run_failed(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("experiment:\n  id: dense-v1\n", encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        yaml.safe_dump(
            {
                "id": "missing-report-v1",
                "configs": [str(config)],
                "output_root": str(tmp_path / "reports"),
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    manifest = run_suite(suite, command_runner=fake_runner)

    assert manifest["status"] == "failed"
    assert manifest["runs"][0]["status"] == "failed"


def test_fractional_error_count_marks_report_invalid(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("experiment:\n  id: dense-v1\n", encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        yaml.safe_dump(
            {
                "id": "invalid-errors-v1",
                "configs": [str(config)],
                "output_root": str(tmp_path / "reports"),
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command, **kwargs):
        del kwargs
        output = Path(command[command.index("--output") + 1])
        _write_valid_report(output)
        report = json.loads(output.read_text(encoding="utf-8"))
        report["errors"] = 0.5
        output.write_text(json.dumps(report), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    manifest = run_suite(suite, command_runner=fake_runner)

    assert manifest["status"] == "failed"


def test_runner_exception_is_persisted_and_stops_when_configured(tmp_path: Path) -> None:
    config_a = tmp_path / "a.yaml"
    config_b = tmp_path / "b.yaml"
    config_a.write_text("experiment:\n  id: dense-v1\n", encoding="utf-8")
    config_b.write_text("experiment:\n  id: hybrid-v1\n", encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        yaml.safe_dump(
            {
                "id": "failure-v1",
                "configs": [str(config_a), str(config_b)],
                "output_root": str(tmp_path / "reports"),
                "continue_on_error": False,
            }
        ),
        encoding="utf-8",
    )

    def failing_runner(command, **kwargs):
        del command, kwargs
        raise FileNotFoundError("python executable missing")

    manifest = run_suite(suite, command_runner=failing_runner)

    assert manifest["status"] == "failed"
    assert len(manifest["runs"]) == 1
    assert manifest["runs"][0]["status"] == "failed"
    assert manifest["runs"][0]["returncode"] is None
    assert "FileNotFoundError" in manifest["runs"][0]["error"]
    manifest_path = Path(manifest["runs"][0]["output"]).parent / "suite-manifest.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_runner_bounds_captured_output(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("experiment:\n  id: dense-v1\n", encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        yaml.safe_dump(
            {
                "id": "bounded-output-v1",
                "configs": [str(config)],
                "output_root": str(tmp_path / "reports"),
            }
        ),
        encoding="utf-8",
    )

    def noisy_runner(command, **kwargs):
        del kwargs
        output = "x" * (MAX_CAPTURE_CHARS + 10)
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr=output)

    manifest = run_suite(suite, command_runner=noisy_runner)

    assert "truncated 10 characters" in manifest["runs"][0]["stdout"]
    assert "truncated 10 characters" in manifest["runs"][0]["stderr"]
