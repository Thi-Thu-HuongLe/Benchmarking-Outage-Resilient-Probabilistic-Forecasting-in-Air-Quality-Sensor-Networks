"""Freeze the second schema-only Houston continuation before any inference."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqriskformer.utils import read_json, sha256_file, utc_now, write_json

PREVIOUS = ROOT / "journal_protocol/third_benchmark_technical_resume_001.json"
PROTOCOL = ROOT / "journal_protocol/third_benchmark_final_holdout_protocol.json"
AMENDMENT = ROOT / "journal_protocol/third_benchmark_technical_amendment_002.json"
OUTPUT = ROOT / "journal_protocol/third_benchmark_technical_resume_002.json"
RUN_ROOT = ROOT / "experiment_protocol/results_third/holdout_2025/runs"
STATUS = ROOT / "experiment_protocol/results_third/holdout_2025/status.json"
ALLOWED_CHANGES = {
    "src/aqriskformer/epa_aqs_holdout.py",
    "scripts/run_third_benchmark_holdout.py",
}


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def main() -> None:
    if AMENDMENT.exists() or OUTPUT.exists():
        raise FileExistsError("Refusing to overwrite technical amendment/resume 002")
    previous = read_json(PREVIOUS)
    if previous.get("status") != "TECHNICAL_RESUME_FROZEN_AFTER_PREPARATION_BEFORE_METRICS":
        raise RuntimeError("Technical resume 001 is invalid")
    if sha256_file(PROTOCOL) != previous["protocol_sha256"]:
        raise RuntimeError("Final protocol changed after technical resume 001")
    for name, expected in previous["input_sha256"].items():
        path = ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen benchmark input changed: {path}")

    changed: dict[str, dict[str, str]] = {}
    for name, expected in previous["source_sha256"].items():
        path = ROOT / name
        actual = sha256_file(path) if path.is_file() else "MISSING"
        if actual != expected:
            if name not in ALLOWED_CHANGES:
                raise RuntimeError(f"Unapproved post-resume source change: {path}")
            changed[name] = {"before_sha256": expected, "after_sha256": actual}
    if set(changed) != ALLOWED_CHANGES:
        raise RuntimeError(f"Unexpected technical-change set: {sorted(changed)}")
    if RUN_ROOT.exists() and any(RUN_ROOT.rglob("test_metrics.json")):
        raise RuntimeError("Test metrics already exist; technical resume 002 is invalid")
    if STATUS.exists():
        raise RuntimeError("Evaluation status exists; inference may already have started")

    amendment = {
        "status": "TECHNICAL_AMENDMENT_002_LOCKED_BEFORE_INFERENCE_AND_METRICS",
        "created_utc": utc_now(),
        "previous_resume_freeze": rel(PREVIOUS),
        "previous_resume_freeze_sha256": sha256_file(PREVIOUS),
        "failure_stage": "holdout loading after preparation; before model loading and metrics",
        "failure": {
            "exception": "KeyError",
            "message": "'train'",
            "cause": (
                "training_event_rates is a cached property computed from a train split, while "
                "the correctly sealed holdout artifact contains only a test split"
            ),
        },
        "repair": (
            "Load training_event_rates only from the hash-frozen 2021-2024 development "
            "artifact, require identical station and pollutant identities, and pass the "
            "result to the test-only OutageData object."
        ),
        "changed_source_sha256": changed,
        "scientific_invariants_unchanged": [
            "models, checkpoints, and predictions",
            "validation-fitted calibration arrays",
            "outage conditions, forecast horizons, endpoints, and statistical tests",
            "Houston values, masks, station identities, and prepared tensors",
        ],
        "holdout_state": {
            "content_parsed": True,
            "model_inference_started": False,
            "test_metrics_computed": False,
            "results_inspected": False,
        },
    }
    write_json(AMENDMENT, amendment)

    source_paths = [
        PROTOCOL,
        PREVIOUS,
        AMENDMENT,
        ROOT / "journal_protocol/third_benchmark_final_holdout_execution_freeze.json",
        ROOT / "src/aqriskformer/data.py",
        ROOT / "src/aqriskformer/epa_aqs_holdout.py",
        ROOT / "src/aqriskformer/outage_data.py",
        ROOT / "src/aqriskformer/outage_evaluation.py",
        ROOT / "src/aqriskformer/outage_models.py",
        ROOT / "src/aqriskformer/development_analysis.py",
        ROOT / "src/aqriskformer/statistics.py",
        ROOT / "src/aqriskformer/utils.py",
        ROOT / "src/aqriskformer/models/baselines.py",
        ROOT / "src/aqriskformer/models/graph_baselines.py",
        ROOT / "src/aqriskformer/models/learned_imputers.py",
        ROOT / "scripts/run_journal_development.py",
        ROOT / "scripts/run_journal_holdout.py",
        ROOT / "scripts/run_third_benchmark_holdout.py",
        ROOT / "scripts/analyze_third_benchmark_holdout.py",
        Path(__file__).resolve(),
        ROOT / "pyproject.toml",
    ]
    resume = {
        "status": "TECHNICAL_RESUME_FROZEN_AFTER_PREPARATION_BEFORE_METRICS",
        "created_utc": utc_now(),
        "technical_amendment": rel(AMENDMENT),
        "technical_amendment_sha256": sha256_file(AMENDMENT),
        "previous_resume_freeze": rel(PREVIOUS),
        "previous_resume_freeze_sha256": sha256_file(PREVIOUS),
        "protocol": rel(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "source_sha256": {rel(path): sha256_file(path) for path in source_paths},
        "input_sha256": {
            **previous["input_sha256"],
            rel(PREVIOUS): sha256_file(PREVIOUS),
        },
        "holdout_content_parsed_at_freeze": True,
        "holdout_metrics_computed_at_freeze": False,
        "holdout_results_inspected_at_freeze": False,
        "holdout_evaluation_authorized": True,
        "authorization_scope": (
            "Resume the unchanged 52-evaluation benchmark using only development-frozen "
            "event climatology; report all results without holdout-driven revision."
        ),
    }
    write_json(OUTPUT, resume)
    print(f"Technical amendment 002: {AMENDMENT.resolve()}")
    print(f"Technical resume freeze 002: {OUTPUT.resolve()}")
    print(f"SHA-256: {sha256_file(OUTPUT)}")


if __name__ == "__main__":
    main()
