"""Freeze a schema-only repair after Houston preparation stopped before inference."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqriskformer.utils import read_json, sha256_file, utc_now, write_json

ORIGINAL = ROOT / "journal_protocol/third_benchmark_final_holdout_execution_freeze.json"
PROTOCOL = ROOT / "journal_protocol/third_benchmark_final_holdout_protocol.json"
AMENDMENT = ROOT / "journal_protocol/third_benchmark_technical_amendment_001.json"
OUTPUT = ROOT / "journal_protocol/third_benchmark_technical_resume_001.json"
PREPARED = ROOT / "data_external/epa_aqs_houston/prepared_holdout/houston_2025.npz"
REPORT = ROOT / "data_external/epa_aqs_houston/prepared_holdout/preparation_report.json"
AVAILABILITY = ROOT / "data_external/epa_aqs_houston/prepared_holdout/availability.csv"
ACCESS_LOG = ROOT / "experiment_protocol/results_third/holdout_2025/test_access_log.json"
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
        raise FileExistsError("Refusing to overwrite the technical amendment or resume freeze")
    original = read_json(ORIGINAL)
    if original.get("status") != "FINAL_HOLDOUT_EXECUTION_FROZEN_READY_TO_OPEN":
        raise RuntimeError("Original execution freeze is invalid")
    if sha256_file(PROTOCOL) != original["protocol_sha256"]:
        raise RuntimeError("Final benchmark protocol changed after the original freeze")
    for name, expected in original["input_sha256"].items():
        path = ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen benchmark input changed: {path}")

    changed: dict[str, dict[str, str]] = {}
    for name, expected in original["source_sha256"].items():
        path = ROOT / name
        actual = sha256_file(path) if path.is_file() else "MISSING"
        if actual != expected:
            if name not in ALLOWED_CHANGES:
                raise RuntimeError(f"Unapproved post-freeze source change: {path}")
            changed[name] = {"before_sha256": expected, "after_sha256": actual}
    if set(changed) != ALLOWED_CHANGES:
        raise RuntimeError(f"Unexpected technical-change set: {sorted(changed)}")

    report = read_json(REPORT)
    if (
        report.get("status") != "PASS_HOLDOUT_PREPARED_AFTER_FINAL_FREEZE"
        or report.get("holdout_content_accessed") is not True
        or report.get("test_metrics_computed") is not False
    ):
        raise RuntimeError("Prepared-holdout report does not document a pre-metric stop")
    if not all(path.is_file() for path in (PREPARED, AVAILABILITY, ACCESS_LOG)):
        raise RuntimeError("Expected preparation artifacts are incomplete")
    if REPORT.is_file() and sha256_file(PREPARED) != report["prepared_sha256"]:
        raise RuntimeError("Prepared holdout differs from its preparation report")
    if RUN_ROOT.exists() and any(RUN_ROOT.rglob("test_metrics.json")):
        raise RuntimeError("Test metrics already exist; a schema-only resume is not permitted")
    if STATUS.exists():
        raise RuntimeError("Evaluation status exists; failure did not occur before inference")

    amendment = {
        "status": "TECHNICAL_AMENDMENT_001_LOCKED_BEFORE_INFERENCE_AND_METRICS",
        "created_utc": utc_now(),
        "original_execution_freeze": rel(ORIGINAL),
        "original_execution_freeze_sha256": sha256_file(ORIGINAL),
        "failure_stage": "after sealed-source preparation; before model loading and metrics",
        "failure": {
            "exception": "TypeError",
            "message": (
                "OutageData.__init__() missing 1 required positional argument: 'event_rates'"
            ),
            "location": "src/aqriskformer/epa_aqs_holdout.py:load_holdout_outage_data",
        },
        "repair": (
            "Pass the already-stored training_event_rates tensor into the holdout OutageData "
            "constructor, validate its frozen shape, and make the runner verify this explicit "
            "post-preparation resume record."
        ),
        "changed_source_sha256": changed,
        "scientific_invariants_unchanged": [
            "all model architectures and checkpoints",
            "all validation-fitted calibration arrays",
            "all outage conditions and forecast horizons",
            "all endpoints, statistical tests, bootstrap settings, and reporting rules",
            "all holdout values, masks, station identities, and prepared tensors",
        ],
        "holdout_state": {
            "content_parsed": True,
            "prepared_artifact_created": True,
            "model_inference_started": False,
            "test_metrics_computed": False,
            "results_inspected": False,
        },
        "authorization": (
            "Resume the same frozen benchmark from the verified prepared artifact. This is a "
            "technical continuation, not a second holdout experiment."
        ),
    }
    write_json(AMENDMENT, amendment)

    source_paths = [
        PROTOCOL,
        ORIGINAL,
        AMENDMENT,
        ROOT / "journal_protocol/third_benchmark_development_freeze.json",
        ROOT / "journal_protocol/third_benchmark_holdout_seal_status.json",
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
    input_paths = [
        PREPARED,
        REPORT,
        AVAILABILITY,
        ACCESS_LOG,
        ROOT / read_json(PROTOCOL)["data"]["sealed_manifest"],
    ]
    resume = {
        "status": "TECHNICAL_RESUME_FROZEN_AFTER_PREPARATION_BEFORE_METRICS",
        "created_utc": utc_now(),
        "technical_amendment": rel(AMENDMENT),
        "technical_amendment_sha256": sha256_file(AMENDMENT),
        "original_execution_freeze": rel(ORIGINAL),
        "original_execution_freeze_sha256": sha256_file(ORIGINAL),
        "protocol": rel(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "source_sha256": {rel(path): sha256_file(path) for path in source_paths},
        "input_sha256": {
            **original["input_sha256"],
            **{rel(path): sha256_file(path) for path in input_paths},
        },
        "holdout_content_parsed_at_freeze": True,
        "holdout_metrics_computed_at_freeze": False,
        "holdout_results_inspected_at_freeze": False,
        "holdout_evaluation_authorized": True,
        "authorization_scope": (
            "Resume the original 52-evaluation benchmark without any scientific change; "
            "report every frozen result and make no holdout-driven revision."
        ),
    }
    write_json(OUTPUT, resume)
    print(f"Technical amendment: {AMENDMENT.resolve()}")
    print(f"Technical resume freeze: {OUTPUT.resolve()}")
    print(f"SHA-256: {sha256_file(OUTPUT)}")


if __name__ == "__main__":
    main()
