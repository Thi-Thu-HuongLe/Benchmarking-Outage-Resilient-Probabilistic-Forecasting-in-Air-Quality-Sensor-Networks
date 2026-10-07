"""Freeze Houston benchmark artifacts and reporting code before metric analysis."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqriskformer.utils import read_json, sha256_file, utc_now, write_json

PROTOCOL = ROOT / "journal_protocol/third_benchmark_final_holdout_protocol.json"
RESUME = ROOT / "journal_protocol/third_benchmark_technical_resume_002.json"
AMENDMENT = ROOT / "journal_protocol/third_benchmark_technical_amendment_003.json"
OUTPUT = ROOT / "journal_protocol/third_benchmark_analysis_execution_freeze.json"
RUN_ROOT = ROOT / "experiment_protocol/results_third/holdout_2025"
ANALYZER = ROOT / "scripts/analyze_third_benchmark_holdout.py"


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def main() -> None:
    if AMENDMENT.exists() or OUTPUT.exists():
        raise FileExistsError("Refusing to overwrite analysis amendment or freeze")
    final = read_json(PROTOCOL)
    resume = read_json(RESUME)
    if resume.get("status") != "TECHNICAL_RESUME_FROZEN_AFTER_PREPARATION_BEFORE_METRICS":
        raise RuntimeError("Technical resume 002 is invalid")
    if sha256_file(PROTOCOL) != resume["protocol_sha256"]:
        raise RuntimeError("Final protocol changed after technical resume 002")

    analyzer_name = rel(ANALYZER)
    expected_analyzer = resume["source_sha256"][analyzer_name]
    actual_analyzer = sha256_file(ANALYZER)
    if actual_analyzer == expected_analyzer:
        raise RuntimeError("Analyzer does not contain the documented gate repair")
    for name, expected in resume["source_sha256"].items():
        if name == analyzer_name:
            continue
        path = ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Unapproved source change after resume 002: {path}")

    status_path = RUN_ROOT / "status.json"
    status = read_json(status_path)
    expected_runs = len(final["seeds"]) * (1 + len(final["learned_comparators"])) + len(
        final["deterministic_comparators"]
    )
    if (
        status.get("state") != "complete"
        or int(status.get("completed_fits", -1)) != expected_runs
        or status.get("test_labels_used_for_fitting") is not False
    ):
        raise RuntimeError("Houston benchmark is not complete and leakage-free")
    if (RUN_ROOT / "analysis").exists():
        raise RuntimeError("Analysis output already exists before the analysis freeze")

    input_hashes: dict[str, str] = {
        rel(status_path): sha256_file(status_path),
        rel(RUN_ROOT / "test_access_log.json"): sha256_file(RUN_ROOT / "test_access_log.json"),
    }
    markers = sorted((RUN_ROOT / "runs").rglob("complete.json"))
    if len(markers) != expected_runs:
        raise RuntimeError(f"Expected {expected_runs} complete markers, found {len(markers)}")
    for marker_path in markers:
        marker = read_json(marker_path)
        if (
            marker.get("state") != "complete"
            or marker.get("test_labels_used_for_fitting") is not False
        ):
            raise RuntimeError(f"Invalid benchmark marker: {marker_path}")
        input_hashes[rel(marker_path)] = sha256_file(marker_path)
        for name, expected in marker["artifact_hashes"].items():
            path = marker_path.parent / name
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"Benchmark artifact hash mismatch: {path}")
            input_hashes[rel(path)] = expected

    data_paths = [
        ROOT / final["data"]["prepared_holdout"],
        ROOT / final["data"]["preparation_report"],
        ROOT / final["data"]["availability_table"],
        ROOT / final["data"]["development_prepared"],
    ]
    input_hashes.update({rel(path): sha256_file(path) for path in data_paths})

    amendment = {
        "status": "TECHNICAL_AMENDMENT_003_LOCKED_AFTER_RUN_BEFORE_METRIC_INSPECTION",
        "created_utc": utc_now(),
        "previous_resume_freeze": rel(RESUME),
        "previous_resume_freeze_sha256": sha256_file(RESUME),
        "timing": (
            "After all 52 evaluations completed, but before opening test_metrics.json, "
            "prediction archives, or any aggregated result."
        ),
        "repair": (
            "Point the analyzer to a dedicated post-run analysis freeze, verify its frozen "
            "source/input mappings, and provide the hash-frozen development artifact solely "
            "for training event-rate climatology required by the holdout loader."
        ),
        "analyzer_sha256_before": expected_analyzer,
        "analyzer_sha256_after": actual_analyzer,
        "analysis_algorithm_unchanged": [
            "statistical reference and all comparators",
            "primary endpoints and 6-hour/24-hour conditions",
            "calibration application",
            "moving-block bootstrap resamples and block length",
            "Diebold-Mariano tests and Holm multiplicity correction",
            "metric summaries and natural-case count extraction",
        ],
        "results_inspected_before_amendment": False,
    }
    write_json(AMENDMENT, amendment)

    source_paths = [
        PROTOCOL,
        RESUME,
        AMENDMENT,
        ROOT / "journal_protocol/third_benchmark_amendment.json",
        ROOT / "src/aqriskformer/data.py",
        ROOT / "src/aqriskformer/epa_aqs_holdout.py",
        ROOT / "src/aqriskformer/outage_data.py",
        ROOT / "src/aqriskformer/outage_evaluation.py",
        ROOT / "src/aqriskformer/development_analysis.py",
        ROOT / "src/aqriskformer/statistics.py",
        ROOT / "src/aqriskformer/utils.py",
        ANALYZER,
        Path(__file__).resolve(),
        ROOT / "pyproject.toml",
    ]
    freeze = {
        "status": "FINAL_ANALYSIS_EXECUTION_FROZEN_AFTER_BENCHMARK",
        "created_utc": utc_now(),
        "technical_amendment": rel(AMENDMENT),
        "technical_amendment_sha256": sha256_file(AMENDMENT),
        "protocol": rel(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "completed_evaluations": expected_runs,
        "source_sha256": {rel(path): sha256_file(path) for path in source_paths},
        "input_sha256": input_hashes,
        "test_labels_used_for_fitting_or_selection": False,
        "results_inspected_at_freeze": False,
        "analysis_authorized": True,
        "authorization_scope": (
            "Run the frozen benchmark analyzer once and report all comparisons regardless "
            "of direction; no model selection, retraining, recalibration, or protocol change."
        ),
    }
    write_json(OUTPUT, freeze)
    print(f"Analysis amendment: {AMENDMENT.resolve()}")
    print(f"Analysis execution freeze: {OUTPUT.resolve()}")
    print(f"SHA-256: {sha256_file(OUTPUT)}")


if __name__ == "__main__":
    main()
