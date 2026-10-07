"""Authorize one Houston 2025 benchmark after all code and inputs are frozen."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqriskformer.utils import read_json, sha256_file, utc_now, write_json

PROTOCOL = ROOT / "journal_protocol/third_benchmark_final_holdout_protocol.json"
DEVELOPMENT = ROOT / "journal_protocol/third_benchmark_development_freeze.json"
SEAL = ROOT / "journal_protocol/third_benchmark_holdout_seal_status.json"
OUTPUT = ROOT / "journal_protocol/third_benchmark_final_holdout_execution_freeze.json"


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def verify_mapping(values: dict[str, str]) -> None:
    for name, expected in values.items():
        path = ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Pre-freeze artifact mismatch: {path}")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark execution freeze: {OUTPUT}")
    final = read_json(PROTOCOL)
    development = read_json(DEVELOPMENT)
    seal = read_json(SEAL)
    if final.get("status") != "THIRD_BENCHMARK_FINAL_HOLDOUT_PROTOCOL_LOCKED_CONTENT_NOT_PARSED":
        raise RuntimeError("Final benchmark protocol is not locked")
    if (
        development.get("status")
        != "THIRD_BENCHMARK_DEVELOPMENT_FROZEN_HOUSTON_2025_NOT_DOWNLOADED"
    ):
        raise RuntimeError("Benchmark development freeze is invalid")
    if sha256_file(DEVELOPMENT) != final["candidate_selection_freeze_sha256"]:
        raise RuntimeError("Development freeze differs from the final protocol")
    verify_mapping(development["source_sha256"])
    verify_mapping(development["run_artifact_sha256"])
    verify_mapping(development["development_analysis_sha256"])
    if seal.get("status") != "THIRD_BENCHMARK_DOWNLOADED_AND_SEALED_NOT_PARSED":
        raise RuntimeError("Houston benchmark seal is not intact")
    if sha256_file(PROTOCOL) != seal["final_protocol_sha256"]:
        raise RuntimeError("Final protocol differs from the seal record")
    if any(
        seal.get(key) is not False
        for key in (
            "gzip_or_json_content_opened",
            "concentration_values_inspected",
            "test_metrics_computed",
        )
    ):
        raise RuntimeError("Houston seal indicates prior holdout access")
    manifest = ROOT / final["data"]["sealed_manifest"]
    if sha256_file(manifest) != final["data"]["sealed_manifest_sha256"]:
        raise RuntimeError("Houston sealed manifest hash mismatch")
    prepared = ROOT / final["data"]["prepared_holdout"]
    report = ROOT / final["data"]["preparation_report"]
    access_log = ROOT / final["output"] / "test_access_log.json"
    if any(path.exists() for path in (prepared, report, access_log)):
        raise RuntimeError("Houston holdout access artifacts already exist")

    source_paths = [
        PROTOCOL,
        DEVELOPMENT,
        SEAL,
        ROOT / "journal_protocol/third_benchmark_amendment.json",
        ROOT / "journal_protocol/third_unseen_holdout_protocol.json",
        ROOT / "journal_protocol/third_development_preprocessing_protocol.json",
        ROOT / "journal_protocol/third_comparison_protocol.json",
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
        ROOT / "scripts/analyze_third_development.py",
        ROOT / "scripts/run_journal_holdout.py",
        ROOT / "scripts/run_third_benchmark_holdout.py",
        ROOT / "scripts/analyze_third_benchmark_holdout.py",
        ROOT / "scripts/seal_third_benchmark_holdout.py",
        Path(__file__).resolve(),
        ROOT / "pyproject.toml",
    ]
    inputs = {
        **development["run_artifact_sha256"],
        **development["development_analysis_sha256"],
        rel(manifest): sha256_file(manifest),
        final["data"]["development_prepared"]: sha256_file(
            ROOT / final["data"]["development_prepared"]
        ),
    }
    record = {
        "status": "FINAL_HOLDOUT_EXECUTION_FROZEN_READY_TO_OPEN",
        "created_utc": utc_now(),
        "benchmark": "Houston-Harris County 2025 external confirmatory benchmark",
        "title": final["title"],
        "protocol": rel(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "statistical_reference": final["reference_model"],
        "reference_is_proposed_model": False,
        "models": [
            final["reference_model"],
            *final["learned_comparators"],
            *final["deterministic_comparators"],
        ],
        "seeds": final["seeds"],
        "source_sha256": {rel(path): sha256_file(path) for path in source_paths},
        "input_sha256": inputs,
        "calibration": {
            "source": (
                "2024 development-validation condition/horizon/pollutant calibration.npz "
                "for each learned model/seed and deterministic baseline"
            ),
            "method": final["calibration"]["method"],
            "test_time_fitting": False,
        },
        "statistics": final["statistics"],
        "reporting_code_frozen": True,
        "holdout_content_parsed_at_freeze": False,
        "holdout_metrics_computed_at_freeze": False,
        "holdout_evaluation_authorized": True,
        "authorization_scope": (
            "One frozen preparation, inference, evaluation, and reporting campaign over all "
            "listed models. Report all results; no Houston-test-driven selection or revision."
        ),
    }
    write_json(OUTPUT, record)
    print(f"Houston benchmark execution frozen: {OUTPUT.resolve()}")
    print(f"SHA-256: {sha256_file(OUTPUT)}")
    print("The next benchmark command may open the sealed 2025 content once.")


if __name__ == "__main__":
    main()
