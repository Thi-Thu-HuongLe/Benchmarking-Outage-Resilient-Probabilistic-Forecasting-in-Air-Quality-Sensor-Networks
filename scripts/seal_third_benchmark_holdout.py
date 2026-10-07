"""Seal Houston 2025 and write the final multi-model benchmark protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqriskformer.utils import read_json, sha256_file, utc_now, write_json


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-freeze",
        type=Path,
        default=ROOT / "journal_protocol/third_benchmark_development_freeze.json",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data_external/epa_aqs_houston/raw_holdout_sealed",
    )
    parser.add_argument(
        "--protocol-output",
        type=Path,
        default=ROOT / "journal_protocol/third_benchmark_final_holdout_protocol.json",
    )
    parser.add_argument(
        "--seal-output",
        type=Path,
        default=ROOT / "journal_protocol/third_benchmark_holdout_seal_status.json",
    )
    args = parser.parse_args()
    if args.protocol_output.exists() or args.seal_output.exists():
        raise FileExistsError("Refusing to overwrite the final benchmark protocol or seal")
    development = read_json(args.development_freeze)
    if (
        development.get("status")
        != "THIRD_BENCHMARK_DEVELOPMENT_FROZEN_HOUSTON_2025_NOT_DOWNLOADED"
    ):
        raise RuntimeError("Houston benchmark development is not frozen")
    if development.get("holdout_download_authorized") is not True:
        raise RuntimeError("Houston 2025 download was not authorized")
    amendment_path = ROOT / development["amendment"]
    if sha256_file(amendment_path) != development["amendment_sha256"]:
        raise RuntimeError("Benchmark amendment changed after development freeze")
    amendment = read_json(amendment_path)

    manifest_path = args.raw_dir / "download_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("stage") != "holdout_api" or manifest.get("holdout_data_included") is not True:
        raise ValueError("Manifest is not a sealed holdout download")
    if manifest.get("state_fips") != "48" or manifest.get("county_fips") != "201":
        raise ValueError("Holdout manifest is not Harris County, Texas")
    if manifest.get("protocol_sha256_at_download") != sha256_file(amendment_path):
        raise ValueError("Holdout was not downloaded against the frozen amendment hash")
    entries = list(manifest.get("files", []))
    expected = {(name, 2025) for name in ("PM2.5", "NO2", "O3", "CO", "SO2")}
    actual = {(str(item["pollutant"]), int(item["year"])) for item in entries}
    if len(entries) != len(expected) or actual != expected:
        raise ValueError("Holdout manifest is not the exact 2025 pollutant grid")
    for item in entries:
        path = args.raw_dir / str(item["file"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Sealed source hash mismatch: {path}")

    models = amendment["models"]
    learned = list(models["learned"])
    deterministic = list(models["deterministic"])
    reference = str(amendment["development_selected_statistical_reference"])
    comparators = [name for name in learned if name != reference]
    final = {
        "status": "THIRD_BENCHMARK_FINAL_HOLDOUT_PROTOCOL_LOCKED_CONTENT_NOT_PARSED",
        "created_utc": utc_now(),
        "name": "houston_2025_external_confirmatory_outage_benchmark",
        "title": amendment["title"],
        "study_type": "multi_model_benchmark",
        "benchmark_amendment": rel(amendment_path),
        "benchmark_amendment_sha256": sha256_file(amendment_path),
        "candidate_selection_freeze": rel(args.development_freeze),
        "candidate_selection_freeze_sha256": sha256_file(args.development_freeze),
        "reference_model": reference,
        "reference_role": amendment["reference_role"],
        "candidate": reference,
        "candidate_role": "compatibility alias: statistical reference, not proposed model",
        "learned_comparators": comparators,
        "deterministic_comparators": deterministic,
        "refinement_models": models["refinement_models"],
        "seeds": models["seeds"],
        "checkpoint_roots": {
            **amendment["checkpoint_roots"],
            "candidate": amendment["checkpoint_roots"]["base"],
            "comparators": amendment["checkpoint_roots"]["base"],
        },
        "development_protocol": "journal_protocol/third_comparison_protocol.json",
        "data": {
            "development_prepared": development["prepared_development"],
            "development_prepared_sha256": development["prepared_development_sha256"],
            "sealed_raw_root": rel(args.raw_dir),
            "sealed_manifest": rel(manifest_path),
            "sealed_manifest_sha256": sha256_file(manifest_path),
            "prepared_holdout": ("data_external/epa_aqs_houston/prepared_holdout/houston_2025.npz"),
            "prepared_dataset_name": "epa_aqs_houston_2025_holdout",
            "preparation_report": (
                "data_external/epa_aqs_houston/prepared_holdout/preparation_report.json"
            ),
            "availability_table": (
                "data_external/epa_aqs_houston/prepared_holdout/availability.csv"
            ),
            "holdout_start_utc": "2025-01-01T06:00:00Z",
            "holdout_end_utc": "2026-01-01T05:00:00Z",
            "expected_holdout_hours": 8760,
            "context_hours_from_development": 168,
            "embargo_hours_not_scored": 168,
            "first_scored_target_utc": "2025-01-08T06:00:00Z",
            "calendar_clock": "fixed UTC-06:00 local standard time",
            "local_standard_utc_offset_hours": -6,
        },
        "forecast": amendment["forecast"],
        "conditions": amendment["conditions"],
        "endpoints": {
            **amendment["endpoints"],
            "clean_guardrail": {
                "endpoints": ["mase", "q95_brier"],
                "maximum_relative_degradation_vs_each_local_control": 0.01,
                "local_controls": ["local_tcn", "spatial_residual"],
            },
        },
        "calibration": amendment["calibration"],
        "statistics": amendment["statistics"],
        "reporting_rules": amendment["reporting_rules"],
        "output": "experiment_protocol/results_third/holdout_2025",
        "access_rule": (
            "The Houston 2025 gzip/JSON content may be opened only after the final execution "
            "freeze exists, authorizes evaluation, and every recorded hash verifies."
        ),
        "rerun_rule": (
            "Report all Houston 2025 results regardless of direction. Never select, retrain, "
            "recalibrate, or revise a model, endpoint, or protocol using holdout labels. A "
            "technical interruption may resume only already-frozen computations."
        ),
    }
    write_json(args.protocol_output, final)
    seal = {
        "status": "THIRD_BENCHMARK_DOWNLOADED_AND_SEALED_NOT_PARSED",
        "created_utc": utc_now(),
        "holdout_manifest": rel(manifest_path),
        "holdout_manifest_sha256": sha256_file(manifest_path),
        "final_protocol": rel(args.protocol_output),
        "final_protocol_sha256": sha256_file(args.protocol_output),
        "gzip_or_json_content_opened": False,
        "concentration_values_inspected": False,
        "test_metrics_computed": False,
        "file_sha256": {str(item["file"]): item["sha256"] for item in entries},
    }
    write_json(args.seal_output, seal)
    print(f"Final benchmark protocol: {args.protocol_output.resolve()}")
    print(f"Houston 2025 seal: {args.seal_output.resolve()}")
    print("The 2025 concentration content remains unopened.")


if __name__ == "__main__":
    main()
