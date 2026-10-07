"""Freeze the Houston development benchmark before downloading the 2025 holdout."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from aqriskformer.utils import read_json, sha256_file, utc_now, write_json
from scripts.freeze_journal_candidate import verify_run

PROTOCOL = ROOT / "journal_protocol/third_comparison_protocol.json"
UNSEEN = ROOT / "journal_protocol/third_unseen_holdout_protocol.json"
ANALYSIS = ROOT / "experiment_protocol/results_third/development/analysis_locked"
BASE = ROOT / "experiment_protocol/results_third/development/full_locked"
REFINEMENT = ROOT / "experiment_protocol/results_third/development/full_refinement_locked"
AMENDMENT = ROOT / "journal_protocol/third_benchmark_amendment.json"
OUTPUT = ROOT / "journal_protocol/third_benchmark_development_freeze.json"
HOLDOUT_ROOT = ROOT / "data_external/epa_aqs_houston/raw_holdout_sealed"


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def verify_mapping(base: Path, values: dict[str, str]) -> None:
    for name, expected in values.items():
        path = base / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen hash mismatch: {path}")


def verify_development_root(path: Path, expected: int) -> None:
    launch = read_json(path / "launch_manifest.json")
    status = read_json(path / "status.json")
    if launch.get("holdout_accessed") is not False:
        raise RuntimeError(f"Development launch reports holdout access: {path}")
    if (
        status.get("state") != "complete"
        or status.get("holdout_accessed") is not False
        or int(status.get("completed_fits", -1)) != expected
    ):
        raise RuntimeError(f"Development campaign is incomplete: {path}")


def main() -> None:
    if AMENDMENT.exists() or OUTPUT.exists():
        raise FileExistsError("Refusing to overwrite the benchmark amendment or freeze")
    if HOLDOUT_ROOT.exists():
        raise RuntimeError("Houston 2025 exists before the development benchmark freeze")

    protocol = read_json(PROTOCOL)
    unseen = read_json(UNSEEN)
    learned = [*protocol["learned_models"], *protocol["refinement_models"]]
    deterministic = list(protocol["deterministic_models"])
    seeds = [int(seed) for seed in protocol["seeds"]]
    verify_development_root(BASE, len(protocol["learned_models"]) * len(seeds) + len(deterministic))
    verify_development_root(REFINEMENT, len(protocol["refinement_models"]) * len(seeds))

    analysis_manifest = read_json(ANALYSIS / "analysis_manifest.json")
    if (
        analysis_manifest.get("development_only") is not True
        or analysis_manifest.get("holdout_accessed") is not False
    ):
        raise RuntimeError("Development analysis does not preserve the unopened holdout")
    verify_mapping(ROOT, analysis_manifest["input_sha256"])
    verify_mapping(ANALYSIS, analysis_manifest["output_sha256"])
    reframing = read_json(ANALYSIS / "fixed_control_reframing_assessment.json")

    run_hashes: dict[str, str] = {}
    base_models = list(protocol["learned_models"])
    refinement_models = list(protocol["refinement_models"])
    for model in base_models:
        for seed in seeds:
            run_hashes.update(verify_run(BASE / "runs" / model / f"seed_{seed}", model, seed))
    for model in refinement_models:
        for seed in seeds:
            run_hashes.update(verify_run(REFINEMENT / "runs" / model / f"seed_{seed}", model, seed))
    for model in deterministic:
        run_hashes.update(verify_run(BASE / "runs" / model, model, None))

    prepared = ROOT / protocol["prepared_development"]
    reference = "ignnk_style_impute_tcn"
    amendment = {
        "status": "BENCHMARK_AMENDMENT_LOCKED_HOUSTON_2025_NOT_DOWNLOADED",
        "created_utc": utc_now(),
        "title": (
            "Benchmarking Outage-Resilient Probabilistic Forecasting in Air-Quality Sensor Networks"
        ),
        "amendment_timing": (
            "Post-development and pre-holdout. This amendment transparently changes the "
            "paper framing after the prespecified adaptive candidate failed its development "
            "gate; it is not claimed as prospective preregistration."
        ),
        "scientific_objective": (
            "Benchmark frozen forecasting approaches under clean operation, synthetic "
            "single-station and regional outages, and observable natural missingness, then "
            "confirm the locked comparison once on untouched Houston-Harris County 2025."
        ),
        "framing": "multi-model outage-resilience benchmark; no proposed winning model",
        "development_selected_statistical_reference": reference,
        "reference_role": (
            "Reference for paired confirmatory contrasts only because it had the strongest "
            "development primary profile; it is not a proposed model and is not a claim of "
            "exact IGNNK reproduction."
        ),
        "candidate_status": {
            "adaptive_graph_mask_tcn": "failed development advance gate; retained and reported",
            "fixed_graph_equal_training": (
                "retained as an equal-training control; not reframed as a novel primary model"
            ),
            "development_decision_sha256": sha256_file(ANALYSIS / "development_decision.json"),
            "reframing_assessment_sha256": sha256_file(
                ANALYSIS / "fixed_control_reframing_assessment.json"
            ),
            "reframing_recommendation": reframing["recommendation"],
        },
        "models": {
            "learned": learned,
            "deterministic": deterministic,
            "refinement_models": refinement_models,
            "seeds": seeds,
        },
        "checkpoint_roots": {
            "base": rel(BASE / "runs"),
            "refinement": rel(REFINEMENT / "runs"),
        },
        "data": {
            "network": unseen["dataset"]["name"],
            "state_fips": unseen["dataset"]["state_fips"],
            "county_fips": unseen["dataset"]["county_fips"],
            "stations": unseen["dataset"]["station_ids"],
            "pollutants": unseen["dataset"]["pollutants"],
            "development": "2021-2024",
            "external_confirmatory_holdout": "2025",
            "prepared_development": rel(prepared),
            "prepared_development_sha256": sha256_file(prepared),
        },
        "forecast": {
            name: protocol[name]
            for name in (
                "lookback",
                "horizon",
                "reported_horizons",
                "test_stride",
                "fill_limit",
                "corruption_seed",
            )
        },
        "conditions": {
            "development_and_test": protocol["development_conditions"],
            "primary": protocol["primary_test_conditions"],
            "regional_robustness": protocol["regional_robustness_conditions"],
            "multi_station_outage_groups": protocol["multi_station_outage_groups"],
            "natural_comissingness": protocol["natural_comissingness"],
        },
        "endpoints": {
            "primary": protocol["primary_endpoints"],
            "secondary": protocol["secondary_endpoints"],
        },
        "calibration": protocol["calibration"],
        "statistics": {
            **protocol["statistics"],
            "primary_contrasts": (
                "development-selected statistical reference versus every other method at "
                "6-hour and 24-hour single-station outages for MASE and Q95 Brier"
            ),
            "reference_selected_before_holdout": True,
        },
        "reporting_rules": [
            "Report all frozen models and both deterministic baselines regardless of direction.",
            "Report uncalibrated and frozen-development-calibrated results.",
            "Report the failed adaptive candidate and negative ablation/control findings.",
            "Report 24-hour undercoverage and natural-event scarcity explicitly.",
            "Do not rank, select, retrain, recalibrate, or revise using Houston 2025 labels.",
            "Treat seed variation as optimization variation, not independent replication.",
        ],
        "holdout_state_at_amendment": {
            "downloaded": False,
            "content_parsed": False,
            "metrics_computed": False,
        },
    }
    write_json(AMENDMENT, amendment)

    source_paths = [
        PROTOCOL,
        UNSEEN,
        ROOT / "journal_protocol/third_development_preprocessing_protocol.json",
        AMENDMENT,
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
        ROOT / "scripts/freeze_third_benchmark_execution.py",
        Path(__file__).resolve(),
        ROOT / "pyproject.toml",
    ]
    analysis_hashes = {
        rel(ANALYSIS / name): digest for name, digest in analysis_manifest["output_sha256"].items()
    }
    record = {
        "status": "THIRD_BENCHMARK_DEVELOPMENT_FROZEN_HOUSTON_2025_NOT_DOWNLOADED",
        "created_utc": utc_now(),
        "amendment": rel(AMENDMENT),
        "amendment_sha256": sha256_file(AMENDMENT),
        "statistical_reference": reference,
        "reference_is_proposed_model": False,
        "models": amendment["models"],
        "checkpoint_roots": amendment["checkpoint_roots"],
        "prepared_development": rel(prepared),
        "prepared_development_sha256": sha256_file(prepared),
        "source_sha256": {rel(path): sha256_file(path) for path in source_paths},
        "run_artifact_sha256": run_hashes,
        "development_analysis_sha256": analysis_hashes,
        "holdout": {
            "year": 2025,
            "downloaded": False,
            "content_parsed": False,
            "metrics_computed": False,
        },
        "holdout_download_authorized": True,
        "holdout_evaluation_authorized": False,
        "next_gate": (
            "Download the exact Houston-Harris County 2025 pollutant grid without parsing, "
            "seal its manifest, then freeze the one-time benchmark execution."
        ),
    }
    write_json(OUTPUT, record)
    print(f"Benchmark amendment: {AMENDMENT.resolve()}")
    print(f"Development freeze: {OUTPUT.resolve()}")
    print(f"SHA-256: {sha256_file(OUTPUT)}")
    print("Houston 2025 may now be downloaded and sealed, but not parsed.")


if __name__ == "__main__":
    main()
