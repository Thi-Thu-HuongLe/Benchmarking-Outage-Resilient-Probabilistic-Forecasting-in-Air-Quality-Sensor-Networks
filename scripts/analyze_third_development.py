"""Analyze the locked Houston development comparison without opening 2025."""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqriskformer.development_analysis import (
    finite_column_mean,
    hourly_loss_grid,
    load_trailing_prediction,
    origin_macro_losses,
    verify_paired_predictions,
)
from aqriskformer.outage_data import load_outage_data
from aqriskformer.outage_evaluation import apply_horizon_sigma_calibration
from aqriskformer.statistics import (
    diebold_mariano,
    holm_adjust,
    paired_moving_block_bootstrap,
)
from aqriskformer.utils import read_json, sha256_file, utc_now, write_json

DEFAULT_PROTOCOL = ROOT / "journal_protocol/third_comparison_protocol.json"
DEFAULT_BASE_ROOT = ROOT / "experiment_protocol/results_third/development/full_locked"
DEFAULT_REFINEMENT_ROOT = (
    ROOT / "experiment_protocol/results_third/development/full_refinement_locked"
)
DEFAULT_OUTPUT = ROOT / "experiment_protocol/results_third/development/analysis_locked"
DURATIONS = (6, 24)
ENDPOINTS = ("mase", "q95_brier")
CALIBRATION_STATES = ("calibrated", "uncalibrated")
SUMMARY_METRICS = (
    "mase",
    "q95_brier",
    "q95_ap",
    "q95_brier_skill",
    "picp80",
    "width80_mase_scaled",
    "interval_score80_mase_scaled",
    "crps_mase_scaled",
    "negative_mass",
)


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_dir(root: Path, model: str, seed: int | None) -> Path:
    directory = root / "runs" / model
    return directory if seed is None else directory / f"seed_{seed}"


def root_for_model(
    model: str,
    refinement_models: set[str],
    base_root: Path,
    refinement_root: Path,
) -> Path:
    return refinement_root if model in refinement_models else base_root


def calibration_factors(directory: Path, duration: int) -> np.ndarray:
    key = f"single_station_trailing_{duration}h__combined_factors"
    path = directory / "calibration.npz"
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise ValueError(f"Missing calibration array {key}: {path}")
        factors = archive[key].copy()
    if factors.ndim != 2 or not np.all(np.isfinite(factors) & (factors > 0)):
        raise ValueError(f"Invalid horizon-by-pollutant calibration: {path}")
    return factors


def trailing_prediction(
    root: Path,
    model: str,
    seed: int,
    duration: int,
    calibration_state: str,
) -> dict[str, np.ndarray]:
    directory = run_dir(root, model, seed)
    prediction = load_trailing_prediction(
        directory / "trailing_validation_predictions.npz", duration
    )
    if calibration_state == "calibrated":
        return apply_horizon_sigma_calibration(prediction, calibration_factors(directory, duration))
    if calibration_state != "uncalibrated":
        raise ValueError(f"Unknown calibration state: {calibration_state}")
    return prediction


def adjust_finite_holm(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    adjusted = np.full(len(result), np.nan)
    finite = np.isfinite(result["dm_p_value"].to_numpy(float))
    adjusted[finite] = holm_adjust(result.loc[finite, "dm_p_value"].to_numpy(float))
    result["dm_p_holm"] = adjusted
    result["dm_significant_holm_0_05"] = adjusted < 0.05
    return result


def paired_tests(
    *,
    candidate: str,
    comparators: list[str],
    refinement_models: set[str],
    base_root: Path,
    refinement_root: Path,
    data,
    seeds: list[int],
    block_length: int,
    resamples: int,
    calibration_state: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_root = root_for_model(candidate, refinement_models, base_root, refinement_root)
    primary_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for comparator_index, comparator in enumerate(comparators):
        comparator_root = root_for_model(comparator, refinement_models, base_root, refinement_root)
        for duration in DURATIONS:
            candidate_losses_by_endpoint = {endpoint: [] for endpoint in ENDPOINTS}
            comparator_losses_by_endpoint = {endpoint: [] for endpoint in ENDPOINTS}
            for seed in seeds:
                candidate_prediction = trailing_prediction(
                    candidate_root,
                    candidate,
                    seed,
                    duration,
                    calibration_state,
                )
                comparator_prediction = trailing_prediction(
                    comparator_root,
                    comparator,
                    seed,
                    duration,
                    calibration_state,
                )
                verify_paired_predictions(candidate_prediction, comparator_prediction)
                candidate_losses = origin_macro_losses(candidate_prediction, data)
                comparator_losses = origin_macro_losses(comparator_prediction, data)
                for endpoint in ENDPOINTS:
                    candidate_grid = hourly_loss_grid(
                        candidate_prediction["origins"], candidate_losses[endpoint]
                    )
                    comparator_grid = hourly_loss_grid(
                        comparator_prediction["origins"], comparator_losses[endpoint]
                    )
                    candidate_losses_by_endpoint[endpoint].append(candidate_grid)
                    comparator_losses_by_endpoint[endpoint].append(comparator_grid)
                    valid = np.isfinite(candidate_grid) & np.isfinite(comparator_grid)
                    difference = candidate_grid[valid] - comparator_grid[valid]
                    comparator_mean = float(comparator_grid[valid].mean())
                    seed_rows.append(
                        {
                            "calibration": calibration_state,
                            "candidate": candidate,
                            "comparator": comparator,
                            "condition": f"each_station_trailing_{duration}h_outage",
                            "duration_hours": duration,
                            "endpoint": endpoint,
                            "seed": seed,
                            "paired_origins": int(valid.sum()),
                            "candidate_mean": float(candidate_grid[valid].mean()),
                            "comparator_mean": comparator_mean,
                            "mean_difference_candidate_minus_comparator": float(difference.mean()),
                            "relative_benefit_percent": float(
                                -100.0 * difference.mean() / comparator_mean
                            ),
                            "candidate_win": bool(difference.mean() < 0),
                        }
                    )
            for endpoint_index, endpoint in enumerate(ENDPOINTS):
                candidate_mean = finite_column_mean(candidate_losses_by_endpoint[endpoint])
                comparator_mean = finite_column_mean(comparator_losses_by_endpoint[endpoint])
                bootstrap = paired_moving_block_bootstrap(
                    candidate_mean,
                    comparator_mean,
                    block_length=block_length,
                    resamples=resamples,
                    seed=(
                        81317
                        + 1000 * comparator_index
                        + 10 * duration
                        + endpoint_index
                        + (100_000 if calibration_state == "uncalibrated" else 0)
                    ),
                )
                dm = diebold_mariano(
                    candidate_mean,
                    comparator_mean,
                    horizon=1,
                    newey_west_lag=block_length - 1,
                )
                valid = np.isfinite(candidate_mean) & np.isfinite(comparator_mean)
                baseline_value = float(comparator_mean[valid].mean())
                primary_rows.append(
                    {
                        "calibration": calibration_state,
                        "candidate": candidate,
                        "comparator": comparator,
                        "condition": f"each_station_trailing_{duration}h_outage",
                        "duration_hours": duration,
                        "endpoint": endpoint,
                        "seeds_averaged_per_origin": len(seeds),
                        "paired_origins": int(valid.sum()),
                        "candidate_mean": float(candidate_mean[valid].mean()),
                        "comparator_mean": baseline_value,
                        "mean_difference_candidate_minus_comparator": bootstrap["mean_difference"],
                        "relative_benefit_percent": float(
                            -100.0 * bootstrap["mean_difference"] / baseline_value
                        ),
                        "bootstrap_ci95_lower": bootstrap["ci95_lower"],
                        "bootstrap_ci95_upper": bootstrap["ci95_upper"],
                        "candidate_better_ci95": bool(bootstrap["ci95_upper"] < 0),
                        "candidate_worse_ci95": bool(bootstrap["ci95_lower"] > 0),
                        "dm_statistic": dm["statistic"],
                        "dm_p_value": dm["p_value"],
                    }
                )
            log(f"paired tests complete: {calibration_state}, {comparator}, {duration} h")
    return adjust_finite_holm(pd.DataFrame(primary_rows)), pd.DataFrame(seed_rows)


def condition_families(metrics: dict[str, object]) -> dict[str, list[str]]:
    keys = list(metrics)
    return {
        "clean": ["clean"],
        "random_block_6h": ["random_block_6h"],
        "random_block_24h": ["random_block_24h"],
        "single_station_full_history": [key for key in keys if key.startswith("station_full_")],
        "single_station_trailing_6h": [
            key for key in keys if key.startswith("station_trailing_6h_")
        ],
        "single_station_trailing_24h": [
            key for key in keys if key.startswith("station_trailing_24h_")
        ],
        "multi_station_trailing_6h": [
            key for key in keys if key.startswith("stations_trailing_6h_")
        ],
        "multi_station_trailing_24h": [
            key for key in keys if key.startswith("stations_trailing_24h_")
        ],
    }


def development_metric_summary(
    *,
    models: list[str],
    deterministic_models: set[str],
    refinement_models: set[str],
    base_root: Path,
    refinement_root: Path,
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for model in models:
        root = root_for_model(model, refinement_models, base_root, refinement_root)
        model_seeds: list[int | None] = [None] if model in deterministic_models else list(seeds)
        for seed in model_seeds:
            metrics = read_json(run_dir(root, model, seed) / "validation_metrics.json")
            for family, conditions in condition_families(metrics).items():
                if not conditions:
                    continue
                for calibration_state in CALIBRATION_STATES:
                    for metric in SUMMARY_METRICS:
                        values = [
                            float(metrics[condition][calibration_state]["summary"][metric])
                            for condition in conditions
                        ]
                        rows.append(
                            {
                                "model": model,
                                "seed": seed,
                                "condition_family": family,
                                "scenario_aggregation": "equal_weight_macro",
                                "calibration": calibration_state,
                                "metric": metric,
                                "value": float(np.mean(values)),
                                "scenario_count": len(values),
                            }
                        )
    detail = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    grouped = detail.groupby(["model", "condition_family", "calibration", "metric"], sort=False)
    for keys, frame in grouped:
        summary_rows.append(
            {
                "model": keys[0],
                "condition_family": keys[1],
                "calibration": keys[2],
                "metric": keys[3],
                "mean": float(frame["value"].mean()),
                "sd_across_seeds": (
                    float(frame["value"].std(ddof=1)) if len(frame) > 1 else np.nan
                ),
                "runs": len(frame),
                "scenario_count": int(frame["scenario_count"].iloc[0]),
                "scenario_aggregation": "equal_weight_macro",
            }
        )
    return detail, pd.DataFrame(summary_rows)


def natural_case_counts(candidate_directory: Path) -> pd.DataFrame:
    metrics = read_json(candidate_directory / "validation_metrics.json")
    natural = metrics["natural_comissingness"]
    rows: list[dict[str, object]] = []
    for duration in DURATIONS:
        for station, item in natural[f"trailing_{duration}h"].items():
            rows.append(
                {
                    "case_type": "core_pollutant_comissingness",
                    "duration_hours": duration,
                    "pollutant": "PM2.5_NO2_O3_core",
                    "station": station,
                    "forecast_origins": int(item["forecast_origins"]),
                    "inferential_role": "descriptive_secondary_only",
                }
            )
        pollutant_cases = natural["per_pollutant_gaps"][f"trailing_{duration}h"]
        for pollutant, stations in pollutant_cases.items():
            for station, item in stations.items():
                rows.append(
                    {
                        "case_type": "pollutant_specific_gap",
                        "duration_hours": duration,
                        "pollutant": pollutant,
                        "station": station,
                        "forecast_origins": int(item["forecast_origins"]),
                        "inferential_role": "descriptive_secondary_only",
                    }
                )
    return pd.DataFrame(rows)


def clean_guardrails(
    summary: pd.DataFrame,
    candidate: str,
    comparators: list[str],
    maximum_relative_degradation: float,
) -> pd.DataFrame:
    clean = summary[
        (summary["condition_family"] == "clean")
        & (summary["calibration"] == "calibrated")
        & (summary["metric"].isin(ENDPOINTS))
    ]
    rows: list[dict[str, object]] = []
    for comparator in comparators:
        for endpoint in ENDPOINTS:
            candidate_mean = float(
                clean[(clean["model"] == candidate) & (clean["metric"] == endpoint)]["mean"].iloc[0]
            )
            comparator_mean = float(
                clean[(clean["model"] == comparator) & (clean["metric"] == endpoint)]["mean"].iloc[
                    0
                ]
            )
            relative_degradation = (candidate_mean - comparator_mean) / comparator_mean
            rows.append(
                {
                    "candidate": candidate,
                    "comparator": comparator,
                    "endpoint": endpoint,
                    "candidate_mean": candidate_mean,
                    "comparator_mean": comparator_mean,
                    "relative_degradation": relative_degradation,
                    "maximum_allowed_relative_degradation": (maximum_relative_degradation),
                    "guardrail_pass": bool(relative_degradation <= maximum_relative_degradation),
                }
            )
    return pd.DataFrame(rows)


def comparator_evidence(
    primary: pd.DataFrame, per_seed: pd.DataFrame, comparator: str
) -> dict[str, object]:
    cells = primary[primary["comparator"] == comparator]
    seed_cells = per_seed[per_seed["comparator"] == comparator]
    return {
        "primary_cells": len(cells),
        "mean_benefit_cells": int((cells["relative_benefit_percent"] > 0).sum()),
        "ci95_better_cells": int(cells["candidate_better_ci95"].sum()),
        "holm_significant_better_cells": int(
            (
                (cells["dm_significant_holm_0_05"])
                & (cells["mean_difference_candidate_minus_comparator"] < 0)
            ).sum()
        ),
        "seed_wins": int(seed_cells["candidate_win"].sum()),
        "seed_comparisons": len(seed_cells),
        "relative_benefit_percent_min": float(cells["relative_benefit_percent"].min()),
        "relative_benefit_percent_max": float(cells["relative_benefit_percent"].max()),
    }


def decision_record(
    *,
    protocol: dict[str, object],
    primary: pd.DataFrame,
    per_seed: pd.DataFrame,
    guardrails: pd.DataFrame,
) -> dict[str, object]:
    candidate = str(protocol["candidate"])
    control = str(protocol["candidate_training_control"])
    base = str(protocol["base_model_for_candidate_and_control"])
    control_rows = primary[primary["comparator"] == control]
    control_mean_better_all = bool(
        (control_rows["mean_difference_candidate_minus_comparator"] < 0).all()
    )
    guardrail_pass = bool(guardrails["guardrail_pass"].all())
    advance = control_mean_better_all and guardrail_pass
    return {
        "created_utc": utc_now(),
        "development_only": True,
        "holdout_accessed": False,
        "candidate": candidate,
        "candidate_training_control": control,
        "base_model": base,
        "decision_rule_status": (
            "conservative post-run reporting gate; the locked protocol prespecified "
            "comparisons and guardrails but not a binary advancement formula"
        ),
        "decision_rule": (
            "Advance adaptive graph only if its calibrated mean loss is lower than the "
            "fixed-graph equal-training control in all four prespecified duration-by-"
            "endpoint cells and every clean guardrail passes."
        ),
        "candidate_vs_equal_training_control": comparator_evidence(primary, per_seed, control),
        "candidate_vs_frozen_base": comparator_evidence(primary, per_seed, base),
        "clean_guardrails_all_pass": guardrail_pass,
        "advance_adaptive_graph_to_holdout": advance,
        "holdout_action_authorized": False,
        "decision": (
            "FREEZE_ADAPTIVE_GRAPH_FOR_HOLDOUT" if advance else "DO_NOT_ADVANCE_ADAPTIVE_GRAPH"
        ),
        "next_gate": (
            "freeze reporting and holdout execution before one-time access"
            if advance
            else (
                "stop before Houston 2025; retain the adverse development result and "
                "review the paper claim without tuning on the sealed holdout"
            )
        ),
        "claim_boundary": (
            "Optimization seeds quantify training variability, not independent dataset "
            "replication. Natural missingness cases remain descriptive. No Houston 2025 "
            "claim is permitted from this analysis."
        ),
    }


def reframing_assessment(
    primary: pd.DataFrame,
    *,
    assessed_model: str,
    prespecified_candidate: str,
) -> dict[str, object]:
    comparator_rows = []
    for comparator, frame in primary.groupby("comparator", sort=False):
        differences = frame["mean_difference_candidate_minus_comparator"]
        comparator_rows.append(
            {
                "comparator": comparator,
                "primary_cells": len(frame),
                "assessed_model_better_cells": int((differences < 0).sum()),
                "assessed_model_worse_cells": int((differences > 0).sum()),
                "uniformly_better": bool((differences < 0).all()),
                "uniformly_worse": bool((differences > 0).all()),
                "relative_benefit_percent_min": float(frame["relative_benefit_percent"].min()),
                "relative_benefit_percent_max": float(frame["relative_benefit_percent"].max()),
            }
        )
    dominated_by = [row["comparator"] for row in comparator_rows if row["uniformly_worse"]]
    return {
        "created_utc": utc_now(),
        "development_only": True,
        "holdout_accessed": False,
        "assessed_model": assessed_model,
        "prespecified_role": "equal-training control",
        "prespecified_candidate": prespecified_candidate,
        "comparisons": comparator_rows,
        "uniformly_dominated_by": dominated_by,
        "recommendation": "KEEP_AS_STRONG_CONTROL_NOT_NOVEL_PRIMARY_MODEL",
        "reason": (
            "This model was prespecified as a training control, was considered only after "
            "the adaptive candidate failed its development gate, and is uniformly worse "
            "than at least one locked comparator on the four primary cells."
            if dominated_by
            else (
                "This model was prespecified as a training control and cannot be relabeled "
                "as a novel primary model without a transparent protocol amendment."
            )
        ),
        "holdout_action_authorized": False,
    }


def validate_inputs(
    protocol_path: Path,
    protocol: dict[str, object],
    base_root: Path,
    refinement_root: Path,
) -> None:
    if "NOT_DOWNLOADED" not in str(protocol["status"]):
        raise RuntimeError("Protocol does not preserve the unopened Houston holdout")
    for root in (base_root, refinement_root):
        status = read_json(root / "status.json")
        launch = read_json(root / "launch_manifest.json")
        if status.get("state") != "complete":
            raise RuntimeError(f"Development run is incomplete: {root}")
        if launch.get("holdout_accessed") is not False:
            raise RuntimeError(f"Development manifest reports holdout access: {root}")
        recorded = launch["signature"]["source_hashes"].get(
            str(protocol_path.relative_to(ROOT)).replace("/", "\\")
        )
        if recorded != sha256_file(protocol_path):
            raise RuntimeError(f"Protocol hash mismatch in launch manifest: {root}")
        prepared = ROOT / str(protocol["prepared_development"])
        if launch["signature"]["prepared_sha256"] != sha256_file(prepared):
            raise RuntimeError(f"Prepared development hash mismatch: {root}")


def artifact_manifest(
    *,
    output: Path,
    inputs: list[Path],
    outputs: list[Path],
) -> dict[str, object]:
    return {
        "created_utc": utc_now(),
        "development_only": True,
        "holdout_accessed": False,
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(set(inputs))
        },
        "output_sha256": {path.name: sha256_file(path) for path in outputs},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--refinement-root", type=Path, default=DEFAULT_REFINEMENT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    base_root = args.base_root.resolve()
    refinement_root = args.refinement_root.resolve()
    output = args.output.resolve()
    protocol = read_json(protocol_path)
    validate_inputs(protocol_path, protocol, base_root, refinement_root)

    seeds = [int(seed) for seed in protocol["seeds"]]
    if seeds != [42, 123, 2026, 3407, 7777]:
        raise ValueError(f"Unexpected locked seed set: {seeds}")
    candidate = str(protocol["candidate"])
    refinement_models = set(protocol["refinement_models"])
    learned_models = list(protocol["learned_models"])
    deterministic_models = set(protocol["deterministic_models"])
    comparators = list(
        dict.fromkeys([str(protocol["candidate_training_control"]), *learned_models])
    )
    comparators = [model for model in comparators if model != candidate]
    all_models = list(dict.fromkeys([candidate, *comparators, *protocol["deterministic_models"]]))
    data = load_outage_data(ROOT / str(protocol["prepared_development"]), development_only=True)
    statistics = protocol["statistics"]

    primary_frames = []
    seed_frames = []
    for calibration_state in CALIBRATION_STATES:
        primary, per_seed = paired_tests(
            candidate=candidate,
            comparators=comparators,
            refinement_models=refinement_models,
            base_root=base_root,
            refinement_root=refinement_root,
            data=data,
            seeds=seeds,
            block_length=int(statistics["temporal_block_length_hours"]),
            resamples=int(statistics["bootstrap_resamples"]),
            calibration_state=calibration_state,
        )
        primary_frames.append(primary)
        seed_frames.append(per_seed)
    primary_all = pd.concat(primary_frames, ignore_index=True)
    per_seed_all = pd.concat(seed_frames, ignore_index=True)
    calibrated_primary = primary_all[primary_all["calibration"] == "calibrated"].copy()
    calibrated_per_seed = per_seed_all[per_seed_all["calibration"] == "calibrated"].copy()

    control = str(protocol["candidate_training_control"])
    control_comparators = [
        model for model in dict.fromkeys([candidate, *learned_models]) if model != control
    ]
    control_primary_frames = []
    control_seed_frames = []
    for calibration_state in CALIBRATION_STATES:
        control_primary, control_per_seed = paired_tests(
            candidate=control,
            comparators=control_comparators,
            refinement_models=refinement_models,
            base_root=base_root,
            refinement_root=refinement_root,
            data=data,
            seeds=seeds,
            block_length=int(statistics["temporal_block_length_hours"]),
            resamples=int(statistics["bootstrap_resamples"]),
            calibration_state=calibration_state,
        )
        control_primary_frames.append(control_primary)
        control_seed_frames.append(control_per_seed)
    control_primary_all = pd.concat(control_primary_frames, ignore_index=True)
    control_per_seed_all = pd.concat(control_seed_frames, ignore_index=True)
    control_assessment = reframing_assessment(
        control_primary_all[control_primary_all["calibration"] == "calibrated"],
        assessed_model=control,
        prespecified_candidate=candidate,
    )

    detail, summary = development_metric_summary(
        models=all_models,
        deterministic_models=deterministic_models,
        refinement_models=refinement_models,
        base_root=base_root,
        refinement_root=refinement_root,
        seeds=seeds,
    )
    guardrail_comparators = list(
        dict.fromkeys(
            [
                str(protocol["candidate_training_control"]),
                str(protocol["base_model_for_candidate_and_control"]),
                "local_tcn",
            ]
        )
    )
    guardrails = clean_guardrails(
        summary,
        candidate,
        guardrail_comparators,
        float(protocol["clean_guardrail"]["maximum_relative_degradation_vs_each_local_control"]),
    )
    natural_counts = natural_case_counts(run_dir(refinement_root, candidate, seeds[0]))
    decision = decision_record(
        protocol=protocol,
        primary=calibrated_primary,
        per_seed=calibrated_per_seed,
        guardrails=guardrails,
    )

    frames = {
        "primary_paired_tests.csv": primary_all,
        "per_seed_effects.csv": per_seed_all,
        "development_metrics_per_run.csv": detail,
        "development_metrics_summary.csv": summary,
        "clean_guardrails.csv": guardrails,
        "natural_case_counts.csv": natural_counts,
        "fixed_control_paired_tests.csv": control_primary_all,
        "fixed_control_per_seed_effects.csv": control_per_seed_all,
    }
    output.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for name, frame in frames.items():
        path = output / name
        atomic_csv(frame, path)
        output_paths.append(path)
    decision_path = output / "development_decision.json"
    write_json(decision_path, decision)
    output_paths.append(decision_path)
    control_assessment_path = output / "fixed_control_reframing_assessment.json"
    write_json(control_assessment_path, control_assessment)
    output_paths.append(control_assessment_path)

    input_paths = [
        Path(__file__).resolve(),
        protocol_path,
        ROOT / str(protocol["prepared_development"]),
        ROOT / "src/aqriskformer/development_analysis.py",
        ROOT / "src/aqriskformer/outage_evaluation.py",
        ROOT / "src/aqriskformer/statistics.py",
        base_root / "status.json",
        base_root / "launch_manifest.json",
        refinement_root / "status.json",
        refinement_root / "launch_manifest.json",
    ]
    for model in all_models:
        root = root_for_model(model, refinement_models, base_root, refinement_root)
        model_seeds: list[int | None] = [None] if model in deterministic_models else seeds
        for seed in model_seeds:
            directory = run_dir(root, model, seed)
            input_paths.extend(
                [
                    directory / "complete.json",
                    directory / "validation_metrics.json",
                    directory / "calibration.npz",
                    directory / "trailing_validation_predictions.npz",
                ]
            )
    manifest = artifact_manifest(output=output, inputs=input_paths, outputs=output_paths)
    write_json(output / "analysis_manifest.json", manifest)
    log(f"analysis complete: {output}")
    log(f"decision: {decision['decision']}")


if __name__ == "__main__":
    main()
