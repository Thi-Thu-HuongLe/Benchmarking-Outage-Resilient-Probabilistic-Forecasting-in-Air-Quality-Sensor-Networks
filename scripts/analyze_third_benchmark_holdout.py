"""Report the frozen Houston 2025 multi-model outage-resilience benchmark."""

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
from aqriskformer.epa_aqs_holdout import (
    load_holdout_outage_data,
)
from aqriskformer.outage_evaluation import apply_horizon_sigma_calibration
from aqriskformer.statistics import (
    diebold_mariano,
    holm_adjust,
    paired_moving_block_bootstrap,
)
from aqriskformer.utils import read_json, sha256_file, utc_now, write_json

PROTOCOL = ROOT / "journal_protocol/third_benchmark_final_holdout_protocol.json"
FREEZE = ROOT / "journal_protocol/third_benchmark_analysis_execution_freeze.json"
CALIBRATION = "frozen_development_calibration"


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_analysis_freeze(root: Path, path: Path) -> dict[str, object]:
    freeze = read_json(path)
    if freeze.get("status") != "FINAL_ANALYSIS_EXECUTION_FROZEN_AFTER_BENCHMARK":
        raise RuntimeError("Final benchmark analysis is not frozen")
    if freeze.get("analysis_authorized") is not True:
        raise RuntimeError("Analysis freeze does not authorize reporting")
    project_root = root.resolve()
    for section in ("source_sha256", "input_sha256"):
        for name, expected in freeze[section].items():
            candidate = Path(name)
            candidate = candidate if candidate.is_absolute() else project_root / candidate
            if not candidate.is_file() or sha256_file(candidate) != expected:
                raise RuntimeError(f"Post-analysis-freeze hash mismatch: {candidate}")
    return freeze


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def result_dir(root: Path, model: str, seed: int | None) -> Path:
    base = root / "runs" / model
    return base if seed is None else base / f"seed_{seed}"


def run_seed(final: dict[str, object], model: str, seed: int) -> int | None:
    return None if model in final["deterministic_comparators"] else seed


def all_models(final: dict[str, object]) -> list[str]:
    return [
        final["reference_model"],
        *final["learned_comparators"],
        *final["deterministic_comparators"],
    ]


def calibrated_prediction(
    root: Path,
    model: str,
    seed: int | None,
    duration: int,
) -> dict[str, np.ndarray]:
    directory = result_dir(root, model, seed)
    prediction = load_trailing_prediction(directory / "trailing_test_predictions.npz", duration)
    key = f"single_station_trailing_{duration}h__combined_factors"
    with np.load(directory / "frozen_calibration.npz", allow_pickle=False) as archive:
        if key not in archive.files:
            raise ValueError(f"Missing {key}: {directory}")
        factors = archive[key].copy()
    return apply_horizon_sigma_calibration(prediction, factors)


def primary_reference_tests(
    root: Path,
    final: dict[str, object],
    data,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = str(final["reference_model"])
    comparators = [model for model in all_models(final) if model != reference]
    seeds = [int(seed) for seed in final["seeds"]]
    endpoints = tuple(final["endpoints"]["primary"])
    block = int(final["statistics"]["temporal_block_length_hours"])
    resamples = int(final["statistics"]["bootstrap_resamples"])
    rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for comparator_index, comparator in enumerate(comparators):
        for duration in (6, 24):
            reference_by_endpoint = {endpoint: [] for endpoint in endpoints}
            comparator_by_endpoint = {endpoint: [] for endpoint in endpoints}
            for seed in seeds:
                reference_prediction = calibrated_prediction(root, reference, seed, duration)
                comparator_prediction = calibrated_prediction(
                    root, comparator, run_seed(final, comparator, seed), duration
                )
                verify_paired_predictions(reference_prediction, comparator_prediction)
                reference_losses = origin_macro_losses(reference_prediction, data)
                comparator_losses = origin_macro_losses(comparator_prediction, data)
                for endpoint in endpoints:
                    reference_grid = hourly_loss_grid(
                        reference_prediction["origins"], reference_losses[endpoint]
                    )
                    comparator_grid = hourly_loss_grid(
                        comparator_prediction["origins"], comparator_losses[endpoint]
                    )
                    reference_by_endpoint[endpoint].append(reference_grid)
                    comparator_by_endpoint[endpoint].append(comparator_grid)
                    valid = np.isfinite(reference_grid) & np.isfinite(comparator_grid)
                    difference = reference_grid[valid] - comparator_grid[valid]
                    comparator_mean = float(comparator_grid[valid].mean())
                    seed_rows.append(
                        {
                            "reference": reference,
                            "comparator": comparator,
                            "duration_hours": duration,
                            "endpoint": endpoint,
                            "seed": seed,
                            "paired_origins": int(valid.sum()),
                            "reference_mean": float(reference_grid[valid].mean()),
                            "comparator_mean": comparator_mean,
                            "mean_difference_reference_minus_comparator": float(difference.mean()),
                            "relative_benefit_percent": float(
                                -100 * difference.mean() / comparator_mean
                            ),
                            "reference_win": bool(difference.mean() < 0),
                        }
                    )
            for endpoint_index, endpoint in enumerate(endpoints):
                reference_mean = finite_column_mean(reference_by_endpoint[endpoint])
                comparator_mean = finite_column_mean(comparator_by_endpoint[endpoint])
                bootstrap = paired_moving_block_bootstrap(
                    reference_mean,
                    comparator_mean,
                    block_length=block,
                    resamples=resamples,
                    seed=2025 + 100 * comparator_index + 10 * duration + endpoint_index,
                )
                dm = diebold_mariano(
                    reference_mean,
                    comparator_mean,
                    horizon=1,
                    newey_west_lag=block - 1,
                )
                valid = np.isfinite(reference_mean) & np.isfinite(comparator_mean)
                comparator_average = float(comparator_mean[valid].mean())
                rows.append(
                    {
                        "reference": reference,
                        "comparator": comparator,
                        "condition": f"each_station_trailing_{duration}h_outage",
                        "duration_hours": duration,
                        "endpoint": endpoint,
                        "seeds_averaged_per_origin": len(seeds),
                        "paired_origins": int(valid.sum()),
                        "reference_mean": float(reference_mean[valid].mean()),
                        "comparator_mean": comparator_average,
                        "mean_difference_reference_minus_comparator": bootstrap["mean_difference"],
                        "relative_benefit_percent": float(
                            -100 * bootstrap["mean_difference"] / comparator_average
                        ),
                        "bootstrap_ci95_lower": bootstrap["ci95_lower"],
                        "bootstrap_ci95_upper": bootstrap["ci95_upper"],
                        "reference_better_ci95": bool(bootstrap["ci95_upper"] < 0),
                        "dm_statistic": dm["statistic"],
                        "dm_p_value": dm["p_value"],
                    }
                )
                log(f"paired test complete: {comparator}, {duration} h, {endpoint}")
    primary = pd.DataFrame(rows)
    primary["dm_p_holm"] = holm_adjust(primary["dm_p_value"].to_numpy())
    primary["dm_significant_holm_0_05"] = primary["dm_p_holm"] < 0.05
    return primary, pd.DataFrame(seed_rows)


def mean_metric_summaries(items: list[dict[str, object]]) -> dict[str, float | int]:
    if not items:
        return {}
    keys = set.intersection(*(set(item) for item in items))
    result: dict[str, float | int] = {}
    for key in sorted(keys):
        values = [item[key] for item in items if item[key] is not None]
        if not values or isinstance(values[0], (str, bool, dict, list)):
            continue
        if key in ("observed_targets", "valid_metric_cells"):
            result[key] = int(sum(int(value) for value in values))
        else:
            result[key] = float(np.mean(values))
    return result


def condition_groups(final: dict[str, object]) -> dict[str, list[str]]:
    stations = len(
        read_json(ROOT / "journal_protocol/third_unseen_holdout_protocol.json")["dataset"][
            "station_ids"
        ]
    )
    groups = final["conditions"]["multi_station_outage_groups"]
    labels = ["-".join(str(value) for value in group) for group in groups]
    return {
        "clean": ["clean"],
        "random_block_6h": ["random_block_6h"],
        "random_block_24h": ["random_block_24h"],
        "each_station_full_history_dropout": [
            f"station_full_{station}" for station in range(stations)
        ],
        "each_station_trailing_6h_outage": [
            f"station_trailing_6h_{station}" for station in range(stations)
        ],
        "each_station_trailing_24h_outage": [
            f"station_trailing_24h_{station}" for station in range(stations)
        ],
        "regional_three_station_trailing_6h_outage": [
            f"stations_trailing_6h_{label}" for label in labels
        ],
        "regional_three_station_trailing_24h_outage": [
            f"stations_trailing_24h_{label}" for label in labels
        ],
    }


def metric_tables(
    root: Path, final: dict[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    natural_rows: list[dict[str, object]] = []
    groups = condition_groups(final)
    for model in all_models(final):
        seeds: list[int | None] = (
            [None]
            if model in final["deterministic_comparators"]
            else [int(seed) for seed in final["seeds"]]
        )
        for seed in seeds:
            metrics = read_json(result_dir(root, model, seed) / "test_metrics.json")
            for group_name, keys in groups.items():
                for calibration in ("uncalibrated", CALIBRATION):
                    summaries = [metrics[key][calibration]["summary"] for key in keys]
                    rows.append(
                        {
                            "model": model,
                            "seed": seed,
                            "condition": group_name,
                            "calibration": calibration,
                            **mean_metric_summaries(summaries),
                        }
                    )
            natural = metrics["natural_comissingness"]
            for duration in (6, 24):
                records = natural[f"trailing_{duration}h"]
                for station, item in records.items():
                    natural_rows.append(
                        {
                            "model": model,
                            "seed": seed,
                            "analysis": "core_comissingness",
                            "duration_hours": duration,
                            "pollutant": "PM2.5_NO2_O3_core",
                            "station": station,
                            "forecast_origins": int(item.get("forecast_origins", 0)),
                        }
                    )
                pollutant_records = natural["per_pollutant_gaps"][f"trailing_{duration}h"]
                for pollutant, station_records in pollutant_records.items():
                    for station, item in station_records.items():
                        natural_rows.append(
                            {
                                "model": model,
                                "seed": seed,
                                "analysis": "per_pollutant_gap",
                                "duration_hours": duration,
                                "pollutant": pollutant,
                                "station": station,
                                "forecast_origins": int(item.get("forecast_origins", 0)),
                            }
                        )
    per_run = pd.DataFrame(rows)
    metric_columns = [
        column
        for column in per_run.columns
        if column not in {"model", "seed", "condition", "calibration"}
    ]
    summary = (
        per_run.groupby(["model", "condition", "calibration"], dropna=False)[metric_columns]
        .mean(numeric_only=True)
        .reset_index()
    )
    return per_run, summary, pd.DataFrame(natural_rows)


def verify_run_complete(root: Path, final: dict[str, object]) -> list[Path]:
    status = read_json(root / "status.json")
    expected = len(final["seeds"]) * (1 + len(final["learned_comparators"])) + len(
        final["deterministic_comparators"]
    )
    if (
        status.get("state") != "complete"
        or status.get("test_labels_used_for_fitting") is not False
        or int(status.get("completed_fits", -1)) != expected
    ):
        raise RuntimeError("Frozen Houston benchmark is not complete")
    files = [root / "status.json", root / "test_access_log.json"]
    for model in all_models(final):
        seeds: list[int | None] = (
            [None]
            if model in final["deterministic_comparators"]
            else [int(seed) for seed in final["seeds"]]
        )
        for seed in seeds:
            directory = result_dir(root, model, seed)
            marker = read_json(directory / "complete.json")
            if marker.get("test_labels_used_for_fitting") is not False:
                raise RuntimeError(f"Holdout-label fitting marker found: {directory}")
            for name, expected_hash in marker["artifact_hashes"].items():
                path = directory / name
                if not path.is_file() or sha256_file(path) != expected_hash:
                    raise RuntimeError(f"Holdout artifact hash mismatch: {path}")
                files.append(path)
            files.append(directory / "complete.json")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_analysis_freeze(ROOT, FREEZE)
    final = read_json(PROTOCOL)
    root = project_path(final["output"])
    inputs = verify_run_complete(root, final)
    prepared = project_path(final["data"]["prepared_holdout"])
    data = load_holdout_outage_data(
        prepared,
        development_path=project_path(final["data"]["development_prepared"]),
    )
    output = args.output.resolve() if args.output else root / "analysis"

    primary, per_seed = primary_reference_tests(root, final, data)
    per_run, summary, natural = metric_tables(root, final)
    outputs = {
        "primary_reference_tests.csv": primary,
        "per_seed_primary_effects.csv": per_seed,
        "test_metrics_per_run.csv": per_run,
        "test_metrics_summary.csv": summary,
        "natural_case_counts.csv": natural,
    }
    output.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for name, frame in outputs.items():
        path = output / name
        atomic_csv(frame, path)
        output_paths.append(path)

    benchmark = {
        "created_utc": utc_now(),
        "holdout_accessed": True,
        "test_labels_used_for_fitting_or_selection": False,
        "title": final["title"],
        "study_type": "multi-model benchmark",
        "statistical_reference": final["reference_model"],
        "reference_is_proposed_model": False,
        "models_reported": all_models(final),
        "primary_reference_contrasts": len(primary),
        "reference_mean_better": int(
            (primary["mean_difference_reference_minus_comparator"] < 0).sum()
        ),
        "reference_better_bootstrap_ci95": int(primary["reference_better_ci95"].sum()),
        "reference_better_holm_dm_0_05": int(
            (
                (primary["mean_difference_reference_minus_comparator"] < 0)
                & primary["dm_significant_holm_0_05"]
            ).sum()
        ),
        "interpretation_rule": (
            "Report the complete benchmark, including unfavorable, null, negative-ablation, "
            "undercoverage, and sparse-natural-event findings. Houston 2025 cannot support "
            "additional model selection, recalibration, or protocol revision."
        ),
    }
    benchmark_path = output / "benchmark_summary.json"
    write_json(benchmark_path, benchmark)
    output_paths.append(benchmark_path)
    source_inputs = [Path(__file__).resolve(), PROTOCOL, FREEZE, prepared, *inputs]
    manifest_path = output / "analysis_manifest.json"
    write_json(
        manifest_path,
        {
            "created_utc": utc_now(),
            "holdout_accessed": True,
            "test_labels_used_for_fitting_or_selection": False,
            "input_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
                for path in sorted(set(source_inputs))
            },
            "output_sha256": {path.name: sha256_file(path) for path in output_paths},
        },
    )
    log(f"frozen Houston benchmark report complete: {output}")


if __name__ == "__main__":
    main()
