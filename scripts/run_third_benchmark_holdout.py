"""Run the frozen Houston 2025 multi-model outage-resilience benchmark."""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from aqriskformer.data import _calendar_features
from aqriskformer.epa_aqs_holdout import (
    load_holdout_outage_data as load_locked_holdout,
)
from aqriskformer.epa_aqs_holdout import prepare_epa_holdout as prepare_locked_holdout
from aqriskformer.outage_data import (
    OutageWindows,
    natural_comissingness_origins,
    natural_pollutant_gap_origins,
)
from aqriskformer.outage_evaluation import (
    apply_horizon_sigma_calibration,
    outage_metrics,
)
from aqriskformer.utils import read_json, sha256_file, utc_now, write_json
from scripts import run_journal_development as development
from scripts import run_journal_holdout as runner

runner.PROTOCOL_PATH = ROOT / "journal_protocol/third_benchmark_final_holdout_protocol.json"
runner.FREEZE_PATH = ROOT / "journal_protocol/third_benchmark_technical_resume_002.json"
runner.PREPROCESSING_PATH = ROOT / "journal_protocol/third_development_preprocessing_protocol.json"


def verify_technical_resume(root: str | Path, freeze_path: str | Path) -> dict[str, object]:
    """Verify the post-preparation technical resume without hiding holdout access."""
    project_root = Path(root).resolve()
    path = Path(freeze_path).resolve()
    freeze = read_json(path)
    if freeze.get("status") != "TECHNICAL_RESUME_FROZEN_AFTER_PREPARATION_BEFORE_METRICS":
        raise RuntimeError("Technical resume is not frozen")
    if freeze.get("holdout_evaluation_authorized") is not True:
        raise RuntimeError("Technical resume does not authorize evaluation")
    if freeze.get("holdout_content_parsed_at_freeze") is not True:
        raise RuntimeError("Technical resume does not disclose prior preparation")
    if freeze.get("holdout_metrics_computed_at_freeze") is not False:
        raise RuntimeError("Technical resume was not created before metric computation")
    for section in ("source_sha256", "input_sha256"):
        for name, expected in freeze[section].items():
            candidate = Path(name)
            candidate = candidate if candidate.is_absolute() else project_root / candidate
            if not candidate.is_file() or sha256_file(candidate) != expected:
                raise RuntimeError(f"Post-resume-freeze hash mismatch: {candidate}")
    return freeze


def load_benchmark_holdout(path: str | Path):
    final = read_json(runner.PROTOCOL_PATH)
    return load_locked_holdout(
        path,
        development_path=ROOT / final["data"]["development_prepared"],
    )


def runtime_protocol(final: dict[str, object]) -> dict[str, object]:
    protocol = read_json(ROOT / "journal_protocol/third_comparison_protocol.json")
    for name in (
        "lookback",
        "horizon",
        "reported_horizons",
        "test_stride",
        "fill_limit",
        "corruption_seed",
    ):
        protocol[name] = final["forecast"][name]
    return protocol


def prepare_benchmark_holdout(*args, **kwargs):
    prepared, report, availability = prepare_locked_holdout(*args, **kwargs)
    protocol = read_json(args[0] if args else kwargs["protocol_path"])
    offset = float(protocol["data"]["local_standard_utc_offset_hours"])
    prepared.calendar = _calendar_features(prepared.timestamps + pd.to_timedelta(offset, unit="h"))
    prepared.name = str(protocol["data"]["prepared_dataset_name"])
    report["calendar_clock"] = {
        "kind": "fixed local standard time",
        "utc_offset_hours": offset,
    }
    return prepared, report, availability


def calibration_bundle(directory: Path, pollutants: int) -> dict[str, np.ndarray]:
    path = directory / "calibration.npz"
    with np.load(path, allow_pickle=False) as archive:
        calibration = {name: archive[name].copy() for name in archive.files}
    for duration in (6, 24):
        key = f"single_station_trailing_{duration}h__combined_factors"
        if key not in calibration:
            raise ValueError(f"Missing frozen calibration array {key}: {path}")
        factors = calibration[key]
        if (
            factors.ndim != 2
            or factors.shape[1] != pollutants
            or not np.all(np.isfinite(factors) & (factors > 0))
        ):
            raise ValueError(f"Invalid frozen calibration array {key}: {path}")
    return calibration


def calibrated_prediction(
    prediction: dict[str, np.ndarray],
    calibration: dict[str, np.ndarray],
    condition: str,
) -> dict[str, np.ndarray]:
    group = development.calibration_group(condition)
    key = f"{group}__combined_factors"
    if key not in calibration:
        raise KeyError(f"No frozen calibration for condition group {group}")
    return apply_horizon_sigma_calibration(prediction, calibration[key])


def evaluate_with_frozen_calibration(
    predictor,
    data,
    protocol: dict[str, object],
    calibration: dict[str, np.ndarray],
) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, np.ndarray]]:
    metrics: dict[str, object] = {}
    trailing_parts = {
        hours: {key: [] for key in ("mu", "sigma", "target", "mask")} for hours in (6, 24)
    }
    conditions = development.evaluation_conditions(
        len(data.stations), smoke=False, protocol=protocol
    )
    clean = predictor("clean")
    for condition, stations in conditions:
        runner.log(f"TEST condition={condition}")
        prediction = clean if condition == "clean" else predictor(condition)
        metrics[condition] = {
            "uncalibrated": outage_metrics(
                prediction,
                data,
                list(protocol["reported_horizons"]),
                station=stations,
            ),
            "frozen_development_calibration": outage_metrics(
                calibrated_prediction(prediction, calibration, condition),
                data,
                list(protocol["reported_horizons"]),
                station=stations,
            ),
        }
        if isinstance(stations, int) and condition.startswith("station_trailing_"):
            hours = int(condition.split("_")[2].removesuffix("h"))
            for key in trailing_parts[hours]:
                trailing_parts[hours][key].append(prediction[key][:, :, stations, :])
    bundle: dict[str, np.ndarray] = {
        "origins": clean["origin"],
        "horizons": np.asarray(protocol["reported_horizons"], dtype=np.int16),
    }
    for hours, parts in trailing_parts.items():
        for key, values in parts.items():
            bundle[f"h{hours}_{key}"] = np.stack(values, axis=2)
    return metrics, clean, bundle


def evaluate_natural_comissingness_test(
    predict_windows,
    data,
    protocol: dict[str, object],
    calibration: dict[str, np.ndarray],
) -> dict[str, object]:
    hourly_protocol = {**protocol, "test_stride": 1}
    hourly = OutageWindows(data, hourly_protocol, "test")
    output: dict[str, object] = {}
    for hours in protocol["natural_comissingness"]["durations_hours"]:
        selected = natural_comissingness_origins(data, hourly.origins, int(hours))
        station_results: dict[str, object] = {}
        for station, station_name in enumerate(data.stations):
            positions = np.flatnonzero(selected[:, station])
            if not len(positions):
                station_results[station_name] = {"forecast_origins": 0, "metrics": None}
                continue
            windows = copy.copy(hourly)
            windows.origins = hourly.origins[positions]
            prediction = predict_windows(windows)
            station_results[station_name] = {
                "forecast_origins": len(positions),
                "origin_timestamps_utc": [
                    str(data.timestamps[origin]) for origin in windows.origins
                ],
                "uncalibrated": outage_metrics(
                    prediction,
                    data,
                    list(protocol["reported_horizons"]),
                    station=station,
                ),
                "frozen_development_calibration": outage_metrics(
                    calibrated_prediction(prediction, calibration, "clean"),
                    data,
                    list(protocol["reported_horizons"]),
                    station=station,
                ),
            }
        output[f"trailing_{hours}h"] = station_results

    per_pollutant: dict[str, object] = {}
    for hours in protocol["natural_comissingness"]["durations_hours"]:
        selected = natural_pollutant_gap_origins(data, hourly.origins, int(hours))
        duration_results: dict[str, object] = {}
        for pollutant, pollutant_name in enumerate(data.pollutants):
            union_positions = np.flatnonzero(selected[:, :, pollutant].any(axis=1))
            if len(union_positions):
                windows = copy.copy(hourly)
                windows.origins = hourly.origins[union_positions]
                union_prediction = predict_windows(windows)
            station_results = {}
            for station, station_name in enumerate(data.stations):
                local_positions = np.flatnonzero(selected[union_positions, station, pollutant])
                if not len(local_positions):
                    station_results[station_name] = {
                        "forecast_origins": 0,
                        "metrics": None,
                    }
                    continue
                prediction = {
                    key: value[local_positions] for key, value in union_prediction.items()
                }
                mask = prediction["mask"].copy()
                mask[..., np.arange(data.n_pollutants) != pollutant] = False
                prediction["mask"] = mask
                station_results[station_name] = {
                    "forecast_origins": len(local_positions),
                    "origin_timestamps_utc": [
                        str(data.timestamps[origin]) for origin in windows.origins[local_positions]
                    ],
                    "uncalibrated": outage_metrics(
                        prediction,
                        data,
                        list(protocol["reported_horizons"]),
                        station=station,
                    ),
                    "frozen_development_calibration": outage_metrics(
                        calibrated_prediction(prediction, calibration, "clean"),
                        data,
                        list(protocol["reported_horizons"]),
                        station=station,
                    ),
                }
            duration_results[pollutant_name] = station_results
        per_pollutant[f"trailing_{hours}h"] = duration_results
    output["per_pollutant_gaps"] = per_pollutant
    return output


def save_test_artifacts(
    directory: Path,
    model_name: str,
    seed: int | None,
    metrics: dict[str, object],
    clean: dict[str, np.ndarray],
    trailing: dict[str, np.ndarray],
    calibration: dict[str, np.ndarray],
    seconds: float,
    parameters: int,
    peak_gpu_memory: int | None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "test_metrics.json": directory / "test_metrics.json",
        "clean_test_predictions.npz": directory / "clean_test_predictions.npz",
        "trailing_test_predictions.npz": directory / "trailing_test_predictions.npz",
        "frozen_calibration.npz": directory / "frozen_calibration.npz",
    }
    write_json(paths["test_metrics.json"], metrics)
    np.savez_compressed(paths["clean_test_predictions.npz"], **clean)
    np.savez_compressed(paths["trailing_test_predictions.npz"], **trailing)
    np.savez_compressed(paths["frozen_calibration.npz"], **calibration)
    write_json(
        directory / "complete.json",
        {
            "state": "complete",
            "model": model_name,
            "seed": seed,
            "test_only": True,
            "test_labels_used_for_fitting": False,
            "inference_and_evaluation_seconds": seconds,
            "parameters": parameters,
            "peak_gpu_memory_bytes": peak_gpu_memory,
            "artifact_hashes": {name: sha256_file(path) for name, path in paths.items()},
        },
    )


def learned_source(final: dict[str, object], model: str, seed: int) -> Path:
    roots = final["checkpoint_roots"]
    root = roots["refinement"] if model in final["refinement_models"] else roots["base"]
    return runner.project_path(root) / model / f"seed_{seed}"


def prepare_once(final: dict[str, object], freeze: dict[str, object]) -> Path:
    data_protocol = final["data"]
    prepared_path = runner.project_path(data_protocol["prepared_holdout"])
    report_path = runner.project_path(data_protocol["preparation_report"])
    availability_path = runner.project_path(data_protocol["availability_table"])
    if prepared_path.is_file() or report_path.is_file():
        if not (prepared_path.is_file() and report_path.is_file()):
            raise RuntimeError("Partial holdout preparation artifacts require manual audit")
        report = read_json(report_path)
        if report.get("status") != "PASS_HOLDOUT_PREPARED_AFTER_FINAL_FREEZE":
            raise RuntimeError("Existing holdout preparation report is invalid")
        if sha256_file(prepared_path) != report["prepared_sha256"]:
            raise RuntimeError("Prepared holdout hash differs from its report")
        return prepared_path

    output_root = runner.project_path(final["output"])
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / "test_access_log.json",
        {
            "access_started_utc": utc_now(),
            "execution_freeze_sha256": sha256_file(runner.FREEZE_PATH),
            "protocol_sha256": sha256_file(runner.PROTOCOL_PATH),
            "holdout_manifest_sha256": data_protocol["sealed_manifest_sha256"],
            "freeze_authorized": freeze["holdout_evaluation_authorized"],
            "purpose": "one-time Houston 2025 multi-model outage benchmark",
        },
    )
    runner.log("FINAL FREEZE VERIFIED; opening Houston 2025 once for the benchmark")
    prepared, report, availability = prepare_benchmark_holdout(
        runner.PROTOCOL_PATH,
        runner.PREPROCESSING_PATH,
        runner.project_path(data_protocol["sealed_raw_root"]),
        runner.project_path(data_protocol["development_prepared"]),
    )
    prepared.save(prepared_path)
    availability_path.parent.mkdir(parents=True, exist_ok=True)
    availability.to_csv(availability_path, index=False)
    report.update(
        completed_utc=utc_now(),
        execution_freeze_sha256=sha256_file(runner.FREEZE_PATH),
        prepared_file=str(prepared_path.relative_to(ROOT)),
        prepared_sha256=sha256_file(prepared_path),
        availability_file=str(availability_path.relative_to(ROOT)),
        availability_sha256=sha256_file(availability_path),
    )
    write_json(report_path, report)
    return prepared_path


runner.runtime_protocol = runtime_protocol
runner.verify_execution_freeze = verify_technical_resume
runner.load_holdout_outage_data = load_benchmark_holdout
runner.prepare_epa_holdout = prepare_benchmark_holdout
runner.calibration_factors = calibration_bundle
runner.evaluate_with_frozen_calibration = evaluate_with_frozen_calibration
runner.evaluate_natural_comissingness_test = evaluate_natural_comissingness_test
runner.save_test_artifacts = save_test_artifacts
runner.learned_source = learned_source
runner.prepare_once = prepare_once


if __name__ == "__main__":
    started = time.perf_counter()
    runner.main()
    runner.log(f"benchmark wrapper elapsed_seconds={time.perf_counter() - started:.1f}")
