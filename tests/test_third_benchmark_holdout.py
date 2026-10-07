from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aqriskformer import epa_aqs_holdout
from scripts.analyze_third_benchmark_holdout import all_models, run_seed
from scripts.run_third_benchmark_holdout import (
    calibrated_prediction,
    calibration_bundle,
)


def test_calibration_bundle_accepts_frozen_condition_horizon_schema(tmp_path: Path) -> None:
    factors = np.ones((5, 5), dtype=np.float32)
    np.savez_compressed(
        tmp_path / "calibration.npz",
        single_station_trailing_6h__combined_factors=factors,
        single_station_trailing_24h__combined_factors=factors * 1.1,
    )
    loaded = calibration_bundle(tmp_path, pollutants=5)
    assert loaded["single_station_trailing_6h__combined_factors"].shape == (5, 5)


def test_calibration_bundle_rejects_nonpositive_factors(tmp_path: Path) -> None:
    factors = np.ones((5, 5), dtype=np.float32)
    factors[0, 0] = 0
    np.savez_compressed(
        tmp_path / "calibration.npz",
        single_station_trailing_6h__combined_factors=factors,
        single_station_trailing_24h__combined_factors=np.ones((5, 5)),
    )
    with pytest.raises(ValueError, match="Invalid frozen calibration"):
        calibration_bundle(tmp_path, pollutants=5)


def test_calibrated_prediction_uses_condition_specific_horizon_factors() -> None:
    prediction = {
        "mu": np.zeros((2, 5, 3, 2), dtype=np.float32),
        "sigma": np.ones((2, 5, 3, 2), dtype=np.float32),
        "target": np.zeros((2, 5, 3, 2), dtype=np.float32),
        "mask": np.ones((2, 5, 3, 2), dtype=bool),
    }
    factors = np.arange(1, 11, dtype=np.float32).reshape(5, 2)
    result = calibrated_prediction(
        prediction,
        {"single_station_trailing_6h__combined_factors": factors},
        "station_trailing_6h_2",
    )
    np.testing.assert_allclose(result["sigma"][0, :, 0, :], factors)
    np.testing.assert_array_equal(result["mu"], prediction["mu"])


def test_benchmark_model_inventory_and_deterministic_seed_handling() -> None:
    final = {
        "reference_model": "reference",
        "learned_comparators": ["learned"],
        "deterministic_comparators": ["persistence"],
    }
    assert all_models(final) == ["reference", "learned", "persistence"]
    assert run_seed(final, "learned", 42) == 42
    assert run_seed(final, "persistence", 42) is None


def test_holdout_loader_carries_frozen_training_event_rates(monkeypatch) -> None:
    hours, stations, pollutants = 4, 2, 3
    event_rates = np.full((stations, pollutants, 3), 0.2, dtype=np.float32)
    prepared = SimpleNamespace(
        split_bounds={"test": (0, hours - 1)},
        observed_mask=np.ones((hours, stations, pollutants), dtype=bool),
        values=np.zeros((hours, stations, pollutants), dtype=np.float32),
        calendar=np.zeros((hours, 4), dtype=np.float32),
        center=np.zeros((stations, pollutants), dtype=np.float32),
        scale=np.ones((stations, pollutants), dtype=np.float32),
        risk_thresholds=np.ones((pollutants, 3), dtype=np.float32),
        training_event_rates=event_rates,
        mase_scale24=np.ones((stations, pollutants), dtype=np.float32),
        timestamps_ns=np.arange(hours, dtype=np.int64),
        stations=["a", "b"],
        pollutants=["p1", "p2", "p3"],
        train_correlation_graph=np.eye(stations, dtype=np.float32),
    )
    monkeypatch.setattr(
        epa_aqs_holdout.PreparedAirQuality,
        "load",
        classmethod(lambda cls, path: prepared),
    )
    loaded = epa_aqs_holdout.load_holdout_outage_data(
        "unused.npz", development_path="development.npz"
    )
    np.testing.assert_array_equal(loaded.event_rates, event_rates)
