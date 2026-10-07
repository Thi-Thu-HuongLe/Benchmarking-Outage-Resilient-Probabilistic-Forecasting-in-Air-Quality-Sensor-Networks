from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import dataset_config, project_path

POLLUTANTS = ("PM2.5", "PM10", "SO2", "NO2", "CO", "O3")
QUANTILES = (0.80, 0.90, 0.95)
WIND_DEGREES = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}


@dataclass
class RawAirQuality:
    name: str
    frame: pd.DataFrame
    timestamps: pd.DatetimeIndex
    stations: list[str]
    pollutants: list[str]
    meteorology: list[str]
    station_static: np.ndarray


@dataclass
class PreparedAirQuality:
    name: str
    timestamps_ns: np.ndarray
    stations: list[str]
    pollutants: list[str]
    feature_names: list[str]
    meteorology: list[str]
    values: np.ndarray
    observed_mask: np.ndarray
    time_gaps: np.ndarray
    calendar: np.ndarray
    station_static: np.ndarray
    native_pollutants: np.ndarray
    target_mask: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    risk_thresholds: np.ndarray
    mase_scale24: np.ndarray
    train_correlation_graph: np.ndarray
    split_bounds: dict[str, tuple[int, int]]

    @cached_property
    def training_event_rates(self) -> np.ndarray:
        """Station/pollutant/threshold climatology from observed TRAIN targets."""
        start, end = self.split_bounds["train"]
        y = self.native_pollutants[start : end + 1]
        observed = self.target_mask[start : end + 1] & np.isfinite(y)
        counts = observed.sum(axis=0)[..., None]
        events = (y[..., None] > self.risk_thresholds[None, None]) & observed[..., None]
        return np.divide(
            events.sum(axis=0),
            counts,
            out=np.full(events.shape[1:], np.nan, dtype=float),
            where=counts > 0,
        )

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        return pd.to_datetime(self.timestamps_ns)

    @property
    def n_pollutants(self) -> int:
        return len(self.pollutants)

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "name": self.name,
            "stations": self.stations,
            "pollutants": self.pollutants,
            "feature_names": self.feature_names,
            "meteorology": self.meteorology,
            "split_bounds": self.split_bounds,
        }
        np.savez_compressed(
            target,
            timestamps_ns=self.timestamps_ns.astype("datetime64[ns]"),
            values=self.values.astype(np.float32),
            observed_mask=self.observed_mask.astype(np.uint8),
            time_gaps=self.time_gaps.astype(np.float32),
            calendar=self.calendar.astype(np.float32),
            station_static=self.station_static.astype(np.float32),
            native_pollutants=self.native_pollutants.astype(np.float32),
            target_mask=self.target_mask.astype(np.uint8),
            center=self.center.astype(np.float32),
            scale=self.scale.astype(np.float32),
            risk_thresholds=self.risk_thresholds.astype(np.float32),
            mase_scale24=self.mase_scale24.astype(np.float32),
            train_correlation_graph=self.train_correlation_graph.astype(np.float32),
            metadata=np.array(json.dumps(metadata)),
        )

    @classmethod
    def load(cls, path: str | Path) -> PreparedAirQuality:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            return cls(
                name=metadata["name"],
                timestamps_ns=archive["timestamps_ns"],
                stations=list(metadata["stations"]),
                pollutants=list(metadata["pollutants"]),
                feature_names=list(metadata["feature_names"]),
                meteorology=list(metadata["meteorology"]),
                values=archive["values"],
                observed_mask=archive["observed_mask"].astype(bool),
                time_gaps=archive["time_gaps"],
                calendar=archive["calendar"],
                station_static=archive["station_static"],
                native_pollutants=archive["native_pollutants"],
                target_mask=archive["target_mask"].astype(bool),
                center=archive["center"],
                scale=archive["scale"],
                risk_thresholds=archive["risk_thresholds"],
                mase_scale24=archive["mase_scale24"],
                train_correlation_graph=archive["train_correlation_graph"],
                split_bounds={k: tuple(v) for k, v in metadata["split_bounds"].items()},
            )


def load_raw_dataset(config: dict[str, Any], name: str) -> RawAirQuality:
    if name == "beijing":
        return _load_beijing(config)
    if name == "dhaka":
        return _load_dhaka(config)
    raise KeyError(f"No loader implemented for dataset '{name}'")


def _load_beijing(config: dict[str, Any]) -> RawAirQuality:
    cfg = dataset_config(config, "beijing")
    pattern = str(project_path(config, cfg["file_glob"]))
    files = sorted(glob.glob(pattern))
    if len(files) != int(cfg["stations"]):
        raise ValueError(f"Expected {cfg['stations']} Beijing station files, found {len(files)}")
    frames = [pd.read_csv(path) for path in files]
    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame[["year", "month", "day", "hour"]])
    degrees = frame["wd"].map(WIND_DEGREES)
    radians = np.deg2rad(degrees)
    frame["wd_sin"] = np.sin(radians)
    frame["wd_cos"] = np.cos(radians)
    meteorology = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM", "wd_sin", "wd_cos"]
    stations = sorted(frame["station"].astype(str).unique().tolist())
    timestamps = pd.date_range(frame["timestamp"].min(), frame["timestamp"].max(), freq="h")
    static = np.zeros((len(stations), 2), dtype=np.float32)
    return RawAirQuality(
        name="beijing",
        frame=frame,
        timestamps=timestamps,
        stations=stations,
        pollutants=list(POLLUTANTS),
        meteorology=meteorology,
        station_static=static,
    )


def _load_dhaka(config: dict[str, Any]) -> RawAirQuality:
    cfg = dataset_config(config, "dhaka")
    archive = project_path(config, cfg["archive"])
    frame = pd.read_csv(archive)
    frame["timestamp"] = pd.to_datetime(frame["date"], errors="raise")
    frame["station"] = frame["Station code"].astype(str)
    stations = sorted(frame["station"].unique().tolist())
    timestamps = pd.date_range(frame["timestamp"].min(), frame["timestamp"].max(), freq="h")
    coords = (
        frame.groupby("station", sort=True)[["Latitude", "Longitude"]]
        .median()
        .reindex(stations)
        .to_numpy(np.float32)
    )
    return RawAirQuality(
        name="dhaka",
        frame=frame,
        timestamps=timestamps,
        stations=stations,
        pollutants=list(POLLUTANTS),
        meteorology=[],
        station_static=coords,
    )


def prepare_dataset(config: dict[str, Any], name: str) -> PreparedAirQuality:
    raw = load_raw_dataset(config, name)
    cfg = dataset_config(config, name)
    feature_names = raw.pollutants + raw.meteorology
    index = pd.MultiIndex.from_product(
        [raw.timestamps, raw.stations], names=["timestamp", "station"]
    )
    aligned = (
        raw.frame.assign(station=raw.frame["station"].astype(str))
        .set_index(["timestamp", "station"])[feature_names]
        .groupby(level=[0, 1])
        .mean()
        .reindex(index)
    )
    t_count, n_count, f_count = len(raw.timestamps), len(raw.stations), len(feature_names)
    native = aligned.to_numpy(np.float32).reshape(t_count, n_count, f_count)
    observed = np.isfinite(native)

    train_mask = _timestamp_mask(raw.timestamps, cfg["train"])
    center, scale = _fit_robust_scaler(native, observed, train_mask)
    filled = _causal_fill(
        native, center, limit=int(config["preprocessing"]["input_short_gap_fill_hours"])
    )
    scaled = (filled - center[None, :, :]) / scale[None, :, :]
    gaps = _time_since_observation(observed)
    calendar = _calendar_features(raw.timestamps)
    pollutant_native = native[:, :, : len(raw.pollutants)]
    target_mask = observed[:, :, : len(raw.pollutants)]
    thresholds = _risk_thresholds(pollutant_native, target_mask, train_mask)
    mase_scale = _mase_scale(pollutant_native, target_mask, train_mask, season=24)
    train_correlation_graph = _training_spearman_graph(
        pollutant_native[:, :, 0], target_mask[:, :, 0], train_mask
    )
    split_bounds = {
        split: _split_index_bounds(raw.timestamps, cfg[split])
        for split in ("train", "validation", "test")
    }
    return PreparedAirQuality(
        name=name,
        timestamps_ns=raw.timestamps.to_numpy(dtype="datetime64[ns]"),
        stations=raw.stations,
        pollutants=raw.pollutants,
        feature_names=feature_names,
        meteorology=raw.meteorology,
        values=scaled.astype(np.float32),
        observed_mask=observed,
        time_gaps=gaps,
        calendar=calendar,
        station_static=_normalize_static(raw.station_static),
        native_pollutants=pollutant_native,
        target_mask=target_mask,
        center=center,
        scale=scale,
        risk_thresholds=thresholds,
        mase_scale24=mase_scale,
        train_correlation_graph=train_correlation_graph,
        split_bounds=split_bounds,
    )


def prepared_path(config: dict[str, Any], name: str) -> Path:
    root = config.get("execution", {}).get("prepared_root", "experiment_protocol/prepared")
    return project_path(config, Path(root) / f"{name}.npz")


def _timestamp_mask(index: pd.DatetimeIndex, bounds: list[str]) -> np.ndarray:
    start, end = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
    return np.asarray((index >= start) & (index <= end))


def _split_index_bounds(index: pd.DatetimeIndex, bounds: list[str]) -> tuple[int, int]:
    mask = _timestamp_mask(index, bounds)
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        raise ValueError(f"Split {bounds} does not overlap dataset")
    return int(positions[0]), int(positions[-1])


def _fit_robust_scaler(
    values: np.ndarray, observed: np.ndarray, train_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    train = np.where(observed[train_mask], values[train_mask], np.nan)
    center = np.nanmedian(train, axis=0)
    q75 = np.nanpercentile(train, 75, axis=0)
    q25 = np.nanpercentile(train, 25, axis=0)
    scale = q75 - q25
    global_center = np.nanmedian(train, axis=(0, 1))
    global_scale = np.nanpercentile(train, 75, axis=(0, 1)) - np.nanpercentile(
        train, 25, axis=(0, 1)
    )
    center = np.where(np.isfinite(center), center, global_center[None, :])
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, global_scale[None, :])
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)


def _causal_fill(values: np.ndarray, train_center: np.ndarray, limit: int) -> np.ndarray:
    output = values.copy()
    for station in range(values.shape[1]):
        for feature in range(values.shape[2]):
            series = pd.Series(values[:, station, feature])
            output[:, station, feature] = (
                series.ffill(limit=limit).fillna(float(train_center[station, feature])).to_numpy()
            )
    return output


def _time_since_observation(observed: np.ndarray) -> np.ndarray:
    gaps = np.zeros(observed.shape, dtype=np.float32)
    for station in range(observed.shape[1]):
        for feature in range(observed.shape[2]):
            elapsed = 0.0
            for t in range(observed.shape[0]):
                if observed[t, station, feature]:
                    elapsed = 0.0
                else:
                    elapsed += 1.0
                gaps[t, station, feature] = elapsed
    return gaps


def _calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    hour = index.hour.to_numpy()
    weekday = index.dayofweek.to_numpy()
    dayofyear = index.dayofyear.to_numpy()
    weekend = (weekday >= 5).astype(np.float32)
    holiday = np.zeros(len(index), dtype=np.float32)
    return np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * weekday / 7),
            np.cos(2 * np.pi * weekday / 7),
            np.sin(2 * np.pi * dayofyear / 365.25),
            np.cos(2 * np.pi * dayofyear / 365.25),
            weekend,
            holiday,
        ]
    ).astype(np.float32)


def _risk_thresholds(
    values: np.ndarray, observed: np.ndarray, train_mask: np.ndarray
) -> np.ndarray:
    output = np.empty((values.shape[2], len(QUANTILES)), dtype=np.float32)
    for pollutant in range(values.shape[2]):
        valid = values[train_mask, :, pollutant][observed[train_mask, :, pollutant]]
        output[pollutant] = np.quantile(valid, QUANTILES)
    return output


def _mase_scale(
    values: np.ndarray, observed: np.ndarray, train_mask: np.ndarray, season: int
) -> np.ndarray:
    train_indices = np.flatnonzero(train_mask)
    start, end = train_indices[0], train_indices[-1] + 1
    output = np.ones((values.shape[1], values.shape[2]), dtype=np.float32)
    for station in range(values.shape[1]):
        for pollutant in range(values.shape[2]):
            current = values[start + season : end, station, pollutant]
            lagged = values[start : end - season, station, pollutant]
            valid = (
                observed[start + season : end, station, pollutant]
                & observed[start : end - season, station, pollutant]
            )
            denominator = np.mean(np.abs(current[valid] - lagged[valid])) if valid.any() else np.nan
            output[station, pollutant] = (
                denominator if np.isfinite(denominator) and denominator > 1e-8 else 1.0
            )
    return output


def _normalize_static(static: np.ndarray) -> np.ndarray:
    if static.size == 0 or np.allclose(static, 0):
        return np.zeros_like(static, dtype=np.float32)
    center = np.nanmean(static, axis=0, keepdims=True)
    scale = np.nanstd(static, axis=0, keepdims=True)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return np.nan_to_num((static - center) / scale).astype(np.float32)


def _training_spearman_graph(
    pm25: np.ndarray, observed: np.ndarray, train_mask: np.ndarray
) -> np.ndarray:
    frame = pd.DataFrame(np.where(observed[train_mask], pm25[train_mask], np.nan))
    ranked = frame.rank(axis=0, method="average", na_option="keep")
    correlation = ranked.corr(method="pearson").abs().to_numpy(np.float32)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 0.0)
    return correlation


def audit_dataset(config: dict[str, Any], name: str) -> dict[str, Any]:
    raw = load_raw_dataset(config, name)
    features = raw.pollutants + raw.meteorology
    report: dict[str, Any] = {
        "dataset": name,
        "rows": len(raw.frame),
        "stations": len(raw.stations),
        "station_names": raw.stations,
        "timestamp_min": str(raw.frame["timestamp"].min()),
        "timestamp_max": str(raw.frame["timestamp"].max()),
        "expected_hourly_timestamps": len(raw.timestamps),
        "features": features,
        "missing_fraction": {column: float(raw.frame[column].isna().mean()) for column in features},
        "duplicate_station_timestamps": int(raw.frame.duplicated(["timestamp", "station"]).sum()),
    }
    if name == "dhaka":
        report["pm25_gt_pm10_fraction"] = float((raw.frame["PM2.5"] > raw.frame["PM10"]).mean())
    return report
