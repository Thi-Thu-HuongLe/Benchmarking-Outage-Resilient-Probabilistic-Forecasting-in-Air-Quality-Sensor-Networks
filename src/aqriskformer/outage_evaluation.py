"""Matched probabilistic loss and metrics for outage-resilience experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from scipy.special import ndtr
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

from .outage_data import OutageData


def scaled_thresholds(data: OutageData) -> np.ndarray:
    pollutants = data.n_pollutants
    return (data.thresholds[None] - data.center[:, :pollutants, None]) / data.scale[
        :, :pollutants, None
    ]


def gaussian_exceedance(
    mu: torch.Tensor, sigma: torch.Tensor, thresholds: torch.Tensor
) -> torch.Tensor:
    z = (thresholds - mu[..., None]) / sigma[..., None]
    return 0.5 * torch.erfc(z / math.sqrt(2))


def matched_gaussian_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
    thresholds: torch.Tensor,
    weights: dict[str, object],
) -> torch.Tensor:
    mu, sigma = output["mu"], output["sigma"]
    safe = torch.where(mask, target, mu.detach())
    nll = sigma.log() + 0.5 * ((safe - mu) / sigma).square() + 0.5 * math.log(2 * math.pi)
    huber = F.huber_loss(mu, safe, reduction="none")
    probability = gaussian_exceedance(mu, sigma, thresholds)
    events = (safe[..., None] > thresholds).float()
    quantile_weights = mu.new_tensor(weights["quantile_weights"])
    brier = ((probability - events).square() * quantile_weights).sum(-1)
    brier = brier / quantile_weights.sum()
    total = (
        float(weights["nll"]) * nll
        + float(weights["huber"]) * huber
        + float(weights["brier"]) * brier
    )
    return torch.where(mask, total, 0.0).sum() / mask.sum().clamp_min(1)


def fit_multiplicative_sigma_calibration(
    prediction: dict[str, np.ndarray], bounds: tuple[float, float]
) -> np.ndarray:
    """Fit one Gaussian scale multiplier per pollutant by masked NLL."""
    lower, upper = bounds
    if not 0 < lower <= upper:
        raise ValueError("Calibration bounds must be positive and ordered")
    mu = np.asarray(prediction["mu"], dtype=float)
    sigma = np.asarray(prediction["sigma"], dtype=float)
    target = np.asarray(prediction["target"], dtype=float)
    mask = np.asarray(prediction["mask"], dtype=bool)
    if not (mu.shape == sigma.shape == target.shape == mask.shape):
        raise ValueError("Calibration arrays have different shapes")
    standardized_squared = ((target - mu) / sigma) ** 2
    factors = np.empty(mu.shape[-1], dtype=np.float32)
    for pollutant in range(mu.shape[-1]):
        valid = mask[..., pollutant]
        if not valid.any():
            raise ValueError(f"No calibration targets for pollutant index {pollutant}")
        optimum = math.sqrt(float(standardized_squared[..., pollutant][valid].mean()))
        factors[pollutant] = np.clip(optimum, lower, upper)
    return factors


def apply_sigma_calibration(
    prediction: dict[str, np.ndarray], factors: np.ndarray
) -> dict[str, np.ndarray]:
    factors = np.asarray(factors, dtype=np.float32)
    if factors.shape != (prediction["sigma"].shape[-1],) or not np.all(factors > 0):
        raise ValueError("Invalid pollutant calibration factors")
    return {
        **prediction,
        "sigma": np.asarray(prediction["sigma"], dtype=np.float32) * factors[None, None, None, :],
    }


def fit_horizon_conformal_sigma_calibration(
    predictions: list[dict[str, np.ndarray]],
    bounds: tuple[float, float],
    *,
    minimum_samples: int = 100,
    shrinkage_samples: float = 200.0,
) -> dict[str, np.ndarray]:
    """Fit NLL then central-80% conformal scales by horizon and pollutant.

    Input predictions must already mask non-target stations for their outage
    condition. Sparse cells shrink toward a pollutant-wide NLL factor; cells
    below ``minimum_samples`` use that fallback before conformalization.
    """
    if not predictions:
        raise ValueError("At least one calibration prediction is required")
    lower, upper = bounds
    if not 0 < lower <= upper:
        raise ValueError("Calibration bounds must be positive and ordered")
    arrays = {
        key: np.concatenate([np.asarray(item[key]) for item in predictions], axis=0)
        for key in ("mu", "sigma", "target", "mask")
    }
    mu, sigma, target = arrays["mu"], arrays["sigma"], arrays["target"]
    mask = arrays["mask"].astype(bool)
    if not (mu.shape == sigma.shape == target.shape == mask.shape):
        raise ValueError("Calibration arrays have different shapes")
    if not (np.isfinite(mu).all() and np.isfinite(sigma).all() and (sigma > 0).all()):
        raise ValueError("Calibration predictions must be finite with positive scales")
    horizons, pollutants = mu.shape[1], mu.shape[-1]
    squared_z = ((target - mu) / sigma) ** 2
    absolute_z = np.abs(target - mu) / sigma
    nll = np.empty((horizons, pollutants), dtype=np.float32)
    conformal = np.empty_like(nll)
    count = np.zeros((horizons, pollutants), dtype=np.int64)
    global_nll = np.empty(pollutants, dtype=np.float32)
    for pollutant in range(pollutants):
        valid = mask[..., pollutant]
        if not valid.any():
            raise ValueError(f"No calibration targets for pollutant index {pollutant}")
        global_nll[pollutant] = np.clip(
            np.sqrt(squared_z[..., pollutant][valid].mean()), lower, upper
        )
    normal_central_80 = 1.2815515655446004
    for horizon in range(horizons):
        for pollutant in range(pollutants):
            valid = mask[:, horizon, :, pollutant]
            count[horizon, pollutant] = int(valid.sum())
            if count[horizon, pollutant] >= minimum_samples:
                local = float(np.sqrt(squared_z[:, horizon, :, pollutant][valid].mean()))
                weight = count[horizon, pollutant] / (
                    count[horizon, pollutant] + shrinkage_samples
                )
                nll[horizon, pollutant] = np.clip(
                    weight * local + (1.0 - weight) * global_nll[pollutant], lower, upper
                )
            else:
                nll[horizon, pollutant] = global_nll[pollutant]
            scores = absolute_z[:, horizon, :, pollutant][valid] / nll[horizon, pollutant]
            if len(scores):
                rank = min(len(scores) - 1, int(np.ceil(0.8 * (len(scores) + 1))) - 1)
                quantile = float(np.partition(scores, rank)[rank])
                conformal[horizon, pollutant] = np.clip(
                    quantile / normal_central_80, lower, upper
                )
            else:
                conformal[horizon, pollutant] = 1.0
    return {
        "nll_factors": nll,
        "conformal_factors": conformal,
        "combined_factors": np.clip(nll * conformal, lower, upper),
        "counts": count,
    }


def apply_horizon_sigma_calibration(
    prediction: dict[str, np.ndarray], factors: np.ndarray
) -> dict[str, np.ndarray]:
    factors = np.asarray(factors, dtype=np.float32)
    expected = (prediction["sigma"].shape[1], prediction["sigma"].shape[-1])
    if factors.shape != expected or not np.all(np.isfinite(factors) & (factors > 0)):
        raise ValueError("Invalid horizon-by-pollutant calibration factors")
    return {
        **prediction,
        "sigma": np.asarray(prediction["sigma"], dtype=np.float32)
        * factors[None, :, None, :],
    }


def outage_metrics(
    prediction: dict[str, np.ndarray],
    data: OutageData,
    horizons: list[int],
    *,
    station: int | Sequence[int] | None = None,
) -> dict[str, object]:
    mu, sigma = prediction["mu"], prediction["sigma"]
    target = prediction["target"]
    mask = prediction["mask"].astype(bool)
    if not (np.isfinite(mu).all() and np.isfinite(sigma).all() and (sigma > 0).all()):
        raise ValueError("Nonfinite predictions or nonpositive Gaussian scale")
    if station is not None:
        selected_stations = [station] if isinstance(station, int) else list(station)
        if not selected_stations or len(selected_stations) != len(set(selected_stations)):
            raise ValueError("Selected outage stations must be unique and nonempty")
        if min(selected_stations) < 0 or max(selected_stations) >= mask.shape[2]:
            raise ValueError("Selected outage station is outside the prediction tensor")
        mask = mask.copy()
        mask[:, :, ~np.isin(np.arange(mask.shape[2]), selected_stations)] = False
    pollutants = data.n_pollutants
    center = data.center[:, :pollutants]
    scale = data.scale[:, :pollutants]
    threshold = scaled_thresholds(data)
    probability = ndtr((mu[..., None] - threshold) / sigma[..., None])
    events = target[..., None] > threshold
    climatology = data.event_rates[None, None]
    count = mask.sum(0)

    def mean_time(values: np.ndarray) -> np.ndarray:
        return np.divide(
            np.where(mask, values, 0).sum(0),
            count,
            out=np.full(count.shape, np.nan, dtype=float),
            where=count > 0,
        )

    error = target - mu
    absolute = np.abs(error)
    z = error / sigma
    crps = sigma * (
        z * (2 * ndtr(z) - 1)
        + 2 * np.exp(-(z**2) / 2) / math.sqrt(2 * math.pi)
        - 1 / math.sqrt(math.pi)
    )
    lo = mu - 1.2815515655446004 * sigma
    hi = mu + 1.2815515655446004 * sigma
    interval_score = hi - lo + 10 * np.maximum(lo - target, 0) + 10 * np.maximum(target - hi, 0)
    cells = {
        "scaled_mae": mean_time(absolute),
        "scaled_rmse": np.sqrt(mean_time(error**2)),
        "mase": mean_time(absolute) * scale / data.mase,
        "q95_brier": mean_time((probability[..., 2] - events[..., 2]) ** 2),
        "q95_brier_reference": mean_time(
            (climatology[..., 2] - events[..., 2]) ** 2
        ),
        "q95_prevalence": mean_time(events[..., 2]),
        "picp80": mean_time((target >= lo) & (target <= hi)),
        "crps_mase_scaled": mean_time(crps) * scale / data.mase,
        "interval_score80_mase_scaled": mean_time(interval_score) * scale / data.mase,
        "width80_mase_scaled": mean_time(hi - lo) * scale / data.mase,
        "negative_mass": mean_time(ndtr((-center / scale - mu) / sigma)),
    }
    cells["q95_brier_skill"] = 1.0 - np.divide(
        cells["q95_brier"],
        cells["q95_brier_reference"],
        out=np.full_like(cells["q95_brier"], np.nan),
        where=cells["q95_brier_reference"] > 0,
    )
    average_precision = np.full(count.shape, np.nan)
    for horizon_index, station_index, pollutant_index in np.ndindex(count.shape):
        valid = mask[:, horizon_index, station_index, pollutant_index]
        labels = events[valid, horizon_index, station_index, pollutant_index, 2]
        if labels.any() and not labels.all():
            average_precision[horizon_index, station_index, pollutant_index] = (
                average_precision_score(
                    labels,
                    probability[valid, horizon_index, station_index, pollutant_index, 2],
                )
            )
    cells["q95_ap"] = average_precision

    def aggregate(index: object = None) -> dict[str, float | int | None]:
        selected = {key: value if index is None else value[index] for key, value in cells.items()}
        result: dict[str, float | int | None] = {
            key: float(np.nanmean(value)) if np.isfinite(value).any() else None
            for key, value in selected.items()
        }
        selected_count = count if index is None else count[index]
        result["observed_targets"] = int(np.asarray(selected_count).sum())
        result["valid_metric_cells"] = int((np.asarray(selected_count) > 0).sum())
        return result

    summary = aggregate()
    summary["selection_score"] = (
        None
        if summary["q95_ap"] is None
        else float(summary["scaled_mae"])
        + 2 * float(summary["q95_brier"])
        - 0.25 * float(summary["q95_ap"])
    )
    common_indices = [data.pollutants.index(name) for name in ("PM2.5", "NO2", "O3")]
    partial_indices = [data.pollutants.index(name) for name in ("CO", "SO2")]
    return {
        "summary": summary,
        "common_core": aggregate((slice(None), slice(None), common_indices)),
        "partial_pollutants": aggregate((slice(None), slice(None), partial_indices)),
        "by_horizon": {str(hour): aggregate(index) for index, hour in enumerate(horizons)},
        "by_pollutant": {
            pollutant: aggregate((slice(None), slice(None), index))
            for index, pollutant in enumerate(data.pollutants)
        },
        "by_station": {
            name: aggregate((slice(None), index, slice(None)))
            for index, name in enumerate(data.stations)
            if station is None or index in selected_stations
        },
    }
