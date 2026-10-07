"""Minimal run-integrity helper required by the Houston development-freeze script.

This module intentionally contains only the generic artifact verification routine imported
by ``freeze_third_benchmark_development.py``. It does not provide the earlier study's
candidate-selection command or depend on that study's local results.
"""

from __future__ import annotations

from pathlib import Path

from aqriskformer.utils import read_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def verify_hashes(base: Path, values: dict[str, str]) -> None:
    """Raise if any file listed in a frozen run marker is missing or has changed."""
    for name, expected in values.items():
        path = base / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen hash mismatch: {path}")


def verify_run(directory: Path, model: str, seed: int | None) -> dict[str, str]:
    """Verify a completed run and return hashes for its marker and frozen artifacts."""
    marker_path = directory / "complete.json"
    marker = read_json(marker_path)
    if marker.get("state") != "complete" or marker.get("model") != model:
        raise RuntimeError(f"Incomplete or mismatched run: {directory}")
    if seed is not None and int(marker.get("seed", -1)) != seed:
        raise RuntimeError(f"Seed mismatch: {directory}")
    verify_hashes(directory, marker["artifact_hashes"])

    required = ["calibration.npz"]
    required.append("best.pt" if seed is not None else "training_residual_sigma.npz")
    for name in required:
        if name not in marker["artifact_hashes"]:
            raise RuntimeError(f"Run marker does not freeze {name}: {directory}")

    return {
        str(marker_path.relative_to(ROOT)).replace("\\", "/"): sha256_file(marker_path),
        **{
            str((directory / name).relative_to(ROOT)).replace("\\", "/"): digest
            for name, digest in marker["artifact_hashes"].items()
        },
    }
