from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Load YAML and attach absolute project/config paths without mutating YAML."""
    config_path = Path(path).expanduser().resolve()
    if config_path in _seen:
        raise ValueError(f"Cyclic config inheritance: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {config_path}")
    config = deepcopy(config)
    parent = config.pop("extends", None)
    if parent:
        inherited = load_config(config_path.parent / parent, _seen | {config_path})
        inherited.pop("_runtime", None)
        config = _merge_config(inherited, config)
    project_root = _find_project_root(config_path.parent)
    config["_runtime"] = {
        "config_path": str(config_path),
        "project_root": str(project_root),
    }
    return config


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def config_fingerprint(config: dict[str, Any]) -> str:
    """Hash effective settings, including inherited values, not just leaf YAML."""
    payload = {k: v for k, v in config.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / "Air Quality Data").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate project root above {start}")


def project_path(config: dict[str, Any], relative: str | Path) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    return Path(config["_runtime"]["project_root"]) / path


def dataset_config(config: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        return config["datasets"][name]
    except KeyError as exc:
        available = ", ".join(sorted(config.get("datasets", {})))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}") from exc
