"""Configuration loading: package defaults <- user YAML <- command-line overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
# the project folder (configs/, data/, models/, results/ ...); BEEMON_PROJECT can point elsewhere (tests)
PROJECT_DIR = Path(os.environ.get("BEEMON_PROJECT", PACKAGE_DIR.parent)).expanduser().resolve()
DEFAULT_CONFIG = PACKAGE_DIR / "default_config.yaml"
USER_CONFIG = PROJECT_DIR / "configs" / "pipeline.yaml"


def read_yaml(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(data: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of ``base`` with ``override`` merged in recursively."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_overrides(cfg: dict, items: list[str] | None) -> dict:
    """Apply ``section.key=value`` strings (values parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    for item in items or []:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        parts = key.strip().split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot set {key}: {part} is not a section")
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """Load the default config, then the user file (if any), then overrides."""
    cfg = read_yaml(DEFAULT_CONFIG)
    user_path = Path(path) if path else (USER_CONFIG if USER_CONFIG.is_file() else None)
    if user_path is not None:
        if not user_path.is_file():
            raise FileNotFoundError(f"Config file not found: {user_path}")
        cfg = deep_merge(cfg, read_yaml(user_path) or {})
    return apply_overrides(cfg, overrides)


def resolve_path(value: str | Path | None) -> Path | None:
    """Resolve a path relative to the current directory, falling back to the project directory."""
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path
    candidate = PROJECT_DIR / path
    return candidate if candidate.exists() else path
