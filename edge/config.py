"""
Configuration loader.

Reads config/system_config.yaml into a dotted-access object so the rest of the
code can say `cfg.scoring.weights.drowsiness` instead of chained dict lookups.
Command-line overrides are applied with the same dotted path
(e.g. --set scoring.thresholds.high=55).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "system_config.yaml"


class ConfigNode(dict):
    """A dict that also supports attribute access, recursively."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(f"No configuration key '{item}'") from exc
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @classmethod
    def wrap(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return cls({k: cls.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls.wrap(v) for v in obj]
        return obj

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: Any = self
        for part in parts[:-1]:
            node = node.setdefault(part, ConfigNode())
        node[parts[-1]] = value


def _coerce(text: str) -> Any:
    """Turn a CLI string into the most sensible Python type."""
    lowered = text.strip().lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", ""):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def load_config(path: str | os.PathLike | None = None,
                overrides: list[str] | None = None) -> ConfigNode:
    """Load the YAML config, apply `key.path=value` overrides, return it."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = ConfigNode.wrap(raw)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' is not of the form key.path=value")
        dotted, _, value = item.partition("=")
        cfg.set_path(dotted.strip(), _coerce(value))

    # Resolve relative paths against the project root so the node can be run
    # from any working directory.
    for dotted in ("server.database",
                   "telemetry.offline_buffer_path"):
        value = cfg.get_path(dotted)
        if isinstance(value, str) and not os.path.isabs(value):
            cfg.set_path(dotted, str(PROJECT_ROOT / value))

    cfg.set_path("_config_path", str(cfg_path))
    return cfg
