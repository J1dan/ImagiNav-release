"""Shared config helpers for video quality evaluation scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if not config_path:
        return {}
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def resolve_path(value: Optional[str | Path], base_dir: Path) -> Optional[Path]:
    """Resolve a potentially relative path *deterministically* anchored to the
    `video_quality_eval` package root.

    Behavior (single deterministic resolution):
      1. If the path is absolute (or contains '~'), expand and return it.
      2. Otherwise, resolve relative to the `video_quality_eval` package root and return that path (may not exist).

    This enforces a single deterministic interpretation for all relative paths
    used by the evaluation scripts.
    """
    if value is None:
        return None
    # Expand user (~) first
    p = Path(value).expanduser()
    if p.is_absolute():
        return p

    # Deterministic anchor: the video_quality_eval package root
    package_root = Path(__file__).resolve().parents[1]

    # Single deterministic resolution: always resolve relative to package root
    return (package_root / p).resolve()


def apply_imaginav_overrides(cfg: Any, overrides: Dict[str, Any]) -> None:
    """Apply config overrides to ImagiNavConfig sections.

    Expects structure:
      imaginav_overrides:
        imagination:
          key: value
        navigation:
          key: value
        logging:
          key: value
        reasoner:
          key: value
    """
    for section in ["imagination", "navigation", "logging", "reasoner"]:
        if section in overrides and hasattr(cfg, section):
            section_obj = getattr(cfg, section)
            for key, value in overrides[section].items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
