"""Configuration loading.

All tunables live in ``config/*.yaml``. ``load_config()`` returns a plain nested
dict with sections ``settings``, ``risk``, ``strategies`` and ``instruments``.
An optional ``config/local.yaml`` is deep-merged on top (same section layout)
so personal overrides never require editing the tracked files.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
SECTIONS = ("settings", "risk", "strategies", "instruments")


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(config_dir: Path | str | None = None) -> dict[str, Any]:
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    cfg: dict[str, Any] = {}
    for sec in SECTIONS:
        p = cdir / f"{sec}.yaml"
        cfg[sec] = yaml.safe_load(p.read_text()) if p.exists() else {}
    local = cdir / "local.yaml"
    if local.exists():
        cfg = deep_merge(cfg, yaml.safe_load(local.read_text()) or {})
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    """Resolve a ``settings.paths`` entry relative to the project root."""
    p = Path(cfg["settings"]["paths"][key])
    return p if p.is_absolute() else PROJECT_ROOT / p


def active_strategy_config(cfg: dict) -> tuple[str, dict]:
    sid = cfg["strategies"]["active_strategy"]
    return sid, cfg["strategies"]["strategies"][sid]
