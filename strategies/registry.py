"""Strategy registry. A config entry has ``class`` (base logic), ``params`` and optional ``filters``.

Strategy ids are unique; changing params or filters must bump ``version`` (governance hashes the config).
Only the validated ORB family is active; retired strategies were deleted (recoverable from git history (commit 8a0cab9 and earlier)).
"""
from __future__ import annotations

from strategies.base import Strategy
from strategies.filters import FilteredStrategy
from strategies.orb.orb_strategy import ORBStrategy

CLASSES = {"orb": ORBStrategy}


def build_strategy(strategy_id: str, cfg: dict, overrides: dict | None = None,
                   filter_overrides: dict | None = None) -> Strategy:
    params = dict(cfg.get("params", {}))
    params.update(overrides or {})
    cls = CLASSES[cfg["class"]]
    base = cls(params, version=cfg.get("version", "0"))
    filters = dict(cfg.get("filters") or {})
    filters.update(filter_overrides or {})
    if filters:
        return FilteredStrategy(base, filters, strategy_id, cfg.get("version", "0"))
    base.strategy_id = strategy_id
    return base
