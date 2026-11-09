"""Strategy registry. A config entry has ``class`` (base logic), ``params`` and optional ``filters``.

Strategy ids are unique; changing params or filters must bump ``version`` (governance hashes the config).
"""
from __future__ import annotations

from strategies.base import Strategy
from strategies.breakout.breakout_retest import BreakoutRetestStrategy
from strategies.breakout.breakout_vwap import BreakoutVWAPStrategy
from strategies.breakout.sr_breakout import SRBreakoutStrategy
from strategies.filters import FilteredStrategy
from strategies.ema_cross.strategy import EMACrossStrategy
from strategies.orb.orb_strategy import ORBStrategy

CLASSES = {"breakout_vwap": BreakoutVWAPStrategy, "sr_breakout": SRBreakoutStrategy,
           "breakout_retest": BreakoutRetestStrategy, "orb": ORBStrategy,
           "ema_cross": EMACrossStrategy}
LEGACY = {"breakout_vwap_v1": "breakout_vwap"}


def build_strategy(strategy_id: str, cfg: dict, overrides: dict | None = None,
                   filter_overrides: dict | None = None) -> Strategy:
    params = dict(cfg.get("params", {}))
    params.update(overrides or {})
    cls = CLASSES[cfg.get("class") or LEGACY.get(strategy_id, strategy_id)]
    base = cls(params, version=cfg.get("version", "0"))
    filters = dict(cfg.get("filters") or {})
    filters.update(filter_overrides or {})
    if filters:
        return FilteredStrategy(base, filters, strategy_id, cfg.get("version", "0"))
    base.strategy_id = strategy_id
    return base
