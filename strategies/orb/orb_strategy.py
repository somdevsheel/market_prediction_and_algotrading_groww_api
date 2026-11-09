"""Live/backtest Strategy wrapper around orb_core (1-minute candles)."""
from __future__ import annotations

from typing import Optional

from strategies.base import MarketContext, Signal, Strategy
from strategies.orb.orb_core import find_entry, passes_filters


class ORBStrategy(Strategy):
    strategy_id = "orb"
    interval_minutes = 1

    def __init__(self, params: dict, version: str = "1.0.0"):
        super().__init__(params)
        self.version = version

    def evaluate(self, ctx: MarketContext) -> Optional[Signal]:
        p = self.params
        R = self.last_reasons = []
        idx = ctx.index_5m  # candles at the strategy interval (1 minute)
        day = idx[idx["timestamp"].dt.date == ctx.now.date()].reset_index(drop=True)
        e = find_entry(day, p["mode"], int(p["or_minutes"]))
        if e is None:
            R.append("no_setup_yet")
            return None
        if e.entry_idx != len(day) - 1:
            R.append("entry_already_passed")
            return None
        if not passes_filters(e, p.get("orb_filters") or {}):
            R.append(f"orb_filter_failed:range={e.range_pct:.2f}%,t={e.entry_ts:%H:%M},gap={e.gap_minutes:.0f}")
            return None
        R.append("signal")
        far = e.entry_px * (1.5 if e.direction == "BULLISH" else 0.5)  # no fixed target: exits are stop/time
        return Signal(self.strategy_id, e.entry_ts.to_pydatetime(), "NIFTY", e.direction, 0.5,
                      f"ORB {p['mode']} {p['or_minutes']}m: 1m close beyond {'H' if e.direction == 'BULLISH' else 'L'}",
                      e.stop_px, far, e.entry_px, (ctx.regime or {}).get("regime", "unknown"), self.version,
                      [f"orb_{p['mode']}_{p['or_minutes']}m"],
                      {"or_high": e.or_high, "or_low": e.or_low, "range_pct": round(e.range_pct, 3),
                       "gap_minutes": e.gap_minutes, "first_breakout": str(e.first_breakout_ts),
                       "stop_distance": abs(e.entry_px - e.stop_px), "stop_mode": "touch"})
