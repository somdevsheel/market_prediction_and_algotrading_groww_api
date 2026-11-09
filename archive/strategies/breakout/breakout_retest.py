"""Strategy C — ``breakout_retest_v1``: opening-range breakout, then a retest that holds, then confirmation.

Recomputed from visible candles on every call (stateless, reproducible):
1. first candle b closing beyond OR level ± buffer;
2. a later candle r whose low (bullish) touches level + tolerance while closing above the level, with no close
   back through level − tolerance in between (failed breakout cancels the setup);
3. the current candle closes above r's high (bullish) within ``max_bars_after_breakout`` -> signal (first time only).
Invalidation = retest candle low (high for bearish).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from quant.features.indicators import opening_range
from strategies.base import MarketContext, Signal, Strategy
from strategies.common import context_ok, futures_confirm, split_day


class BreakoutRetestStrategy(Strategy):
    strategy_id = "breakout_retest_v1"

    def __init__(self, params: dict, version: str = "1.0.0"):
        super().__init__(params)
        self.version = version

    def _setup(self, day: pd.DataFrame, upto: int, lvl_up: float, lvl_dn: float):
        p = self.params
        buf, tol = float(p.get("breakout_buffer_points", 5)), float(p.get("retest_tolerance_points", 8))
        start = int(p.get("opening_range_minutes", 15)) // 5
        for b in range(start, upto + 1):
            c = day.close.iloc[b]
            if c > lvl_up + buf or c < lvl_dn - buf:
                direction = "BULLISH" if c > lvl_up + buf else "BEARISH"
                lvl = lvl_up if direction == "BULLISH" else lvl_dn
                retest = None
                for k in range(b + 1, upto + 1):
                    bar = day.iloc[k]
                    if direction == "BULLISH":
                        if bar.close < lvl - tol:
                            return None
                        if retest is None and bar.low <= lvl + tol and bar.close > lvl:
                            retest = k
                        elif retest is not None and bar.close > day.high.iloc[retest]:
                            return direction, lvl, b, retest, k
                    else:
                        if bar.close > lvl + tol:
                            return None
                        if retest is None and bar.high >= lvl - tol and bar.close < lvl:
                            retest = k
                        elif retest is not None and bar.close < day.low.iloc[retest]:
                            return direction, lvl, b, retest, k
                return ("PENDING", lvl, b, retest, None)
        return None

    def evaluate(self, ctx: MarketContext) -> Optional[Signal]:
        p = self.params
        R = self.last_reasons = []
        today = ctx.now.date()
        day, _ = split_day(ctx.index_5m, today)
        orm = int(p.get("opening_range_minutes", 15))
        if len(day) < orm // 5 + 2:
            R.append("opening_range_incomplete")
            return None
        hi, lo = opening_range(day, orm)
        i = len(day) - 1
        s = self._setup(day, i, hi, lo)
        if s is None or s[0] == "PENDING":
            R.append("no_setup" if s is None else "awaiting_retest_or_confirmation")
            return None
        direction, lvl, b, r, k = s
        if k != i:
            R.append("confirmation_already_passed")
            return None
        if i - b > int(p.get("max_bars_after_breakout", 12)):
            R.append("setup_too_old")
            return None
        last = day.iloc[i]
        reasons = ["or_breakout", "retest_held", "confirmation_close"]
        diag = {"level": lvl, "breakout_bar": int(b), "retest_bar": int(r), "close": float(last.close)}
        if not futures_confirm(ctx.futures_5m, today, last.timestamp, direction, p, reasons, R, diag):
            return None
        if not context_ok(ctx.index_5m, last.timestamp, direction, p, reasons, R, diag):
            return None
        inval = float(day.low.iloc[r] if direction == "BULLISH" else day.high.iloc[r])
        risk = abs(last.close - inval)
        if risk <= 0:
            R.append("non_positive_risk")
            return None
        tgt = float(last.close + 2 * risk if direction == "BULLISH" else last.close - 2 * risk)
        R.append("signal")
        return Signal(self.strategy_id, (last.timestamp + pd.Timedelta(minutes=5)).to_pydatetime(), "NIFTY",
                      direction, 0.6, f"retest of OR {'high' if direction == 'BULLISH' else 'low'} {lvl:.2f} held",
                      inval, tgt, float(last.close), (ctx.regime or {}).get("regime", "unknown"), self.version,
                      reasons, diag)
