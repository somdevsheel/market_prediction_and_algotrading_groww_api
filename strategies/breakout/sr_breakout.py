"""Strategy A — ``sr_breakout_v1``: support/resistance breakout on completed 5-minute NIFTY candles.

Levels: previous-day high/low, today's opening range, and swing highs/lows that were *confirmed* before the
current candle. Signal on the first close beyond a level (+buffer) with a strong candle, not over-extended.
Optional futures VWAP/volume and 15m context confirmations (off by default: pure S/R).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from quant.features.indicators import opening_range
from strategies.base import MarketContext, Signal, Strategy
from strategies.common import (candle_strength_ok, confirmed_swings, context_ok, futures_confirm, prior_day_levels,
                               split_day)


class SRBreakoutStrategy(Strategy):
    strategy_id = "sr_breakout_v1"

    def __init__(self, params: dict, version: str = "1.0.0"):
        super().__init__(params)
        self.version = version

    def levels(self, day: pd.DataFrame, prev: pd.DataFrame, i: int) -> list[tuple[float, str]]:
        p = self.params
        lv = []
        pd_ = prior_day_levels(prev)
        if pd_:
            lv += [(pd_[0], "pdh"), (pd_[1], "pdl")]
        orr = opening_range(day.iloc[: i + 1], int(p.get("opening_range_minutes", 15)))
        if orr and i >= int(p.get("opening_range_minutes", 15)) // 5:
            lv += [(orr[0], "or_high"), (orr[1], "or_low")]
        if p.get("use_swings", True):
            for _, s in confirmed_swings(day, i).iterrows():
                lv.append((float(s.price), f"swing_{s.kind}"))
        return lv

    def evaluate(self, ctx: MarketContext) -> Optional[Signal]:
        p = self.params
        R = self.last_reasons = []
        today = ctx.now.date()
        day, prev = split_day(ctx.index_5m, today)
        orm = int(p.get("opening_range_minutes", 15))
        if len(day) < orm // 5 + 1:
            R.append("opening_range_incomplete")
            return None
        i = len(day) - 1
        last, pv = day.iloc[i], day.iloc[i - 1]
        buf = float(p.get("breakout_buffer_points", 5))
        crossed_up = [(l, n) for l, n in self.levels(day, prev, i - 1) if pv.close <= l + buf < last.close]
        crossed_dn = [(l, n) for l, n in self.levels(day, prev, i - 1) if pv.close >= l - buf > last.close]
        if crossed_up and not crossed_dn:
            direction, (lvl, name) = "BULLISH", max(crossed_up)
            ext = last.close - (lvl + buf)
        elif crossed_dn and not crossed_up:
            direction, (lvl, name) = "BEARISH", min(crossed_dn)
            ext = (lvl - buf) - last.close
        else:
            R.append("no_level_breakout")
            return None
        reasons = [f"break_{name}"]
        if not candle_strength_ok(last, direction, p.get("min_body_ratio", 0.5)):
            R.append("weak_breakout_candle")
            return None
        if ext > p.get("max_extension_points", 60):
            R.append("overextended")
            return None
        diag = {"level": lvl, "level_name": name, "close": float(last.close), "extension": round(float(ext), 2)}
        if not futures_confirm(ctx.futures_5m, today, last.timestamp, direction, p, reasons, R, diag):
            return None
        if not context_ok(ctx.index_5m, last.timestamp, direction, p, reasons, R, diag):
            return None
        regime = (ctx.regime or {}).get("regime", "unknown")
        inval = float(last.low if direction == "BULLISH" else last.high)
        risk = abs(last.close - inval)
        if risk <= 0:
            R.append("non_positive_risk")
            return None
        tgt = float(last.close + 2 * risk if direction == "BULLISH" else last.close - 2 * risk)
        R.append("signal")
        return Signal(self.strategy_id, (last.timestamp + pd.Timedelta(minutes=5)).to_pydatetime(), "NIFTY",
                      direction, 0.5, f"close beyond {name} {lvl:.2f}±{buf}", inval, tgt, float(last.close), regime,
                      self.version, reasons, diag)
