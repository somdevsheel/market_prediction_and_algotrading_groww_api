"""Live/paper Strategy for the 9/21 EMA crossover (same rules as strategies/ema_cross/core.py).

Runs on 1-minute candles (engine interval 1); bars are resampled to ``tf`` exactly as in the backtest and a
decision is made only when a ``tf`` bin has just completed. ``exit_signal`` tells the engine to close on the
opposite cross; the engine then re-evaluates the same bar, which produces the reversal entry if the cross is
inside the entry window. Buy modes only (ITM5_BUY via contract_selection.itm_steps=5, ATM_BUY via itm_steps=0).
"""
from __future__ import annotations

from datetime import time
from typing import Optional

import pandas as pd

from strategies.base import MarketContext, Signal, Strategy
from strategies.ema_cross.core import SQ_OFF, resample, with_emas


class EMACrossStrategy(Strategy):
    strategy_id = "ema_cross"
    interval_minutes = 1
    history_days = 12  # calendar days of 1-minute history so EMAs are warm (>= 30 bars at 30m)

    def __init__(self, params: dict, version: str = "1.0.0"):
        super().__init__(params)
        self.version = version

    def _last_bar(self, ctx: MarketContext):
        tf = int(self.params.get("tf", 15))
        bars = with_emas(resample(ctx.index_5m[["timestamp", "open", "high", "low", "close"]], tf))
        if bars.empty:
            return None, tf
        last = bars.iloc[-1]
        T = last["timestamp"] + pd.Timedelta(minutes=tf)
        if T > pd.Timestamp(ctx.now) or (pd.Timestamp(ctx.now) - T).total_seconds() > 300:
            return None, tf  # act only shortly after a bin completes (engine acts on each signal once)
        return last, tf

    def evaluate(self, ctx: MarketContext) -> Optional[Signal]:
        R = self.last_reasons = []
        last, tf = self._last_bar(ctx)
        if last is None or last["cross"] == 0:
            R.append("no_cross_on_completed_bar")
            return None
        T = last["timestamp"] + pd.Timedelta(minutes=tf)
        st = time(*map(int, self.params.get("start", "09:30").split(":")))
        ct = time(*map(int, self.params.get("cutoff", "14:30").split(":")))
        if not (st <= T.time() < ct) or T.time() >= SQ_OFF:
            R.append(f"cross_outside_window:{T:%H:%M}")
            return None
        d = "BULLISH" if last["cross"] > 0 else "BEARISH"
        c = float(last["close"])
        R.append("signal")
        # no structural stop in the rules: invalidation is far away, so the premium stop = risk budget cap
        inval = c * (0.97 if d == "BULLISH" else 1.03)
        return Signal(self.strategy_id, T.to_pydatetime(), "NIFTY", d, 0.5,
                      f"EMA9 crossed {'above' if d == 'BULLISH' else 'below'} EMA21 on {tf}m close", inval,
                      c * (1.5 if d == "BULLISH" else 0.5), c, (ctx.regime or {}).get("regime", "unknown"),
                      self.version, [f"ema9_21_cross_{tf}m"],
                      {"ema9": float(last["ema9"]), "ema21": float(last["ema21"]), "tf": tf, "exit_mode": "opposite_cross"})

    def exit_signal(self, ctx: MarketContext, direction: str) -> bool:
        last, _ = self._last_bar(ctx)
        if last is None:
            return False
        return (direction == "BULLISH" and last["cross"] < 0) or (direction == "BEARISH" and last["cross"] > 0)
