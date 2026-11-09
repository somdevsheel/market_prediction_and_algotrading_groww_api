"""Test-only strategy: emits a BULLISH signal on the first completed 5-minute candle at/after ``at`` (HH:MM).
Lets engine tests exercise entry/exit/recovery logic independently of any real strategy."""
from __future__ import annotations

import pandas as pd

from strategies.base import MarketContext, Signal, Strategy


class FakeStrategy(Strategy):
    strategy_id = "fake_v1"
    interval_minutes = 5

    def __init__(self, params: dict, version: str = "1.0.0"):
        super().__init__(params)
        self.version = version

    def evaluate(self, ctx: MarketContext):
        self.last_reasons = []
        day = ctx.index_5m[ctx.index_5m["timestamp"].dt.date == ctx.now.date()]
        if day.empty:
            return None
        last = day.iloc[-1]
        at = pd.Timestamp.combine(ctx.now.date(), pd.Timestamp(self.params.get("at", "09:45")).time()).tz_localize(last["timestamp"].tz)
        if last["timestamp"] < at:
            self.last_reasons.append("waiting")
            return None
        c = float(last["close"])
        return Signal("fake_v1", (last["timestamp"] + pd.Timedelta(minutes=5)).to_pydatetime(), "NIFTY", "BULLISH", 1.0,
                      "test", float(last["low"]) - 10, c + 100, c, "test", self.version, ["test"], {})


def install(cfg: dict, at: str = "09:45") -> None:
    from strategies import registry

    registry.CLASSES["fake"] = FakeStrategy
    cfg["strategies"]["strategies"]["fake_v1"] = {"class": "fake", "status": "RESEARCH", "version": "1.0.0",
                                                  "params": {"at": at}}
    cfg["strategies"]["active_strategy"] = "fake_v1"
