"""Swing strategies (long-only, delivery). Each has prepare() (causal indicators, computed once) and decide()
(called at each close with data up to that day; returns symbols to sell and a ranked list to buy)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _eligible(panel, min_turnover_cr: float = 5.0, min_price: float = 20.0) -> pd.DataFrame:
    liq = panel["turnover"].rolling(20, min_periods=15).median() >= min_turnover_cr * 1e7
    return liq & (panel["close"] >= min_price)


def _market_ok(panel, nifty: pd.Series | None) -> pd.Series:
    if nifty is None:
        return pd.Series(True, index=panel["close"].index)
    up = (nifty > nifty.rolling(200, min_periods=150).mean()).reindex(panel["close"].index).ffill()
    return up.fillna(False).astype(bool)


class Momentum:
    """Cross-sectional momentum: at each month-end, hold the top-N stocks by return over `lookback` days, skipping
    the most recent `skip` days; equal weight. Optional market filter: in cash when NIFTY < 200-day average."""
    name = "momentum"

    def prepare(self, panel, p):
        c = panel["close"]
        mom = c.shift(p["skip"]) / c.shift(p["lookback"]) - 1
        elig = _eligible(panel)
        idx = c.index
        month_end = pd.Series(idx.to_period("M"), index=idx)
        is_me = month_end != month_end.shift(-1)
        return {"mom": mom.where(elig), "is_me": is_me, "mkt": _market_ok(panel, p.get("nifty") if p.get("market_filter") else None)}

    def decide(self, d, s, held, pos, p):
        if not s["is_me"].loc[d]:
            return [], []
        if not s["mkt"].loc[d]:
            return list(held), []
        top = s["mom"].loc[d].dropna().sort_values(ascending=False).head(p["n"]).index.tolist()
        return [h for h in held if h not in top], [t for t in top if t not in held]


class Breakout52w:
    """New 252-day closing high with volume > vol_mult x 50-day average, stock above its 200-day average and the
    market in an uptrend -> buy. Exit when the close falls below the 50-day average."""
    name = "breakout_52w"

    def prepare(self, panel, p):
        c, v = panel["close"], panel["volume"]
        hi = c.rolling(252, min_periods=200).max().shift(1)
        sig = (c > hi) & (v > p["vol_mult"] * v.rolling(50, min_periods=40).mean().shift(1)) & \
              (c > c.rolling(200, min_periods=150).mean()) & _eligible(panel)
        strength = (c / hi - 1).where(sig)
        return {"entry": strength, "exit": c < c.rolling(50, min_periods=40).mean(),
                "mkt": _market_ok(panel, p.get("nifty") if p.get("market_filter") else None)}

    def decide(self, d, s, held, pos, p):
        ex = s["exit"].loc[d]
        sells = [h for h in held if bool(ex.get(h, False))]
        if not s["mkt"].loc[d]:
            return sells, []
        buys = s["entry"].loc[d].dropna().sort_values(ascending=False).index.tolist()
        return sells, buys


class PullbackRSI2:
    """Short-term pullback in an uptrend: RSI(2) below `rsi` while above the 200-day average -> buy (deepest
    first). Exit when the close is above the 5-day average or after `max_days` trading days."""
    name = "pullback_rsi2"

    def prepare(self, panel, p):
        c = panel["close"]
        d = c.diff()
        up = d.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
        rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
        sig = (rsi < p["rsi"]) & (c > c.rolling(200, min_periods=150).mean()) & _eligible(panel)
        return {"entry": rsi.where(sig), "exit": c > c.rolling(5).mean(), "dates": c.index,
                "mkt": _market_ok(panel, p.get("nifty") if p.get("market_filter") else None)}

    def decide(self, d, s, held, pos, p):
        ex = s["exit"].loc[d]
        sells = [h for h in held if bool(ex.get(h, False)) or
                 (s["dates"].get_loc(d) - s["dates"].get_loc(pos[h]["entry_date"]) >= p["max_days"])]
        if not s["mkt"].loc[d]:
            return sells, []
        return sells, s["entry"].loc[d].dropna().sort_values().index.tolist()


STRATEGIES = {"momentum": Momentum(), "breakout_52w": Breakout52w(), "pullback_rsi2": PullbackRSI2()}
