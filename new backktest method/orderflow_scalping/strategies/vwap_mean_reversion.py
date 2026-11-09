"""Strategy 5 — VWAP mean reversion: fade statistically large deviations from VWAP, exit at VWAP (or stop/time).
Disabled in strong directional regimes (ADX above adx_max)."""
from __future__ import annotations

from orderflow_scalping.strategies.common import fresh

LABEL = "VWAP Mean Reversion"


def signals(df, p: dict):
    calm = df["adx"] < p["adx_max"]
    lc = (df["vwap_z"] < -p["entry_z"]) & calm   # far below VWAP -> long
    sc = (df["vwap_z"] > p["entry_z"]) & calm    # far above VWAP -> short
    ex_long = (df["close"] >= df["vwap"]).to_numpy() if p.get("exit_at_vwap", True) else None
    ex_short = (df["close"] <= df["vwap"]).to_numpy() if p.get("exit_at_vwap", True) else None
    return fresh(lc, sc), ex_long, ex_short
