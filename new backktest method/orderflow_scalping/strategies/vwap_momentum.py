"""Strategy 4 — VWAP momentum: continuation when price is extended from VWAP in the direction of the VWAP slope,
confirmed by relative volume and a trend (ADX) filter."""
from __future__ import annotations

from orderflow_scalping.strategies.common import fresh

LABEL = "VWAP Momentum"


def signals(df, p: dict):
    vol_ok = df["rel_volume"] >= p["vol_mult"]
    trend = df["adx"] >= p["adx_min"]
    lc = (df["vwap_z"] > p["dev_z"]) & (df["vwap_slope"] > 0) & vol_ok & trend
    sc = (df["vwap_z"] < -p["dev_z"]) & (df["vwap_slope"] < 0) & vol_ok & trend
    return fresh(lc, sc), None, None
