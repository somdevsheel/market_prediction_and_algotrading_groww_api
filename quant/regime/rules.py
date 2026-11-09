"""Baseline rule-based market-regime classifier (transparent; HMM comes later)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from quant.features.indicators import adx, atr, ema

REGIMES = ("trending_bullish", "trending_bearish", "range_bound", "high_volatility", "low_volatility",
           "volatility_expansion", "unstable")


@dataclass
class RegimeConfig:
    adx_trend: float = 22.0
    adx_range: float = 16.0
    atr_expansion: float = 1.5     # ATR(14) / ATR(14) 20 bars ago
    vix_high: float = 20.0
    vix_low: float = 12.0
    vix_unstable: float = 25.0


def classify(df: pd.DataFrame, vix: Optional[float] = None, cfg: RegimeConfig = RegimeConfig()) -> dict:
    """Classify the *latest completed* bar of ``df`` (OHLC). Returns label + diagnostics."""
    if len(df) < 40:
        return {"regime": "unknown", "reason": "insufficient_bars"}
    a = adx(df).iloc[-1]
    at = atr(df)
    ratio = at.iloc[-1] / at.iloc[-21] if at.iloc[-21] else float("nan")
    e20, e50 = ema(df["close"], 20).iloc[-1], ema(df["close"], 50).iloc[-1]
    c = df["close"].iloc[-1]
    diag = {"adx": round(float(a), 2), "atr_ratio": round(float(ratio), 3), "ema20": round(float(e20), 2),
            "ema50": round(float(e50), 2), "vix": vix}
    if vix is not None and vix >= cfg.vix_unstable:
        label = "unstable"
    elif ratio >= cfg.atr_expansion:
        label = "volatility_expansion"
    elif a >= cfg.adx_trend and c > e20 > e50:
        label = "trending_bullish"
    elif a >= cfg.adx_trend and c < e20 < e50:
        label = "trending_bearish"
    elif vix is not None and vix >= cfg.vix_high:
        label = "high_volatility"
    elif a <= cfg.adx_range:
        label = "range_bound"
    elif vix is not None and vix <= cfg.vix_low:
        label = "low_volatility"
    else:
        label = "range_bound"
    return {"regime": label, **diag}
