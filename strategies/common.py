"""Shared helpers for intraday breakout-family strategies (all causal)."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from quant.features.indicators import ema, resample, session_vwap, volume_expansion, swing_points


def split_day(idx: pd.DataFrame, today) -> tuple[pd.DataFrame, pd.DataFrame]:
    day = idx[idx["timestamp"].dt.date == today].reset_index(drop=True)
    prev = idx[idx["timestamp"].dt.date < today]
    return day, prev


def prior_day_levels(prev: pd.DataFrame) -> Optional[tuple[float, float]]:
    if prev.empty:
        return None
    last_day = prev[prev["timestamp"].dt.date == prev["timestamp"].dt.date.iloc[-1]]
    return float(last_day.high.max()), float(last_day.low.min())


def candle_strength_ok(bar, direction: str, min_body_ratio: float) -> bool:
    rng = bar["high"] - bar["low"]
    body = abs(bar["close"] - bar["open"])
    up = bar["close"] > bar["open"]
    return rng > 0 and body / rng >= min_body_ratio and (up if direction == "BULLISH" else not up)


def futures_confirm(fut: Optional[pd.DataFrame], today, ts, direction: str, p: dict, reasons: list, R: list,
                    diag: dict) -> bool:
    if not (p.get("use_vwap_filter") or p.get("use_volume_filter")):
        return True
    if fut is None or fut.empty:
        R.append("no_futures_data")
        return False
    fday = fut[fut["timestamp"].dt.date == today].reset_index(drop=True)
    fbar = fday[fday["timestamp"] == ts]
    if fbar.empty:
        R.append("futures_candle_missing")
        return False
    i = fbar.index[0]
    if p.get("use_vwap_filter"):
        vw = session_vwap(fday).iloc[i]
        c = fday["close"].iloc[i]
        diag["fut_close"], diag["fut_vwap"] = float(c), float(vw)
        if not (c > vw if direction == "BULLISH" else c < vw):
            R.append("vwap_filter_failed")
            return False
        reasons.append("futures_vwap_confirm")
    if p.get("use_volume_filter"):
        vx = volume_expansion(fday["volume"], int(p.get("volume_lookback", 6))).iloc[i]
        diag["volume_ratio"] = None if pd.isna(vx) else round(float(vx), 3)
        if pd.isna(vx) or vx < p.get("volume_multiplier", 1.2):
            R.append("volume_filter_failed")
            return False
        reasons.append("futures_volume_expansion")
    return True


def context_ok(idx: pd.DataFrame, last_ts, direction: str, p: dict, reasons: list, R: list, diag: dict) -> bool:
    if not p.get("use_context_filter"):
        return True
    m15 = resample(idx, 15)
    m15 = m15[m15["_n"] == 3]
    m15 = m15[m15["timestamp"] + pd.Timedelta(minutes=15) <= last_ts + pd.Timedelta(minutes=5)]
    e = ema(m15["close"], int(p.get("context_ema", 20)))
    if e.dropna().empty:
        R.append("context_insufficient_history")
        return False
    c15, e15 = m15["close"].iloc[-1], e.iloc[-1]
    diag["ctx15_close"], diag["ctx15_ema"] = float(c15), float(e15)
    if (direction == "BULLISH" and c15 <= e15) or (direction == "BEARISH" and c15 >= e15):
        R.append("context_filter_failed")
        return False
    reasons.append("15m_ema_context")
    return True


def confirmed_swings(day: pd.DataFrame, upto_i: int, left: int = 2, right: int = 2) -> pd.DataFrame:
    sw = swing_points(day.iloc[: upto_i + 1], left, right)
    return sw[sw["confirmed_at"] < upto_i]
