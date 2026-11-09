"""Data validation: duplicates, bad prices, invalid OHLC, out-of-session bars, holidays/short sessions, gaps.
Problems are reported and offending rows dropped from the research set (never filled with invented values)."""
from __future__ import annotations

from datetime import time

import pandas as pd


def validate(df: pd.DataFrame, min_bars_per_day: int = 300) -> tuple[pd.DataFrame, dict]:
    rep: dict = {"rows_in": len(df)}
    d = df.drop_duplicates("timestamp")
    rep["duplicates"] = len(df) - len(d)
    bad = (d[["open", "high", "low", "close"]] <= 0).any(axis=1) | d[["open", "high", "low", "close"]].isna().any(axis=1)
    inv = (d["high"] < d[["open", "close"]].max(axis=1)) | (d["low"] > d[["open", "close"]].min(axis=1))
    neg = d["volume"].fillna(0) < 0
    t = d["timestamp"].dt.time
    out = (t < time(9, 15)) | (t >= time(15, 30))
    rep.update(bad_prices=int(bad.sum()), invalid_ohlc=int(inv.sum()), negative_volume=int(neg.sum()),
               outside_session=int(out.sum()))
    d = d[~(bad | inv | neg | out)].sort_values("timestamp")
    counts = d.groupby(d["timestamp"].dt.date).size()
    short = counts[counts < min_bars_per_day]
    rep["sessions"] = int(len(counts))
    rep["short_sessions_dropped"] = {str(k): int(v) for k, v in short.items()}  # muhurat / half days / gaps
    d = d[~d["timestamp"].dt.date.isin(short.index)]
    full = counts[counts >= min_bars_per_day]
    rep["missing_minutes_in_kept_sessions"] = int((375 - full.clip(upper=375)).sum())
    rep["zero_volume_bars"] = int((d["volume"].fillna(0) == 0).sum())
    rep["rows_out"] = len(d)
    return d.reset_index(drop=True), rep
