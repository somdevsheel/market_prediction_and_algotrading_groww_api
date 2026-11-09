"""Shared helpers: causal rolling z-scores and 'fresh' signal edges."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rz(x: pd.Series, window: int = 375) -> pd.Series:
    """Rolling z-score using only PAST bars for mean/std (current value vs history)."""
    m = x.rolling(window, min_periods=60).mean().shift(1)
    s = x.rolling(window, min_periods=60).std().shift(1)
    return (x - m) / s.replace(0, np.nan)


def fresh(long_cond: pd.Series, short_cond: pd.Series) -> np.ndarray:
    """+1 / -1 only on the bar where a condition becomes true (prevents re-entering a stale condition)."""
    lc, sc = long_cond.fillna(False).astype(bool), short_cond.fillna(False).astype(bool)
    sig = np.zeros(len(lc), dtype=np.int8)
    sig[(lc & ~lc.shift(1, fill_value=False)).to_numpy()] = 1
    sig[(sc & ~sc.shift(1, fill_value=False)).to_numpy()] = -1
    return sig


def persist(cond: pd.Series, n: int) -> pd.Series:
    return cond.fillna(False).astype(int).rolling(n).sum().eq(n) if n > 1 else cond.fillna(False)
