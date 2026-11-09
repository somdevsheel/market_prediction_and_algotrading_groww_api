"""Session VWAP (resets daily) from 1-minute futures bars, using typical price x volume (an approximation of
trade-level VWAP, which needs tick data). Deviation is normalised by the rolling std of (close - VWAP)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def session_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = df["timestamp"].dt.date
    pv = (tp * df["volume"]).groupby(day).cumsum()
    v = df["volume"].groupby(day).cumsum()
    return (pv / v.replace(0, np.nan)).rename("vwap")


def deviation_z(df: pd.DataFrame, vwap: pd.Series, window: int = 60) -> pd.Series:
    dev = df["close"] - vwap
    sd = dev.rolling(window, min_periods=20).std().shift(1)
    return (dev / sd.replace(0, np.nan)).rename("vwap_z")


def slope(vwap: pd.Series, bars: int = 10) -> pd.Series:
    return (vwap - vwap.shift(bars)).rename("vwap_slope")
