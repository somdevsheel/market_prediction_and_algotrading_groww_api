"""Market regime (causal): ADX(14) on completed 5-minute bars, realized volatility percentile, VWAP slope."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _adx(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    mdi = 100 * mdm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def add_regime(df: pd.DataFrame, vwap_slope: pd.Series) -> pd.DataFrame:
    g = df.set_index("timestamp")
    b5 = g.resample("5min", origin="start_day", offset="15min").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    adx5 = _adx(b5["high"], b5["low"], b5["close"])
    # a 5-minute bar is known only when it closes: shift its timestamp to its close time before aligning
    adx5.index = adx5.index + pd.Timedelta(minutes=5)
    out = df.copy()
    out["adx"] = adx5.reindex(g.index, method="ffill").to_numpy()
    r = np.log(out["close"]).diff()
    out["rv"] = r.rolling(30, min_periods=15).std() * np.sqrt(375 * 252) * 100
    w = 375 * 20  # ~20 sessions; rolling z of realized vol (O(n)) instead of a rolling percentile rank
    m, sd = out["rv"].rolling(w, min_periods=375).mean().shift(1), out["rv"].rolling(w, min_periods=375).std().shift(1)
    out["rv_z"] = (out["rv"] - m) / sd.replace(0, np.nan)
    lab = np.full(len(out), "sideways", dtype=object)
    lab[(out["adx"] >= 25) & (vwap_slope > 0)] = "trending_bullish"
    lab[(out["adx"] >= 25) & (vwap_slope < 0)] = "trending_bearish"
    weak = out["adx"] < 25
    lab[weak & (out["rv_z"] >= 1.0)] = "high_volatility"
    lab[weak & (out["rv_z"] <= -1.0)] = "low_volatility"
    out["regime"] = lab
    return out


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean().rename("atr")
