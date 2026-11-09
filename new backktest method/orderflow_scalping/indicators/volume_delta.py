"""Volume delta.

* ``delta_from_trades``: GENUINE — aggressor-side trades (side = +1 buy, -1 sell) -> buy volume - sell volume.
* ``delta_bvc``: PROXY — Bulk Volume Classification (Easley, Lopez de Prado, O'Hara 2012):
    buy_fraction = Phi(dP / sigma_dP);  delta = V * (2 * buy_fraction - 1)
  It does not use candle colour alone (as the brief requires) but is still an ESTIMATE, labelled proxy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def delta_from_trades(qty: np.ndarray, side: np.ndarray) -> float:
    return float(np.sum(np.asarray(qty) * np.asarray(side)))


def delta_bvc(df: pd.DataFrame, sigma_window: int = 60) -> pd.Series:
    dp = df["close"].diff()
    sig = dp.rolling(sigma_window, min_periods=20).std().shift(1)  # sigma from PAST bars only
    frac = pd.Series(norm.cdf((dp / sig.replace(0, np.nan)).fillna(0)), index=df.index)
    return (df["volume"] * (2 * frac - 1)).rename("delta_bvc")


def cumulative(delta: pd.Series, window: int) -> pd.Series:
    return delta.rolling(window).sum()


def zscore(x: pd.Series, window: int = 120) -> pd.Series:
    m, s = x.rolling(window, min_periods=30).mean().shift(1), x.rolling(window, min_periods=30).std().shift(1)
    return (x - m) / s.replace(0, np.nan)
