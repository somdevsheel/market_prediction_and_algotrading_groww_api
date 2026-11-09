"""Historical / realized volatility.

``r(t) = ln(S(t)/S(t-1))``; annualized HV = std(r) * sqrt(periods_per_year) * 100.
For intraday bars pass the matching ``periods_per_year`` (e.g. 5-minute bars:
252 * 75 = 18 900); never reuse the daily factor on intraday returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MINUTES_PER_SESSION = 375


def periods_per_year(interval_minutes: int | None) -> float:
    if not interval_minutes or interval_minutes >= 1440:
        return TRADING_DAYS
    return TRADING_DAYS * MINUTES_PER_SESSION / interval_minutes


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def historical_volatility(close: pd.Series, window: int, ppy: float = TRADING_DAYS) -> pd.Series:
    """Rolling annualized HV in percent (sample std, ddof=1)."""
    return log_returns(close).rolling(window).std(ddof=1) * np.sqrt(ppy) * 100


def hv_table(close: pd.Series, windows=(5, 10, 20, 30, 60), ppy: float = TRADING_DAYS) -> pd.DataFrame:
    return pd.DataFrame({f"HV{w}": historical_volatility(close, w, ppy) for w in windows})


def realized_volatility(returns: pd.Series, ppy: float = TRADING_DAYS) -> float:
    """Annualized realized vol (%) from sum of squared returns (zero-mean)."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    return float(np.sqrt((r ** 2).sum() * ppy / len(r)) * 100)


def intraday_realized_vol(df: pd.DataFrame, interval_minutes: int) -> pd.Series:
    """Per-day realized vol (%) from intraday closes (overnight gap excluded)."""
    out = {}
    for d, g in df.groupby(df["timestamp"].dt.date):
        out[d] = realized_volatility(log_returns(g["close"]), periods_per_year(interval_minutes))
    return pd.Series(out, name="intraday_rv")


def volatility_percentile(series: pd.Series, lookback: int = 252) -> pd.Series:
    """Percent of the trailing ``lookback`` values that are <= the current value."""
    return series.rolling(lookback, min_periods=20).apply(lambda w: (w <= w[-1]).mean() * 100, raw=True)


def vol_regime_change(hv_short: pd.Series, hv_long: pd.Series, threshold: float = 1.2) -> pd.Series:
    ratio = hv_short / hv_long
    return pd.Series(np.where(ratio > threshold, "expansion", np.where(ratio < 1 / threshold, "contraction", "stable")),
                     index=ratio.index).where(ratio.notna())
