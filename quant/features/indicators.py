"""Technical indicators. Every function is causal: value at bar t uses bars <= t only."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    m = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = m.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": m, "signal": sig, "hist": m - sig})


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / tr
    mdi = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Intraday VWAP that resets each trading day. Requires real traded volume (use futures, not the index)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = df["timestamp"].dt.date
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vv = df["volume"].groupby(day).cumsum()
    return pv / vv.replace(0, np.nan)


def opening_range(day_df: pd.DataFrame, minutes: int = 15) -> tuple[float, float] | None:
    """High/low of the first ``minutes`` of the session (needs those candles to be complete)."""
    if day_df.empty:
        return None
    start = day_df["timestamp"].iloc[0].normalize() + pd.Timedelta(hours=9, minutes=15)
    w = day_df[day_df["timestamp"] < start + pd.Timedelta(minutes=minutes)]
    if w.empty:
        return None
    return float(w["high"].max()), float(w["low"].min())


def swing_points(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    """Fractal swing highs/lows. A swing at bar i is only *known* at bar i+right (column ``confirmed_at``)."""
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    rows = []
    for i in range(left, len(df) - right):
        if h[i] == max(h[i - left:i + right + 1]):
            rows.append({"idx": i, "kind": "high", "price": h[i], "confirmed_at": i + right})
        if l[i] == min(l[i - left:i + right + 1]):
            rows.append({"idx": i, "kind": "low", "price": l[i], "confirmed_at": i + right})
    return pd.DataFrame(rows, columns=["idx", "kind", "price", "confirmed_at"])


def volume_expansion(volume: pd.Series, lookback: int = 6) -> pd.Series:
    """Ratio of current volume to the mean of the previous ``lookback`` bars (excludes current bar)."""
    return volume / volume.shift(1).rolling(lookback, min_periods=lookback).mean()


def momentum(close: pd.Series, n: int = 3) -> pd.Series:
    return close - close.shift(n)


def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample intraday candles anchored at 09:15; only emits bars whose all source bars exist."""
    g = df.set_index("timestamp")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = g.resample(f"{minutes}min", origin="start_day", offset="15min").agg(agg).dropna(subset=["open"])
    counts = g["close"].resample(f"{minutes}min", origin="start_day", offset="15min").count()
    return out.assign(_n=counts).reset_index().rename(columns={"index": "timestamp"})


def add_standard_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for s in (9, 20, 50, 200):
        out[f"ema{s}"] = ema(out["close"], s)
    out["rsi14"] = rsi(out["close"])
    out = out.join(macd(out["close"]))
    out["atr14"] = atr(out)
    out["adx14"] = adx(out)
    out["mom3"] = momentum(out["close"])
    return out
