"""Simple baselines the brief requires every strategy to beat."""
from __future__ import annotations

from orderflow_scalping.strategies.common import fresh


def vwap_cross(df, p: dict):
    above = df["close"] > df["vwap"]
    sig = fresh(above, ~above & df["vwap"].notna())
    return sig, (~above).to_numpy(), above.to_numpy()


def momentum(df, p: dict):
    mv = df["close"] - df["close"].shift(int(p["lookback"]))
    thr = p["min_move_atr"] * df["atr"]
    return fresh(mv > thr, mv < -thr), None, None
