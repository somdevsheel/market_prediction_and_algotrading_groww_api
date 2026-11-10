"""Intraday equity research on 5-minute candles ("stocks in play" opening-range ideas).

Step 1 (per stock, vectorised): a day table with features known at the decision time and the outcome of a long
and a short trade under each stop multiple:
  * decision at the close of the first 5-min bar (09:15-09:20) [orb5] or of the first 15 min [drive15];
  * rel_vol = first-bar volume / mean first-bar volume of the previous 14 sessions (past only);
  * atr14 = 14-day ATR of daily ranges built from 5-min bars, previous days only;
  * entry = OPEN of the next bar; stop = entry -/+ s * atr14; exit at the stop (or a gapped open) or at the open of
    the 15:15 bar. Stop checks use bar lows/highs AFTER entry (entry bar included, stop assumed first).
Step 2 (research script): pick the top-K stocks by rel_vol each day, size by risk, apply charges & slippage.
"""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from core.config import load_config, resolve_path

STOPS = (0.05, 0.10, 0.20)


def load_5m(sym: str) -> pd.DataFrame:
    root = resolve_path(load_config(), "data_lake") / "silver" / "equity" / f"symbol={sym}" / "interval=5minute"
    files = sorted(root.glob("year=*/month=*/part.parquet"))
    if not files:
        return pd.DataFrame()
    x = pd.concat([pd.read_parquet(f, columns=["timestamp", "open", "high", "low", "close", "volume"]) for f in files])
    ts = pd.to_datetime(x["timestamp"])
    x["timestamp"] = ts.dt.tz_convert("Asia/Kolkata") if ts.dt.tz is not None else ts.dt.tz_localize("Asia/Kolkata")
    x = x.drop_duplicates("timestamp").sort_values("timestamp")
    t = x["timestamp"].dt.time
    return x[(t >= time(9, 15)) & (t < time(15, 30))].reset_index(drop=True)


def _sim(day: pd.DataFrame, k: int, side: int, stop_pts: float) -> tuple[float, float, str]:
    """Enter at open of bar k; returns (entry, exit, reason)."""
    entry = day["open"].iat[k]
    stop = entry - side * stop_pts
    lows, highs, opens, ts = day["low"].to_numpy(), day["high"].to_numpy(), day["open"].to_numpy(), day["timestamp"]
    last = int(np.searchsorted(ts.dt.time.to_numpy(), time(15, 15)))
    last = min(last, len(day) - 1)
    for j in range(k, last):
        if side > 0 and lows[j] <= stop:
            return entry, min(stop, opens[j]) if j > k else stop, "stop"
        if side < 0 and highs[j] >= stop:
            return entry, max(stop, opens[j]) if j > k else stop, "stop"
    return entry, opens[last], "eod"


def day_table(sym: str) -> pd.DataFrame:
    df = load_5m(sym)
    if df.empty:
        return df
    df["date"] = df["timestamp"].dt.date
    g = df.groupby("date")
    daily = pd.DataFrame({"h": g["high"].max(), "l": g["low"].min(), "c": g["close"].last(), "n": g.size()})
    tr = pd.concat([daily.h - daily.l, (daily.h - daily.c.shift()).abs(), (daily.l - daily.c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=10).mean().shift(1)          # previous days only
    first = g.head(1).set_index("date")
    rel = first["volume"] / first["volume"].rolling(14, min_periods=10).mean().shift(1)
    rows = []
    for d, day in g:
        if len(day) < 60 or not np.isfinite(atr.get(d, np.nan)) or not np.isfinite(rel.get(d, np.nan)):
            continue
        day = day.reset_index(drop=True)
        f = first.loc[d]
        base = {"date": d, "sym": sym, "rel_vol": float(rel[d]), "atr14": float(atr[d]), "price": float(f["close"]),
                "first_dir": int(np.sign(f["close"] - f["open"])),
                "drive_dir": int(np.sign(day["close"].iat[2] - day["open"].iat[0]))}
        for setup, k, dcol in (("orb5", 1, "first_dir"), ("drive15", 3, "drive_dir")):
            side = base[dcol]
            if side == 0:
                continue
            for s in STOPS:
                en, ex, why = _sim(day, k, side, s * base["atr14"])
                rows.append({**base, "setup": setup, "stop_mult": s, "side": side, "entry": en, "exit": ex, "reason": why})
    return pd.DataFrame(rows)
