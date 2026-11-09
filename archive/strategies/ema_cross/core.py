"""9/21 EMA crossover (intraday, NIFTY spot signal, options execution) — rules with every ambiguity explicit.

Bars & EMAs
* 1-minute NIFTY bars are resampled to TF ∈ {5, 15, 30} minutes, bins anchored at 09:15. Only COMPLETE bins
  (all TF one-minute bars present) are used.
* EMA9/EMA21 = exponential moving averages (adjust=False) of resampled closes, computed CONTINUOUSLY across
  days (yesterday's bars warm up today's EMAs). Values before 30 bars of history are not traded.

Crosses
* diff = EMA9 − EMA21 on each closed bar. A bullish cross happens at bar i when the last non-zero diff before
  i was < 0 and diff_i > 0 (bearish: mirror). diff == 0 exactly is a TIE: it is not a cross and does not reset
  the side (the next non-zero diff decides).
* The signal is known when the bar CLOSES: decision time T = bar start + TF.
* cross_no = 1, 2, 3 … counts every cross of the day (including crosses before the entry window).

Trading
* Entry window: start ≤ T < cutoff. Crosses outside it never open trades.
* FILL BAR: the 1-minute bar starting at T; fills use its OPEN (spot and option).
* Exit: the next opposite cross (at its T, any time before square-off) or the mandatory square-off at 15:00
  (fill at the open of the 15:00 one-minute bar). If the opposite cross is inside the window, the position is
  reversed at the same fill bar (close + open). Crosses at/after 15:00 are ignored.
* No-trade days: holidays/missing data (fewer than 300 one-minute bars, or no complete bins in the window),
  or no cross inside the window. Recorded with a reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

SQ_OFF = time(15, 0)


def resample(df1: pd.DataFrame, tf: int) -> pd.DataFrame:
    g = df1.set_index("timestamp")
    agg = g.resample(f"{tf}min", origin="start_day", offset="15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    n = g["close"].resample(f"{tf}min", origin="start_day", offset="15min").count()
    out = agg.assign(n=n).dropna(subset=["open"])
    out = out[out["n"] == tf].reset_index()
    t = out["timestamp"].dt.time
    return out[(t >= time(9, 15)) & (t < time(15, 30))].reset_index(drop=True)


def with_emas(bars: pd.DataFrame) -> pd.DataFrame:
    b = bars.copy()
    b["ema9"] = b["close"].ewm(span=9, adjust=False).mean()
    b["ema21"] = b["close"].ewm(span=21, adjust=False).mean()
    b["diff"] = b["ema9"] - b["ema21"]
    side, last = [], 0
    crosses = []
    for i, d in enumerate(b["diff"].to_numpy()):
        s = 1 if d > 0 else (-1 if d < 0 else 0)
        c = 0
        if s != 0 and last != 0 and s != last and i >= 30:
            c = s
        if s != 0:
            last = s
        crosses.append(c)
    b["cross"] = crosses
    return b


@dataclass
class EMAConfig:
    tf: int = 5
    start: str = "09:30"
    cutoff: str = "14:30"


def day_trades(day_bars: pd.DataFrame, day_1m: pd.DataFrame, cfg: EMAConfig) -> tuple[list[dict], str]:
    """Spot trades for one day. ``day_bars`` = resampled bars WITH emas/cross for that day only."""
    if len(day_1m) < 300:
        return [], "missing_data"
    st = time(*map(int, cfg.start.split(":")))
    ct = time(*map(int, cfg.cutoff.split(":")))
    opens = dict(zip(day_1m["timestamp"], day_1m["open"]))
    sq_ts = pd.Timestamp.combine(day_1m["timestamp"].iloc[0].date(), SQ_OFF).tz_localize(day_1m["timestamp"].dt.tz)
    events = []
    cross_no = 0
    for _, b in day_bars.iterrows():
        if b["cross"] == 0:
            continue
        cross_no += 1
        T = b["timestamp"] + pd.Timedelta(minutes=cfg.tf)
        if T.time() >= SQ_OFF:
            break
        events.append((T, int(b["cross"]), cross_no))
    trades, pos = [], None
    for T, s, no in events:
        px = opens.get(T)
        if px is None:
            continue
        inside = st <= T.time() < ct
        if pos is not None and s != pos["side"]:
            pos.update(exit_ts=T, exit_spot=px, exit_reason="opposite_cross")
            trades.append(pos)
            pos = None
        if pos is None and inside:
            pos = {"side": s, "direction": "BULLISH" if s > 0 else "BEARISH", "entry_ts": T, "entry_spot": px,
                   "cross_no": no, "reversal": bool(trades) and trades[-1]["exit_ts"] == T}
    if pos is not None:
        px = opens.get(sq_ts)
        if px is None:  # square-off bar missing -> last bar before 15:00
            before = day_1m[day_1m["timestamp"] <= sq_ts]
            px, sq = float(before["open"].iloc[-1]), before["timestamp"].iloc[-1]
        else:
            sq = sq_ts
        pos.update(exit_ts=sq, exit_spot=px, exit_reason="square_off_15:00")
        trades.append(pos)
    for t in trades:
        t["spot_points"] = round(t["side"] * (t["exit_spot"] - t["entry_spot"]), 2)
    return trades, ("no_cross_in_window" if not trades else "traded")


def spot_backtest(df1: pd.DataFrame, cfg: EMAConfig) -> tuple[pd.DataFrame, dict]:
    bars = with_emas(resample(df1, cfg.tf))
    by_day_bars = {d: g for d, g in bars.groupby(bars["timestamp"].dt.date)}
    out, nodays = [], {}
    for d, g1 in df1.groupby(df1["timestamp"].dt.date):
        tr, why = day_trades(by_day_bars.get(d, bars.iloc[0:0]), g1.reset_index(drop=True), cfg)
        if why != "traded":
            nodays[why] = nodays.get(why, 0) + 1
        for t in tr:
            t["date"] = str(d)
            out.append(t)
    return pd.DataFrame(out), nodays
