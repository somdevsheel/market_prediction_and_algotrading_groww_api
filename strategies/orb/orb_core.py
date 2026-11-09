"""Opening Range Breakout setups on 1-minute NIFTY candles (breakout / retest / reversal).

Definitions (from the user's ORB research brief; ambiguities resolved as documented):
* Opening range (OR) = high/low of candles starting in [09:15, 09:15 + N min). range_pct = (H - L) / day open * 100.
* Signals use 1-minute candle CLOSES after the OR is complete; entry price (spot research) = signal close.
* breakout: first close > H (long) or < L (short). Stop = opposite boundary.
* retest:   first breakout, then a close back inside [L, H], then a close beyond the SAME boundary again -> entry.
            If price closes beyond the opposite boundary first, the retest setup is void. Stop = opposite boundary.
            gap_minutes = entry time - initial breakout time.
* reversal: first breakout, then the first close beyond the OPPOSITE boundary -> entry in the opposite direction.
            Stop = the boundary of the initial breakout side.
* Exit (spot research): stop touched (low <= stop for long) -> exit at stop (or the open if it gapped through);
  otherwise at the close of the last candle at or before ``exit_time`` (default 15:29).
* Max one trade per day per setup.

``find_entry`` is causal (uses candles up to the entry only). ``simulate_spot`` uses later candles for the exit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import time
from typing import Optional

import numpy as np
import pandas as pd

SESSION_START_MIN = 9 * 60 + 15


@dataclass
class ORBEntry:
    mode: str
    or_minutes: int
    direction: str            # BULLISH | BEARISH
    entry_idx: int            # index (row) of the signal candle in the day frame
    entry_ts: pd.Timestamp    # decision time = signal candle start + 1 minute
    entry_px: float
    stop_px: float
    or_high: float
    or_low: float
    range_pct: float
    first_breakout_ts: pd.Timestamp
    gap_minutes: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entry_ts"], d["first_breakout_ts"] = str(self.entry_ts), str(self.first_breakout_ts)
        return d


def opening_range(day: pd.DataFrame, minutes: int) -> Optional[tuple[float, float, int]]:
    """Returns (high, low, first_index_after_OR) or None if the OR is incomplete."""
    mins = day["timestamp"].dt.hour * 60 + day["timestamp"].dt.minute
    in_or = (mins >= SESSION_START_MIN) & (mins < SESSION_START_MIN + minutes)
    if in_or.sum() < max(1, minutes * 0.8):  # tolerate a missing minute, not a missing range
        return None
    after = np.flatnonzero((mins >= SESSION_START_MIN + minutes).to_numpy())
    if len(after) == 0:
        return None
    return float(day.loc[in_or, "high"].max()), float(day.loc[in_or, "low"].min()), int(after[0])


def find_entry(day: pd.DataFrame, mode: str, or_minutes: int, upto: Optional[int] = None) -> Optional[ORBEntry]:
    """First qualifying entry using candles [0..upto] (inclusive). Causal."""
    if day.empty:
        return None
    rng = opening_range(day, or_minutes)
    if rng is None:
        return None
    H, L, start = rng
    open_px = float(day["open"].iloc[0])
    rpct = (H - L) / open_px * 100
    closes = day["close"].to_numpy()
    ts = day["timestamp"]
    last = len(day) - 1 if upto is None else min(upto, len(day) - 1)
    first_dir, first_i, been_inside = None, None, False
    for i in range(start, last + 1):
        c = closes[i]
        side = "UP" if c > H else ("DOWN" if c < L else "IN")
        if first_dir is None:
            if side == "IN":
                continue
            first_dir, first_i = side, i
            if mode == "breakout":
                return _mk(mode, or_minutes, side, i, ts, c, H, L, rpct, first_i)
            continue
        if side == "IN":
            been_inside = True
            continue
        if mode == "retest":
            if side != first_dir:
                return None  # opposite boundary first -> retest void
            if been_inside:
                return _mk(mode, or_minutes, side, i, ts, c, H, L, rpct, first_i)
        elif mode == "reversal" and side != first_dir:
            return _mk(mode, or_minutes, side, i, ts, c, H, L, rpct, first_i, reversal=True)
    return None


def _mk(mode, orm, side, i, ts, c, H, L, rpct, first_i, reversal=False) -> ORBEntry:  # noqa: ARG001
    direction = "BULLISH" if side == "UP" else "BEARISH"
    # breakout/retest: stop at the opposite boundary. reversal: the trade direction is opposite to the initial
    # breakout, so "the boundary of the initial breakout side" is again the boundary opposite to the trade.
    stop = L if direction == "BULLISH" else H
    t_entry = ts.iloc[i] + pd.Timedelta(minutes=1)
    t_first = ts.iloc[first_i] + pd.Timedelta(minutes=1)
    return ORBEntry(mode, orm, direction, i, t_entry, float(c), float(stop), H, L, rpct, t_first,
                    (t_entry - t_first).total_seconds() / 60)


def simulate_spot(day: pd.DataFrame, e: ORBEntry, exit_time: time = time(15, 29)) -> dict:
    d = 1 if e.direction == "BULLISH" else -1
    exit_px, reason, exit_ts = None, "time_exit", None
    for i in range(e.entry_idx + 1, len(day)):
        b = day.iloc[i]
        if b["timestamp"].time() > exit_time:
            break
        if d == 1 and b["low"] <= e.stop_px:
            exit_px, reason, exit_ts = min(b["open"], e.stop_px), "stop", b["timestamp"]
            break
        if d == -1 and b["high"] >= e.stop_px:
            exit_px, reason, exit_ts = max(b["open"], e.stop_px), "stop", b["timestamp"]
            break
        exit_px, exit_ts = b["close"], b["timestamp"]
    if exit_px is None:
        exit_px, exit_ts = e.entry_px, e.entry_ts
    return {**e.to_dict(), "exit_px": float(exit_px), "exit_ts": str(exit_ts), "exit_reason": reason,
            "points": round(d * (exit_px - e.entry_px), 2), "date": str(e.entry_ts.date()),
            "entry_time": e.entry_ts.strftime("%H:%M"), "stop_distance": abs(e.entry_px - e.stop_px)}


def passes_filters(e: ORBEntry, f: dict) -> bool:
    if f.get("min_range_pct") is not None and not e.range_pct > f["min_range_pct"]:
        return False
    if f.get("max_range_pct") is not None and not e.range_pct <= f["max_range_pct"]:
        return False
    if f.get("entry_before") and not e.entry_ts.strftime("%H:%M") < f["entry_before"]:
        return False
    if f.get("max_gap_minutes") is not None and not e.gap_minutes <= f["max_gap_minutes"]:
        return False
    return True
