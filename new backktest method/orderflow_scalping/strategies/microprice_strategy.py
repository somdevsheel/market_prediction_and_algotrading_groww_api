"""Strategy 2 — Microprice prediction. GENUINE ONLY: needs recorded L2 snapshots (bid, ask, bid_qty, ask_qty).

``signals_from_snapshots`` works on snapshot data (e.g. from data/ingestion/record_depth.py). For candle data
``signals`` returns None so the runner reports 'not testable' instead of inventing order-book values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from orderflow_scalping.indicators.microprice import deviation

LABEL = "Microprice"
TESTABLE_ON_CANDLES = False


def signals(df, p: dict):
    return None


def signals_from_snapshots(snap: pd.DataFrame, p: dict, tick: float = 0.05) -> np.ndarray:
    mpd = deviation(snap["bid"], snap["ask"], snap["bid_qty"], snap["ask_qty"])
    thr = p.get("entry_ticks", 0.5) * tick
    sig = np.zeros(len(snap), dtype=np.int8)
    sig[mpd > thr] = 1
    sig[mpd < -thr] = -1
    return sig
