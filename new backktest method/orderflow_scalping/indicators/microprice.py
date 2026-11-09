"""Microprice (GENUINE only — requires best bid/ask quantities).

MP = (Ask * BidQty + Bid * AskQty) / (BidQty + AskQty);  MPD = MP - Mid.
There is deliberately NO candle proxy: bid/ask quantities cannot be inferred from OHLCV, and the research brief
forbids inventing order-book data. Historical microprice tests therefore require recorded L2 snapshots
(see data/ingestion/record_depth.py).
"""
from __future__ import annotations

import numpy as np


def microprice(bid, ask, bidq, askq) -> np.ndarray:
    bid, ask, bidq, askq = map(lambda x: np.asarray(x, float), (bid, ask, bidq, askq))
    tot = bidq + askq
    return np.where(tot > 0, (ask * bidq + bid * askq) / np.where(tot > 0, tot, 1), (bid + ask) / 2)


def deviation(bid, ask, bidq, askq) -> np.ndarray:
    return microprice(bid, ask, bidq, askq) - (np.asarray(bid, float) + np.asarray(ask, float)) / 2
