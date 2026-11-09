"""Order Flow Imbalance.

* ``ofi_cks``: event-based Cont-Kukanov-Stoikov OFI from best bid/ask prices and quantities (GENUINE; needs L2).
    e_n = 1{Pb_n >= Pb_{n-1}} qb_n - 1{Pb_n <= Pb_{n-1}} qb_{n-1} - 1{Pa_n <= Pa_{n-1}} qa_n + 1{Pa_n >= Pa_{n-1}} qa_{n-1}
* ``ofi_proxy``: candle PROXY = close-location value x volume, CLV = ((C-L) - (H-C)) / (H-L). It estimates where
  volume traded within the bar; it is NOT order-book flow and results using it are labelled proxy.
Normalisation: NOFI = rolling_sum(OFI, w) / (rolling mean depth-or-volume + eps).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


def ofi_cks(bid: np.ndarray, ask: np.ndarray, bidq: np.ndarray, askq: np.ndarray) -> np.ndarray:
    bid, ask, bidq, askq = map(np.asarray, (bid, ask, bidq, askq))
    e = np.zeros(len(bid))
    e[1:] = ((bid[1:] >= bid[:-1]) * bidq[1:] - (bid[1:] <= bid[:-1]) * bidq[:-1]
             - (ask[1:] <= ask[:-1]) * askq[1:] + (ask[1:] >= ask[:-1]) * askq[:-1])
    return e


def ofi_proxy(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    return (clv.fillna(0) * df["volume"]).rename("ofi_proxy")


def normalized(ofi: pd.Series, depth_or_volume: pd.Series, window: int, norm_window: int = 60) -> pd.Series:
    return ofi.rolling(window).sum() / (depth_or_volume.rolling(norm_window).mean() * window + EPS)
