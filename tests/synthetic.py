"""SYNTHETIC market data for tests only. Never presented as real performance."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from core.clock import IST
from core.models import ChainRow, OptionChain, Quote


def day_candles(d: date, path: list[float], vol: list[float] | None = None, wiggle: float = 3.0) -> pd.DataFrame:
    """5-minute candles from a list of closes; open = previous close."""
    rows = []
    prev = path[0]
    for i, c in enumerate(path):
        ts = datetime.combine(d, time(9, 15), IST) + timedelta(minutes=5 * i)
        o = prev
        rows.append([ts, o, max(o, c) + wiggle, min(o, c) - wiggle, c, (vol[i] if vol else 1000.0), None])
        prev = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "open_interest"])


def history(days: list[date], start: float = 25000.0, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames, px = [], start
    for d in days:
        path = list(px + np.cumsum(rng.normal(0, 8, 75)))
        frames.append(day_candles(d, path))
        px = path[-1]
    return pd.concat(frames, ignore_index=True)


def breakout_day(d: date, base: float = 25000.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Opening range 25000..25040, then a clean bullish breakout at 09:45 candle with volume spike."""
    closes = [25010, 25030, 25020,  # OR (09:15-09:25), high ~25033+3
              25025, 25028, 25026,   # 09:30-09:40 inside
              25080]                 # 09:45 breakout candle closes above OR high + buffer
    closes = [c - 25000 + base for c in closes]
    idx = day_candles(d, closes, wiggle=2.0)
    # make breakout candle strong: open near prior close, close near high
    idx.loc[6, "open"] = closes[5]
    idx.loc[6, "high"] = closes[6] + 1
    idx.loc[6, "low"] = closes[5] - 1
    fut_closes = [c + 40 for c in closes]
    vols = [5000, 4000, 3500, 3000, 3000, 3000, 9000]
    fut = day_candles(d, fut_closes, vol=vols, wiggle=2.0)
    return idx, fut


def quote(sym: str, bid: float, ask: float, now: datetime, volume=600000, oi=80000) -> Quote:
    return Quote(trading_symbol=sym, ltp=(bid + ask) / 2, bid=bid, ask=ask, volume=volume, open_interest=oi,
                 last_trade_time=now, received_at=now)


def chain(expiry: date, spot: float, now: datetime) -> OptionChain:
    rows = []
    for k in range(int(spot // 50 * 50) - 500, int(spot // 50 * 50) + 550, 50):
        for ot in ("CE", "PE"):
            sym = f"NIFTY{expiry:%y}X{expiry:%d}{k}{ot}"
            rows.append(ChainRow(k, ot, sym, 200.0, 80000, 600000, 15.0, 0.6 if ot == "CE" else -0.6,
                                 0.0006, -15, 14, 2))
    return OptionChain("NIFTY", expiry, spot, rows, now)


def instruments(expiry: date, spot: float) -> pd.DataFrame:
    rows = []
    for k in range(int(spot // 50 * 50) - 500, int(spot // 50 * 50) + 550, 50):
        for ot in ("CE", "PE"):
            rows.append({"exchange": "NSE", "trading_symbol": f"NIFTY{expiry:%y}X{expiry:%d}{k}{ot}",
                         "groww_symbol": f"NSE-NIFTY-{expiry:%d%b%y}-{k}-{ot}", "instrument_type": ot,
                         "segment": "FNO", "underlying_symbol": "NIFTY", "expiry_date": expiry.isoformat(),
                         "strike_price": k, "lot_size": 65, "tick_size": 0.05, "buy_allowed": 1})
    rows.append({"exchange": "NSE", "trading_symbol": "NIFTYFUT", "groww_symbol": "NSE-NIFTY-FUT",
                 "instrument_type": "FUT", "segment": "FNO", "underlying_symbol": "NIFTY",
                 "expiry_date": (expiry + timedelta(days=14)).isoformat(), "strike_price": None, "lot_size": 65,
                 "tick_size": 0.05, "buy_allowed": 1})
    return pd.DataFrame(rows)
