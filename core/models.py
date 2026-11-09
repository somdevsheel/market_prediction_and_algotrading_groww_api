"""Shared, broker-independent data models."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Optional


@dataclass
class Quote:
    trading_symbol: str
    ltp: float
    bid: Optional[float]
    ask: Optional[float]
    bid_qty: Optional[int] = None
    ask_qty: Optional[int] = None
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    last_trade_time: Optional[datetime] = None
    received_at: Optional[datetime] = None
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def mid(self) -> Optional[float]:
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return None

    @property
    def spread_pct(self) -> Optional[float]:
        m = self.mid
        if m and m > 0:
            return (self.ask - self.bid) / m * 100
        return None


@dataclass
class OptionContract:
    underlying: str
    trading_symbol: str
    groww_symbol: Optional[str]
    expiry: date
    strike: float
    option_type: str          # CE | PE
    lot_size: int
    tick_size: float = 0.05
    exchange: str = "NSE"
    segment: str = "FNO"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["expiry"] = self.expiry.isoformat()
        return d


@dataclass
class ChainRow:
    strike: float
    option_type: str
    trading_symbol: str
    ltp: float
    open_interest: float
    volume: float
    iv: Optional[float]
    delta: Optional[float]
    gamma: Optional[float]
    theta: Optional[float]
    vega: Optional[float]
    rho: Optional[float]


@dataclass
class OptionChain:
    underlying: str
    expiry: date
    underlying_ltp: float
    rows: list[ChainRow]
    fetched_at: datetime

    def get(self, strike: float, option_type: str) -> Optional[ChainRow]:
        for r in self.rows:
            if r.strike == strike and r.option_type == option_type:
                return r
        return None
