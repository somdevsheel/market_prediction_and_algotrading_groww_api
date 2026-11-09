"""Strategy interface and the structured signal format.

Strategies decide *direction on the underlying* only. They never pick option
contracts, size positions or place orders; the contract selector and the risk
engine do that, and the risk engine can veto anything.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd


@dataclass
class Signal:
    strategy_id: str
    timestamp: datetime            # decision time (close of the last completed candle)
    underlying: str
    direction: str                 # BULLISH | BEARISH
    signal_strength: float
    entry_condition: str
    invalidation_level: float      # underlying level that invalidates the trade
    target_level: float
    reference_price: float         # underlying close at decision
    regime: str
    model_version: str
    reason_codes: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    @property
    def signal_id(self) -> str:
        key = f"{self.strategy_id}|{self.model_version}|{self.timestamp.isoformat()}|{self.direction}"
        return hashlib.sha1(key.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        d["signal_id"] = self.signal_id
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


@dataclass
class MarketContext:
    """Point-in-time market state. Every frame must contain only *completed* candles."""
    now: datetime
    index_5m: pd.DataFrame                    # NIFTY spot, several days, completed candles
    futures_5m: Optional[pd.DataFrame] = None  # near-month futures with real volume
    vix: Optional[float] = None
    regime: Optional[dict] = None
    features: dict = field(default_factory=dict)   # gold daily features known before today's open + ml_prob etc.


class Strategy(ABC):
    strategy_id: str = "base"
    version: str = "0"

    def __init__(self, params: dict):
        self.params = params
        self.last_reasons: list[str] = []   # why the latest evaluation did/didn't signal

    @abstractmethod
    def evaluate(self, ctx: MarketContext) -> Optional[Signal]: ...
