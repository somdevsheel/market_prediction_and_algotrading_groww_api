"""Provider-independent market-data interfaces.

Historical ingestion and live execution are separate concerns; a provider may
implement one or both interfaces but strategies only see these abstractions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Optional

import pandas as pd

from core.models import OptionChain, Quote

CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]


class HistoricalDataProvider(ABC):
    name: str = "base"

    @abstractmethod
    def get_historical_candles(self, instrument: str, start: datetime, end: datetime, interval: str,
                               segment: str = "CASH") -> pd.DataFrame:
        """Return candles with ``CANDLE_COLUMNS``; timestamps tz-aware IST, candle *start* times."""

    @abstractmethod
    def get_instruments(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_expiries(self, underlying: str, year: int, month: Optional[int] = None) -> list[date]: ...

    @abstractmethod
    def get_option_contracts(self, underlying: str, expiry: date) -> list[str]: ...


class LiveDataProvider(ABC):
    @abstractmethod
    def get_quote(self, trading_symbol: str, segment: str) -> Quote: ...

    @abstractmethod
    def get_ltp(self, symbols: list[str], segment: str) -> dict[str, float]: ...

    @abstractmethod
    def get_option_chain(self, underlying: str, expiry: date) -> OptionChain: ...


class ProviderError(RuntimeError):
    pass
