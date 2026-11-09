"""Broker-independent execution interface. Paper and live differ only here."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from execution.order_state import OrderState


@dataclass
class OrderRequest:
    order_ref: str              # our idempotency key (8-20 alnum, <=2 hyphens; Groww order_reference_id)
    trading_symbol: str
    side: str                   # BUY | SELL
    qty: int
    order_type: str             # LIMIT | SL | MARKET
    price: float = 0.0
    trigger_price: Optional[float] = None
    product: str = "MIS"
    validity: str = "DAY"
    segment: str = "FNO"
    exchange: str = "NSE"


@dataclass
class BrokerOrderStatus:
    state: OrderState
    filled_qty: int = 0
    avg_price: Optional[float] = None
    broker_order_id: Optional[str] = None
    raw_status: Optional[str] = None
    raw: dict = field(default_factory=dict, repr=False)


class BrokerError(RuntimeError):
    """Raised when a broker call failed. ``ambiguous=True`` means the order MAY have reached the exchange."""

    def __init__(self, msg: str, ambiguous: bool = True):
        super().__init__(msg)
        self.ambiguous = ambiguous


class BrokerAdapter(ABC):
    mode: str = "abstract"

    @abstractmethod
    def place_order(self, req: OrderRequest) -> BrokerOrderStatus: ...

    @abstractmethod
    def cancel_order(self, order_ref: str, broker_order_id: Optional[str]) -> None: ...

    @abstractmethod
    def get_order_status(self, order_ref: str, broker_order_id: Optional[str] = None) -> Optional[BrokerOrderStatus]:
        """Return None only when the broker positively reports the order does not exist."""

    @abstractmethod
    def get_net_positions(self) -> dict[str, int]:
        """trading_symbol -> net quantity (non-zero only)."""

    @abstractmethod
    def get_available_funds(self) -> float: ...
