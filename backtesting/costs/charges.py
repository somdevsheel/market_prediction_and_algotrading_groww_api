"""Indian F&O option-buying transaction charges (rates from config/instruments.yaml; verify periodically)."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ChargeBreakdown:
    brokerage: float
    stt: float
    exchange: float
    sebi: float
    stamp: float
    ipft: float
    gst: float

    @property
    def total(self) -> float:
        return round(sum(asdict(self).values()), 2)

    def to_dict(self) -> dict:
        return {**asdict(self), "total": self.total}


def option_charges(buy_premium_value: float, sell_premium_value: float, cfg: dict,
                   buy_orders: int = 1, sell_orders: int = 1) -> ChargeBreakdown:
    """Charges for a round trip (or one leg: pass 0 value / 0 orders for the other side)."""
    turnover = buy_premium_value + sell_premium_value
    brokerage = cfg["brokerage_per_order"] * (buy_orders * (buy_premium_value > 0) + sell_orders * (sell_premium_value > 0))
    stt = sell_premium_value * cfg["stt_sell_pct"] / 100
    exch = turnover * cfg["exchange_txn_pct"] / 100
    sebi = turnover * cfg["sebi_pct"] / 100
    stamp = buy_premium_value * cfg["stamp_buy_pct"] / 100
    ipft = turnover * cfg.get("ipft_pct", 0) / 100
    gst = (brokerage + exch + sebi + ipft) * cfg["gst_pct"] / 100
    r = lambda x: round(x, 4)  # noqa: E731
    return ChargeBreakdown(r(brokerage), r(stt), r(exch), r(sebi), r(stamp), r(ipft), r(gst))


def estimate_round_trip(entry_price: float, exit_price: float, qty: int, cfg: dict) -> float:
    return option_charges(entry_price * qty, exit_price * qty, cfg).total
