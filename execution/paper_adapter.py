"""Paper broker: simulated fills against *live* quotes from the market-data provider.

Fill rules (documented in docs/BACKTESTING_METHODOLOGY.md):
* LIMIT BUY fills when ask <= limit, at min(limit, ask + slippage).
* LIMIT SELL fills when bid >= limit, at max(limit, bid - slippage).
* SL SELL (stop-limit) triggers when LTP <= trigger, then behaves as a LIMIT SELL.
  A gap through the limit leaves it unfilled (as on the exchange).
Simulated fills are never presented as real fills.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Optional

from core.clock import IST
from core.models import Quote
from execution.broker_interface import BrokerAdapter, BrokerOrderStatus, OrderRequest
from execution.order_state import OrderState as S

log = logging.getLogger(__name__)


class PaperBroker(BrokerAdapter):
    mode = "paper"

    def __init__(self, quote_fn: Callable[[str], Quote], starting_funds: float, slippage_ticks: int = 1,
                 tick: float = 0.05):
        self.quote_fn = quote_fn
        self.funds = starting_funds
        self.slip = slippage_ticks * tick
        self.tick = tick
        self.orders: dict[str, dict] = {}
        self.positions: dict[str, int] = {}

    def _r(self, x: float) -> float:
        return round(round(x / self.tick) * self.tick, 2)

    @staticmethod
    def _levels(q: Quote, side: str) -> list[tuple[float, int]]:
        """Opposite-side book levels [(price, qty)] from quote depth; falls back to top of book."""
        depth = (q.raw or {}).get("depth") or {}
        book = depth.get("sell" if side == "BUY" else "buy") or []
        lv = [(float(x["price"]), int(x.get("quantity") or 0)) for x in book if x and x.get("price")]
        if not lv:
            px = q.ask if side == "BUY" else q.bid
            qty = (q.ask_qty if side == "BUY" else q.bid_qty) or 10 ** 9
            lv = [(px, int(qty))] if px else []
        return lv

    def _try_fill(self, o: dict) -> None:
        if o["state"] not in (S.OPEN, S.PARTIALLY_FILLED):
            return
        req: OrderRequest = o["req"]
        q = self.quote_fn(req.trading_symbol)
        if req.order_type == "SL" and not o.get("triggered"):
            if q.ltp and q.ltp <= req.trigger_price:
                o["triggered"] = True
            else:
                return
        remaining = req.qty - o["filled_qty"]
        fills: list[tuple[int, float]] = []
        for px, avail in self._levels(q, req.side):
            if remaining <= 0:
                break
            if req.order_type != "MARKET":
                if req.side == "BUY" and px > req.price:
                    break
                if req.side == "SELL" and px < req.price:
                    break
            take = min(remaining, avail)
            if take <= 0:
                continue
            fp = px + self.slip if req.side == "BUY" else px - self.slip
            if req.order_type != "MARKET":
                fp = min(req.price, fp) if req.side == "BUY" else max(req.price, fp)
            fills.append((take, self._r(fp)))
            remaining -= take
        if not fills:
            return
        n = sum(f[0] for f in fills)
        prev_n, prev_avg = o["filled_qty"], o["avg_price"] or 0.0
        avg = (prev_avg * prev_n + sum(a * b for a, b in fills)) / (prev_n + n)
        o["filled_qty"] = prev_n + n
        o["avg_price"] = round(avg, 4)
        o["state"] = S.FILLED if o["filled_qty"] >= req.qty else S.PARTIALLY_FILLED
        sign = 1 if req.side == "BUY" else -1
        self.positions[req.trading_symbol] = self.positions.get(req.trading_symbol, 0) + sign * n
        self.funds -= sign * sum(a * b for a, b in fills)
        log.info("PAPER FILL %s %s %d/%d avg %.2f (simulated, depth-walked)", req.side, req.trading_symbol,
                 o["filled_qty"], req.qty, o["avg_price"])

    def place_order(self, req: OrderRequest) -> BrokerOrderStatus:
        if req.order_ref in self.orders:  # idempotent: same ref never creates a second order
            return self._status(self.orders[req.order_ref])
        o = {"req": req, "state": S.OPEN, "filled_qty": 0, "avg_price": None, "broker_order_id": "PAPER-" + req.order_ref}
        self.orders[req.order_ref] = o
        self._try_fill(o)
        return self._status(o)

    def _status(self, o: dict) -> BrokerOrderStatus:
        return BrokerOrderStatus(state=o["state"], filled_qty=o["filled_qty"], avg_price=o["avg_price"],
                                 broker_order_id=o["broker_order_id"], raw_status=o["state"].value)

    def cancel_order(self, order_ref: str, broker_order_id: Optional[str]) -> None:
        o = self.orders.get(order_ref)
        if o and o["state"] in (S.OPEN, S.PARTIALLY_FILLED):
            o["state"] = S.CANCELLED  # filled_qty (if any) is kept, as on the exchange

    def get_order_status(self, order_ref: str, broker_order_id: Optional[str] = None) -> Optional[BrokerOrderStatus]:
        o = self.orders.get(order_ref)
        if o is None:
            return None
        self._try_fill(o)
        return self._status(o)

    def get_net_positions(self) -> dict[str, int]:
        return {k: v for k, v in self.positions.items() if v}

    def get_available_funds(self) -> float:
        return self.funds

    def restore(self, open_trades: list[dict], realized: float, starting: float) -> None:
        """Rebuild state after restart from the ledger (the source of truth)."""
        self.funds = starting + realized
        for t in open_trades:
            if t["status"] in ("OPEN", "EXIT_PENDING") and t.get("entry_price"):
                self.positions[t["trading_symbol"]] = int(t["qty"])
                self.funds -= t["entry_price"] * t["qty"]
