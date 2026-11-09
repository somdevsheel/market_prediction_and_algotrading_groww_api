"""Live Groww execution adapter.

* Uses ``order_reference_id`` as idempotency key so status can be recovered
  after a timeout without resubmitting.
* Order placement is NEVER auto-retried: an exception is treated as ambiguous
  and resolved by reconciliation (status by reference) before any further action.
* Fill state is derived from ``filled_quantity`` (documented field), not only
  from the status string. Unknown status strings map to UNKNOWN.
"""
from __future__ import annotations

import logging
from typing import Optional

from core.ratelimit import RateLimiter
from execution.broker_interface import BrokerAdapter, BrokerError, BrokerOrderStatus, OrderRequest
from execution.order_state import GROWW_STATUS_MAP, OrderState as S

log = logging.getLogger(__name__)


class GrowwBroker(BrokerAdapter):
    mode = "live"

    def __init__(self, client, settings: dict, segment: str = "FNO"):
        self.client = client
        self.segment = segment
        rl = settings["rate_limits"]
        self.order_rl = RateLimiter(rl["orders_per_sec"], rl["orders_per_min"])
        self.nt_rl = RateLimiter(rl["non_trading_per_sec"], rl["non_trading_per_min"])
        self.timeout = 10

    def place_order(self, req: OrderRequest) -> BrokerOrderStatus:
        self.order_rl.acquire()
        try:
            r = self.client.place_order(
                validity=req.validity, exchange=req.exchange, order_type=req.order_type, product=req.product,
                quantity=req.qty, segment=req.segment, trading_symbol=req.trading_symbol,
                transaction_type=req.side, order_reference_id=req.order_ref, price=req.price,
                trigger_price=req.trigger_price, timeout=self.timeout)
        except Exception as e:  # noqa: BLE001
            name = type(e).__name__
            # 400/401/403 responses mean the request was refused; anything else may have reached the exchange
            ambiguous = name not in ("GrowwAPIBadRequestException", "GrowwAPIAuthenticationException",
                                     "GrowwAPIAuthorisationException") and not (
                name == "GrowwAPIException" and getattr(e, "code", "") not in ("", None, "500", "502", "503", "504"))
            raise BrokerError(f"place_order failed: {name}: {e}", ambiguous=ambiguous) from e
        return self._map(r, req.qty)

    def _map(self, r: dict, qty: Optional[int] = None) -> BrokerOrderStatus:
        raw = str(r.get("order_status") or "").upper()
        state = GROWW_STATUS_MAP.get(raw, S.UNKNOWN)
        filled = int(r.get("filled_quantity") or 0)
        if qty:
            if filled >= qty and filled > 0:
                state = S.FILLED
            elif 0 < filled < qty and state not in (S.CANCELLED, S.REJECTED, S.EXPIRED):
                state = S.PARTIALLY_FILLED
            elif state == S.FILLED and filled == 0 and "filled_quantity" in r:
                state = S.UNKNOWN  # status says filled but quantity says otherwise -> reconcile
        return BrokerOrderStatus(state=state, filled_qty=filled, avg_price=r.get("average_fill_price"),
                                 broker_order_id=r.get("groww_order_id"), raw_status=raw, raw=r)

    def cancel_order(self, order_ref: str, broker_order_id: Optional[str]) -> None:
        if not broker_order_id:
            st = self.get_order_status(order_ref)
            broker_order_id = st.broker_order_id if st else None
        if not broker_order_id:
            raise BrokerError("cannot cancel: broker order id unknown")
        self.order_rl.acquire()
        self.client.cancel_order(groww_order_id=broker_order_id, segment=self.segment, timeout=self.timeout)

    def get_order_status(self, order_ref: str, broker_order_id: Optional[str] = None,
                         qty: Optional[int] = None) -> Optional[BrokerOrderStatus]:
        self.nt_rl.acquire()
        try:
            r = self.client.get_order_status_by_reference(segment=self.segment, order_reference_id=order_ref,
                                                          timeout=self.timeout)
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ == "GrowwAPINotFoundException":
                return None
            raise BrokerError(f"status failed: {type(e).__name__}: {e}") from e
        st = self._map(r)
        if st.broker_order_id and st.filled_qty > 0:
            try:  # average_fill_price is only in order detail
                self.nt_rl.acquire()
                d = self.client.get_order_detail(segment=self.segment, groww_order_id=st.broker_order_id,
                                                 timeout=self.timeout)
                st.avg_price = d.get("average_fill_price") or st.avg_price
                q = d.get("quantity")
                st2 = self._map({**r, **d}, int(q) if q else None)
                st.state = st2.state
            except Exception as e:  # noqa: BLE001
                log.warning("order detail failed for %s: %s", order_ref, e)
        return st

    def get_net_positions(self) -> dict[str, int]:
        self.nt_rl.acquire()
        r = self.client.get_positions_for_user(segment=self.segment, timeout=self.timeout)
        return {p["trading_symbol"]: int(p.get("quantity") or 0) for p in r.get("positions", [])
                if int(p.get("quantity") or 0) != 0}

    def get_available_funds(self) -> float:
        self.nt_rl.acquire()
        r = self.client.get_available_margin_details(timeout=self.timeout)
        return float((r.get("fno_margin_details") or {}).get("option_buy_balance_available") or 0.0)
