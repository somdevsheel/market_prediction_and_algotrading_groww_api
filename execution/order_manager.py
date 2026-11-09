"""Order manager: persists every order transition and prevents duplicates.

Rules:
* Every order gets a deterministic ``order_ref`` derived from (trade_id, role,
  attempt), so a crash/restart can never produce a second order for the same intent.
* A failed/timeout ``place_order`` is NOT retried. The order goes to UNKNOWN and is
  reconciled by reference. If the broker cannot confirm either way it becomes
  RECONCILIATION_REQUIRED and the caller must block trading.
"""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import date
from typing import Callable, Optional

from execution.broker_interface import BrokerAdapter, BrokerError, BrokerOrderStatus, OrderRequest
from execution.order_state import TERMINAL, OrderState as S, check_transition
from ledger.store import Ledger

log = logging.getLogger(__name__)


def make_order_ref(trade_id: str, role: str, attempt: int = 0) -> str:
    """8-20 alphanumeric chars (Groww order_reference_id constraint)."""
    h = hashlib.sha1(f"{trade_id}|{role}|{attempt}".encode()).hexdigest()[:10].upper()
    return f"NQ{role[0].upper()}{attempt % 10}{h}"   # 14 chars


class OrderManager:
    def __init__(self, broker: BrokerAdapter, ledger: Ledger, settings: dict, sleep: Callable = time.sleep):
        self.broker = broker
        self.ledger = ledger
        ex = settings["execution"]
        self.reconcile_retries = ex.get("reconcile_retries", 6)
        self.reconcile_wait = ex.get("reconcile_wait_seconds", 2)
        self.sleep = sleep

    def _transition(self, order_id: str, new: S, detail: dict | None = None, **fields) -> None:
        cur = self.ledger.get_order(order_id)
        old = S(cur["state"])
        if old == new and not fields:
            return
        check_transition(old, new)
        self.ledger.update_order(order_id, new.value, detail, **fields)

    def _apply(self, order_id: str, st: BrokerOrderStatus) -> dict:
        cur = self.ledger.get_order(order_id)
        old = S(cur["state"])
        fields = {"filled_qty": st.filled_qty}
        if st.avg_price is not None:
            fields["avg_price"] = float(st.avg_price)
        if st.broker_order_id:
            fields["broker_order_id"] = st.broker_order_id
        if old in TERMINAL:
            if old != st.state:
                log.error("broker reports %s for terminal order %s (%s)", st.state, order_id, old)
            return self.ledger.get_order(order_id)
        if old == S.SUBMITTED or st.state != old or fields.get("filled_qty") != cur["filled_qty"]:
            try:
                self._transition(order_id, st.state, {"raw_status": st.raw_status}, **fields)
            except Exception as e:  # invalid transition -> needs a human
                log.error("invalid transition for %s: %s", order_id, e)
                self.ledger.update_order(order_id, S.RECONCILIATION_REQUIRED.value, {"error": str(e)})
        return self.ledger.get_order(order_id)

    def submit(self, req: OrderRequest, *, trade_id: str, signal_id: Optional[str], role: str, mode: str,
               trading_day: date) -> dict:
        existing = self.ledger.get_order(req.order_ref)
        if existing:  # duplicate prevention: never submit the same intent twice
            log.warning("order %s already exists in state %s; not resubmitting", req.order_ref, existing["state"])
            return self.refresh(req.order_ref)
        self.ledger.insert_order({"order_id": req.order_ref, "trade_id": trade_id, "signal_id": signal_id,
                                  "role": role, "mode": mode, "trading_symbol": req.trading_symbol, "side": req.side,
                                  "qty": req.qty, "order_type": req.order_type, "price": req.price,
                                  "trigger_price": req.trigger_price, "state": S.CREATED.value,
                                  "trading_day": trading_day.isoformat()})
        if req.qty <= 0 or req.side not in ("BUY", "SELL"):
            self._transition(req.order_ref, S.REJECTED, {"reason": "invalid_request"})
            return self.ledger.get_order(req.order_ref)
        self._transition(req.order_ref, S.VALIDATED)
        self._transition(req.order_ref, S.SUBMITTED)
        try:
            st = self.broker.place_order(req)
        except BrokerError as e:
            self.ledger.record_error("order_manager", f"place_order {req.order_ref}: {e}")
            if not e.ambiguous:
                self._transition(req.order_ref, S.REJECTED, {"error": str(e)}, last_error=str(e)[:500])
                return self.ledger.get_order(req.order_ref)
            self._transition(req.order_ref, S.UNKNOWN, {"error": str(e)}, last_error=str(e)[:500])
            return self.reconcile(req.order_ref)
        return self._apply(req.order_ref, st)

    def reconcile(self, order_id: str) -> dict:
        """Resolve an UNKNOWN order by asking the broker by reference. Never resubmits."""
        for i in range(self.reconcile_retries):
            try:
                st = self.broker.get_order_status(order_id)
            except BrokerError as e:
                log.warning("reconcile %s attempt %d failed: %s", order_id, i + 1, e)
                st = "error"
            if st is not None and st != "error":
                return self._apply(order_id, st)
            self.sleep(self.reconcile_wait)
        cur = self.ledger.get_order(order_id)
        if S(cur["state"]) in (S.UNKNOWN, S.SUBMITTED):
            self.ledger.update_order(order_id, S.RECONCILIATION_REQUIRED.value,
                                     {"reason": "broker could not confirm order existence"})
        return self.ledger.get_order(order_id)

    def refresh(self, order_id: str) -> dict:
        cur = self.ledger.get_order(order_id)
        if S(cur["state"]) in TERMINAL:
            return cur
        try:
            st = self.broker.get_order_status(order_id, cur.get("broker_order_id"))
        except BrokerError as e:
            log.warning("refresh %s failed: %s", order_id, e)
            return cur
        if st is None:
            return self.reconcile(order_id) if S(cur["state"]) in (S.UNKNOWN, S.SUBMITTED) else cur
        return self._apply(order_id, st)

    def cancel(self, order_id: str) -> dict:
        cur = self.refresh(order_id)
        if S(cur["state"]) in TERMINAL:
            return cur
        try:
            self.broker.cancel_order(order_id, cur.get("broker_order_id"))
        except Exception as e:  # noqa: BLE001
            log.warning("cancel %s failed: %s", order_id, e)
            self.ledger.record_error("order_manager", f"cancel {order_id}: {e}")
        for _ in range(self.reconcile_retries):
            cur = self.refresh(order_id)
            if S(cur["state"]) in TERMINAL:
                return cur
            self.sleep(self.reconcile_wait)
        return cur

    def wait_terminal(self, order_id: str, timeout_s: float, poll_s: float = 1.0) -> dict:
        waited = 0.0
        cur = self.refresh(order_id)
        while S(cur["state"]) not in TERMINAL and waited < timeout_s:
            self.sleep(poll_s)
            waited += poll_s
            cur = self.refresh(order_id)
        return cur
