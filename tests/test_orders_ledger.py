from datetime import date, datetime

import pytest

from core.clock import IST
from core.config import load_config
from execution.broker_interface import BrokerAdapter, BrokerError, BrokerOrderStatus, OrderRequest
from execution.groww_adapter import GrowwBroker
from execution.order_manager import OrderManager, make_order_ref
from execution.order_state import InvalidTransition, OrderState as S, check_transition
from execution.paper_adapter import PaperBroker
from ledger.store import Ledger
from tests import synthetic as syn

CFG = load_config()
D = date(2026, 11, 11)
NOW = datetime(2026, 11, 11, 10, 0, tzinfo=IST)


def test_state_machine():
    check_transition(S.CREATED, S.VALIDATED)
    check_transition(S.SUBMITTED, S.FILLED)
    with pytest.raises(InvalidTransition):
        check_transition(S.FILLED, S.OPEN)
    with pytest.raises(InvalidTransition):
        check_transition(S.CREATED, S.FILLED)


def test_order_ref_format():
    r = make_order_ref("20261111-abcdef12", "ENTRY")
    assert 8 <= len(r) <= 20 and r.isalnum()
    assert r == make_order_ref("20261111-abcdef12", "ENTRY")
    assert r != make_order_ref("20261111-abcdef12", "EXIT")


class FlakyBroker(BrokerAdapter):
    """Accepts the order but the response is lost (timeout); later reports it filled."""
    mode = "live"

    def __init__(self, exists=True):
        self.calls = 0
        self.exists = exists

    def place_order(self, req):
        self.calls += 1
        raise BrokerError("timeout", ambiguous=True)

    def cancel_order(self, ref, bid):
        pass

    def get_order_status(self, ref, bid=None):
        return BrokerOrderStatus(S.FILLED, 65, 250.0, "G1", "EXECUTED") if self.exists else None

    def get_net_positions(self):
        return {}

    def get_available_funds(self):
        return 0


def req(ref="NQE0ABCDEF1234"):
    return OrderRequest(ref, "SYM", "BUY", 65, "LIMIT", 250.0)


def test_ambiguous_failure_reconciles_without_resubmitting():
    led, b = Ledger(":memory:"), FlakyBroker(True)
    om = OrderManager(b, led, CFG["settings"], sleep=lambda s: None)
    o = om.submit(req(), trade_id="T", signal_id="S", role="ENTRY", mode="live", trading_day=D)
    assert o["state"] == "FILLED" and o["avg_price"] == 250.0
    assert b.calls == 1
    o2 = om.submit(req(), trade_id="T", signal_id="S", role="ENTRY", mode="live", trading_day=D)  # duplicate intent
    assert b.calls == 1 and o2["order_id"] == o["order_id"]
    states = [e["to_state"] for e in led.query("SELECT to_state FROM order_events ORDER BY id")]
    assert states == ["CREATED", "VALIDATED", "SUBMITTED", "UNKNOWN", "FILLED"]


def test_unconfirmable_order_requires_reconciliation():
    led, b = Ledger(":memory:"), FlakyBroker(False)
    om = OrderManager(b, led, CFG["settings"], sleep=lambda s: None)
    o = om.submit(req(), trade_id="T", signal_id="S", role="ENTRY", mode="live", trading_day=D)
    assert o["state"] == "RECONCILIATION_REQUIRED"
    assert b.calls == 1


def test_paper_fills_and_stop_limit():
    px = {"bid": 250.0, "ask": 251.0}
    pb = PaperBroker(lambda s: syn.quote(s, px["bid"], px["ask"], NOW), 20000, 1)
    st = pb.place_order(OrderRequest("NQE0AAAAAAAAAA", "SYM", "BUY", 65, "LIMIT", 252.0))
    assert st.state == S.FILLED and st.avg_price == pytest.approx(251.05)
    assert pb.get_net_positions() == {"SYM": 65}
    sl = pb.place_order(OrderRequest("NQS0AAAAAAAAAA", "SYM", "SELL", 65, "SL", 238.0, 240.0))
    assert sl.state == S.OPEN
    px.update(bid=239.0, ask=239.5)  # ltp 239.25 <= trigger, bid >= limit -> fill
    st = pb.get_order_status("NQS0AAAAAAAAAA")
    assert st.state == S.FILLED and st.avg_price == pytest.approx(238.95)
    assert pb.get_net_positions() == {}


def test_paper_stop_limit_gap_leaves_unfilled():
    px = {"bid": 250.0, "ask": 251.0}
    pb = PaperBroker(lambda s: syn.quote(s, px["bid"], px["ask"], NOW), 20000, 1)
    pb.place_order(OrderRequest("NQS0BBBBBBBBBB", "SYM", "SELL", 65, "SL", 238.0, 240.0))
    px.update(bid=230.0, ask=231.0)
    assert pb.get_order_status("NQS0BBBBBBBBBB").state == S.OPEN


class FakeGroww:
    def __init__(self, status):
        self.status = status

    def get_order_status_by_reference(self, **kw):
        return self.status

    def get_order_detail(self, **kw):
        return {**self.status, "average_fill_price": 251.5, "quantity": 65}


def test_groww_status_mapping_uses_filled_quantity():
    g = GrowwBroker(FakeGroww({"groww_order_id": "G", "order_status": "EXECUTED", "filled_quantity": 65}), CFG["settings"])
    st = g.get_order_status("REF12345")
    assert st.state == S.FILLED and st.avg_price == 251.5
    g = GrowwBroker(FakeGroww({"groww_order_id": "G", "order_status": "OPEN", "filled_quantity": 30}), CFG["settings"])
    assert g.get_order_status("REF12345").state == S.PARTIALLY_FILLED
    g = GrowwBroker(FakeGroww({"groww_order_id": "G", "order_status": "WEIRD", "filled_quantity": 0}), CFG["settings"])
    assert g.get_order_status("REF12345").state == S.UNKNOWN


def test_ledger_persists_across_reopen(tmp_path):
    p = tmp_path / "l.sqlite3"
    led = Ledger(p)
    led.insert_trade({"trade_id": "T1", "mode": "paper", "trading_day": "2026-11-11", "trading_symbol": "X",
                      "qty": 65, "status": "OPEN", "entry_price": 250.0, "details": {"a": 1}})
    led.conn.close()
    led2 = Ledger(p)
    assert [t["trade_id"] for t in led2.open_trades()] == ["T1"]
    assert led2.day_stats(D, "paper")["open_positions"] == 1
