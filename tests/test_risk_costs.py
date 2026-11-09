from datetime import datetime, timedelta

import pytest

from backtesting.costs.charges import option_charges
from core.clock import IST
from core.config import load_config
from risk import pre_trade
from risk.kill_switch import KillSwitch

CFG = load_config()
RISK, CH = CFG["risk"], CFG["instruments"]["charges"]
NOW = datetime(2026, 11, 11, 10, 0, tzinfo=IST)


def test_charges_breakdown():
    c = option_charges(250 * 65, 260 * 65, CH)
    assert c.brokerage == 40
    assert c.stt == pytest.approx(260 * 65 * 0.001)
    assert c.stamp == pytest.approx(250 * 65 * 0.00003)
    assert c.gst == pytest.approx((c.brokerage + c.exchange + c.sebi + c.ipft) * 0.18, abs=1e-3)
    assert 50 < c.total < 80


def test_stop_points_capped_by_budget_and_floor():
    pts, how = pre_trade.compute_stop_points(250, 0.6, 100, 65, RISK, CH)
    assert how == "capped_by_risk_budget"
    charges = option_charges(250 * 65, 250 * 65, CH).total
    assert pts * 65 + charges + RISK["slippage_estimate_ticks"] * 0.05 * 65 <= RISK["risk_per_trade_max"] + 0.05 * 65
    pts2, how2 = pre_trade.compute_stop_points(250, 0.5, 2, 65, RISK, CH)
    assert how2 == "floored_min_stop" and pts2 == RISK["min_stop_points"]
    pts3, how3 = pre_trade.compute_stop_points(250, 0.6, 20, 65, RISK, CH)
    assert how3 == "structural" and pts3 == pytest.approx(12.0)


def plan(**kw):
    base = dict(mode="paper", trading_symbol="X", option_type="CE", lot_size=65, lots=1, entry_price=250.0,
                stop_price=238.0, target_price=274.0, quote_time=NOW)
    base.update(kw)
    return pre_trade.TradePlan(**base)


def day(**kw):
    base = dict(trades_today=0, realized_pnl_today=0.0, open_positions=0, pending_orders=0, reconciliation_required=False)
    base.update(kw)
    return pre_trade.DayState(**base)


def ev(p=None, d=None, funds=20000.0, kill=False, window=True, live_ok=False):
    return pre_trade.evaluate(p or plan(), NOW, d or day(), funds, kill, window, RISK, CH, live_ok)


def test_approves_good_trade():
    r = ev()
    assert r.approved, r.failures
    assert r.worst_case_loss > 250 * 65  # full premium + charges


@pytest.mark.parametrize("kwargs,fail", [
    ({"d": day(trades_today=1)}, "daily_trade_count"),
    ({"d": day(open_positions=1)}, "no_existing_position"),
    ({"d": day(realized_pnl_today=-2000)}, "daily_loss_limit"),
    ({"d": day(reconciliation_required=True)}, "no_reconciliation_pending"),
    ({"kill": True}, "kill_switch_off"),
    ({"window": False}, "entry_window"),
    ({"funds": 600.0}, "sufficient_funds"),
    ({"p": plan(lots=2)}, "max_lots"),
    ({"p": plan(stop_price=200.0)}, "planned_loss_within_budget"),
    ({"p": plan(entry_price=400.0, stop_price=390.0, target_price=420.0)}, "premium_outlay_limit"),
    ({"p": plan(quote_time=NOW - timedelta(minutes=2))}, "quote_fresh"),
    ({"p": plan(mode="live")}, "live_approved"),
])
def test_rejections(kwargs, fail):
    r = ev(**kwargs)
    assert not r.approved
    assert any(f.startswith(fail) for f in r.failures), r.failures


def test_kill_switch(tmp_path):
    k = KillSwitch(tmp_path / "KILL")
    assert not k.active()
    k.activate("test")
    assert k.active() and k.info()["reason"] == "test"
    k.deactivate()
    assert not k.active()
