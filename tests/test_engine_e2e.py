"""End-to-end paper trading on SYNTHETIC data with a frozen clock."""
import copy
from datetime import date, datetime, time, timedelta

import pandas as pd
import pytest

from apps.worker.engine import TradingEngine
from core.clock import IST, FrozenClock
from core.config import load_config
from execution.paper_adapter import PaperBroker
from ledger.store import Ledger
from risk.kill_switch import KillSwitch
from tests import synthetic as syn

D = date(2026, 11, 11)
EXP = date(2026, 11, 17)


class FakeProvider:
    name = "fake"

    def __init__(self, clock):
        self.clock = clock
        self.idx, self.fut = syn.breakout_day(D)
        self.inst = syn.instruments(EXP, 25044.0)
        self.px = {"bid": 250.0, "ask": 250.5}
        self.calls = 0

    def get_historical_candles(self, symbol, start, end, interval, segment="CASH"):
        self.calls += 1
        df = self.idx if symbol == "NSE-NIFTY" else self.fut
        return df[(df.timestamp >= pd.Timestamp(start)) & (df.timestamp <= pd.Timestamp(end))].reset_index(drop=True)

    def futures_contracts(self, u):
        f = self.inst[self.inst.instrument_type == "FUT"].copy()
        f["expiry_date"] = pd.to_datetime(f["expiry_date"]).dt.date
        return f

    def get_ltp(self, syms, seg):
        return {"NSE_INDIAVIX": 14.0}

    def get_expiries(self, u, y, m):
        return [EXP] if m == 11 else []

    def get_option_chain(self, u, e):
        return syn.chain(e, 25080.0, self.clock.now())

    def get_quote(self, s, seg):
        return syn.quote(s, self.px["bid"], self.px["ask"], self.clock.now())

    def get_instruments(self):
        return self.inst


@pytest.fixture
def env(tmp_path):
    cfg = copy.deepcopy(load_config())
    from tests.fake_strategy import install

    install(cfg, "09:45")  # signals at the 09:45 candle close (09:50), like the original breakout fixture
    cfg["settings"]["paths"]["kill_switch_file"] = str(tmp_path / "KILL")
    clock = FrozenClock(datetime(2026, 11, 11, 9, 0, tzinfo=IST))
    prov = FakeProvider(clock)
    led = Ledger(tmp_path / "l.sqlite3")
    kill = KillSwitch(tmp_path / "KILL")

    def make():
        b = PaperBroker(lambda s: prov.get_quote(s, "FNO"), 20000, 1)
        e = TradingEngine(cfg, prov, b, led, clock, kill, sleep=lambda s: None)
        e.startup()
        return e

    return cfg, clock, prov, led, kill, make


def run_until(e, clock, hhmm):
    h, m = map(int, hhmm.split(":"))
    while clock.now().time() < time(h, m):
        e.step()
        clock.advance(seconds=30)


def test_full_paper_trade_stop_loss(env):
    cfg, clock, prov, led, kill, make = env
    e = make()
    run_until(e, clock, "09:49")
    assert e.trade is None
    run_until(e, clock, "09:51")  # 09:45 candle completed at 09:50:05 -> signal -> entry
    assert e.trade is not None, led.query("SELECT payload FROM risk_decisions")
    t = e.trade
    assert t["status"] == "OPEN" and t["option_type"] == "CE" and t["strike"] == 25050 and t["qty"] == 65
    assert t["entry_price"] == pytest.approx(250.55)  # ask + 1 tick paper slippage
    roles = {o["role"]: o["state"] for o in led.query("SELECT * FROM orders")}
    assert roles == {"ENTRY": "FILLED", "STOP": "OPEN"}
    prov.px.update(bid=t["planned_stop"] - 0.5, ask=t["planned_stop"])
    e.step()
    assert e.trade is None
    closed = led.trades("paper")[0]
    assert closed["status"] == "CLOSED" and closed["exit_reason"].startswith("stop_loss")
    assert closed["net_pnl"] == pytest.approx(closed["gross_pnl"] - closed["charges"])
    assert closed["net_pnl"] < 0 and abs(closed["net_pnl"]) <= cfg["risk"]["risk_per_trade_max"] + 100
    assert led.query("SELECT state FROM orders WHERE role='STOP'")[0]["state"] in ("CANCELLED", "FILLED")
    # no second trade on the same day
    prov.px.update(bid=250.0, ask=250.5)
    run_until(e, clock, "11:00")
    assert len(led.trades("paper")) == 1


def test_target_and_restart_recovery(env):
    cfg, clock, prov, led, kill, make = env
    e = make()
    run_until(e, clock, "09:51")
    tid = e.trade["trade_id"]
    e2 = make()  # simulated crash + restart
    assert e2.trade is not None and e2.trade["trade_id"] == tid
    prov.px.update(bid=e2.trade["target"] + 1, ask=e2.trade["target"] + 1.5)
    e2.step()
    t = led.get_trade(tid)
    assert t["status"] == "CLOSED" and t["exit_reason"] == "target" and t["net_pnl"] > 0


def test_time_exit_and_kill_switch(env):
    cfg, clock, prov, led, kill, make = env
    e = make()
    run_until(e, clock, "09:51")
    assert e.trade
    kill.activate("test")
    e.step()
    assert e.trade is None and led.trades("paper")[0]["exit_reason"] == "kill_switch"


def test_square_off(env):
    cfg, clock, prov, led, kill, make = env
    e = make()
    run_until(e, clock, "09:51")
    prov.idx = prov.idx.iloc[:7]  # no invalidation candles
    clock.dt = datetime(2026, 11, 11, 15, 15, 1, tzinfo=IST)
    e.step()
    assert led.trades("paper")[0]["exit_reason"] == "time_exit"


def test_previous_day_open_trade_blocks(env):
    cfg, clock, prov, led, kill, make = env
    led.insert_trade({"trade_id": "OLD", "mode": "paper", "trading_day": "2026-11-09", "trading_symbol": "X",
                      "qty": 65, "status": "OPEN", "entry_price": 100.0, "details": {}})
    e = make()
    assert kill.active() and e.blocked_reason
    run_until(e, clock, "09:52")
    assert led.get_trade("OLD")["status"] == "RECONCILIATION_REQUIRED"
    assert len(led.trades("paper")) == 1


def test_insufficient_funds_rejected(env):
    cfg, clock, prov, led, kill, make = env
    e = make()
    e.broker.funds = 600.0
    run_until(e, clock, "09:52")
    assert e.trade is None
    d = led.query("SELECT approved, payload FROM risk_decisions")
    assert d and d[-1]["approved"] == 0 and "sufficient_funds" in d[-1]["payload"]


def test_holiday_no_activity(env):
    cfg, clock, prov, led, kill, make = env
    clock.dt = datetime(2026, 11, 10, 10, 0, tzinfo=IST)
    e = make()
    e.step()
    assert e.done_for_day and prov.calls == 0


def test_live_locked_by_default(env, tmp_path):
    cfg, clock, prov, led, kill, make = env

    class LiveB(PaperBroker):
        mode = "live"

    b = LiveB(lambda s: prov.get_quote(s, "FNO"), 0, 1)
    e = TradingEngine(cfg, prov, b, led, clock, kill)
    with pytest.raises(RuntimeError, match="LIVE TRADING LOCKED"):
        e.startup()


def test_transient_data_failure_does_not_end_day(env):
    cfg, clock, prov, led, kill, make = env
    e = make()
    prov.px.update(bid=1.0, ask=1.05)  # premium out of bounds -> no trade, keep evaluating
    run_until(e, clock, "09:52")
    assert e.last_decision_candle is not None
    good = prov.idx
    prov.idx = good.iloc[0:0]  # simulate API returning nothing
    clock.dt = datetime(2026, 11, 11, 10, 0, 10, tzinfo=IST)
    e.step()
    assert not e.done_for_day
    prov.idx = good
    e.step()
    assert not e.done_for_day


def test_rate_limit_message_is_retryable():
    from ingestion.providers.groww import _retryable

    class GrowwAPIException(Exception):
        pass

    assert _retryable(GrowwAPIException("Rate limit has breached for your request. Please try again later."))
    assert not _retryable(GrowwAPIException("Invalid symbol"))
