"""Engine end-to-end with the 1-minute ORB retest + cheap-IV strategy (SYNTHETIC data)."""
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

D, EXP = date(2026, 11, 11), date(2026, 11, 17)


def one_min_day():
    # 30-min OR between ~24990 and ~25040 (0.2%), break up at 09:50, back inside 09:55, break again 10:00
    path = [25000 + (i % 10) * 4 for i in range(30)]                      # OR 09:15-09:44
    path += [25020, 25030, 25035, 25040, 25045, 25060, 25065]            # 09:45-09:51 (close>H at 09:50)
    path += [25035, 25030, 25035]                                        # back inside
    path += [25070, 25080, 25090] + [25100] * 300
    rows, prev = [], path[0]
    for i, c in enumerate(path[:375]):
        ts = datetime.combine(D, time(9, 15), IST) + timedelta(minutes=i)
        rows.append([ts, prev, max(prev, c) + 1, min(prev, c) - 1, c, 1000.0, None])
        prev = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "open_interest"])


class Prov:
    name = "fake"

    def __init__(self, clock, iv):
        self.clock, self.iv = clock, iv
        self.idx = one_min_day()
        self.inst = syn.instruments(EXP, 25044.0)
        self.px = {"bid": 250.0, "ask": 250.5}
        self.client = type("C", (), {"get_greeks": lambda *a, **k: {"greeks": {"delta": 0.6}}})()

    def get_historical_candles(self, symbol, start, end, interval, segment="CASH"):
        df = self.idx
        return df[(df.timestamp >= pd.Timestamp(start)) & (df.timestamp <= pd.Timestamp(end))].reset_index(drop=True)

    def futures_contracts(self, u):
        f = self.inst[self.inst.instrument_type == "FUT"].copy()
        f["expiry_date"] = pd.to_datetime(f["expiry_date"]).dt.date
        return f

    def get_ltp(self, s, seg):
        return {"NSE_INDIAVIX": 14.0}

    def get_expiries(self, u, y, m):
        return [EXP] if m == 11 else []

    def get_option_chain(self, u, e):
        ch = syn.chain(e, 25090.0, self.clock.now())
        for r in ch.rows:
            r.iv = self.iv
        return ch

    def get_quote(self, s, seg):
        return syn.quote(s, self.px["bid"], self.px["ask"], self.clock.now())

    def get_instruments(self):
        return self.inst


def make(tmp_path, iv):
    cfg = copy.deepcopy(load_config())
    cfg["strategies"]["active_strategy"] = "orb_retest30_cheapiv_v1"
    clock = FrozenClock(datetime(2026, 11, 11, 9, 40, tzinfo=IST))
    prov = Prov(clock, iv)
    led = Ledger(tmp_path / "l.sqlite3")
    e = TradingEngine(cfg, prov, PaperBroker(lambda s: prov.get_quote(s, "FNO"), 20000, 1), led, clock,
                      KillSwitch(tmp_path / "K"), sleep=lambda s: None)
    e.features_table = pd.DataFrame([{"date": pd.Timestamp(D), "garch_fc": 15.0}])
    e.risk_cfg["max_premium_outlay"] = 20000
    e.startup()
    return e, clock, prov, led


def run(e, clock, hhmm):
    h, m = map(int, hhmm.split(":"))
    while clock.now().time() < time(h, m):
        e.step()
        clock.advance(seconds=20)


def test_rich_iv_rejected(tmp_path):
    e, clock, prov, led = make(tmp_path, iv=19.0)  # 19/15 = 1.27 > 1.0
    run(e, clock, "10:10")
    assert e.trade is None
    d = led.query("SELECT payload FROM risk_decisions")
    assert d and "contract_iv_filter" in d[-1]["payload"]


def test_cheap_iv_trades_and_underlying_stop(tmp_path):
    e, clock, prov, led = make(tmp_path, iv=12.0)
    run(e, clock, "10:02")
    assert e.trade is not None, led.query("SELECT payload FROM risk_decisions")
    t = e.trade
    assert t["option_type"] == "CE" and t["strategy_id"] == "orb_retest30_cheapiv_v1"
    import json

    assert json.loads(t["details"])["signal"]["diagnostics"]["stop_mode"] == "touch"
    # underlying falls through the OR low (stop) -> exit on touch
    lo = json.loads(t["details"])["invalidation_level"]
    k = len(prov.idx[prov.idx.timestamp < pd.Timestamp(clock.now())])
    prov.idx.loc[k:, ["open", "high", "low", "close"]] = [lo - 5, lo - 4, lo - 10, lo - 6]
    run(e, clock, "10:06")
    assert e.trade is None and led.trades("paper")[0]["exit_reason"] == "underlying_stop"
