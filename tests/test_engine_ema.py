"""Engine end-to-end for the 9/21 EMA crossover: entry, opposite-cross exit, same-bar reversal, 15:00 square-off."""
import copy
import json
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from apps.worker.engine import TradingEngine
from core.clock import IST, FrozenClock
from core.config import load_config
from execution.paper_adapter import PaperBroker
from ledger.store import Ledger
from risk.kill_switch import KillSwitch
from tests import synthetic as syn
from tests.test_engine_orb import Prov

D, EXP = date(2026, 11, 11), date(2026, 11, 17)


def path():
    rows = []
    for d, px in ((date(2026, 11, 9), np.linspace(25200, 25000, 375)),
                  (D, np.r_[np.linspace(25000, 24950, 75), np.linspace(24950, 25250, 90), np.linspace(25250, 24850, 90), np.full(120, 24850.0)])):
        prev = px[0]
        for i, c in enumerate(px):
            ts = datetime.combine(d, time(9, 15), IST) + timedelta(minutes=i)
            rows.append([ts, prev, max(prev, c) + 1, min(prev, c) - 1, c, 0.0, None])
            prev = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "open_interest"])


def test_ema_entry_exit_reversal_squareoff(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["strategies"]["active_strategy"] = "ema921_15m_atm_v1"
    cfg["strategies"]["strategies"]["ema921_15m_atm_v1"]["params"].update(start="09:30", cutoff="14:30")
    clock = FrozenClock(datetime(2026, 11, 11, 9, 20, tzinfo=IST))
    prov = Prov(clock, iv=12.0)
    prov.idx = path()
    led = Ledger(tmp_path / "l.sqlite3")
    e = TradingEngine(cfg, prov, PaperBroker(lambda s: prov.get_quote(s, "FNO"), 20000, 1), led, clock,
                      KillSwitch(tmp_path / "K"), sleep=lambda s: None)
    e.risk_cfg.update(risk_per_trade_max=100000, max_premium_outlay=100000, capital=100000, daily_loss_limit=10**6)
    e.startup()
    e.broker.funds = 100000
    while clock.now().time() < time(15, 5):
        e.step()
        clock.advance(seconds=30)
    tr = led.trades("paper")
    dirs = [json.loads(t["details"])["direction"] for t in tr]
    reasons = [t["exit_reason"] for t in tr]
    assert dirs[:2] == ["BULLISH", "BEARISH"], (dirs, reasons, led.query("SELECT payload FROM risk_decisions"))
    assert reasons[0] == "opposite_cross"
    assert reasons[-1] == "time_exit" and all(t["status"] == "CLOSED" for t in tr)
    assert tr[1]["strike"] % 50 == 0 and tr[1]["option_type"] == "PE"
