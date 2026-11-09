from datetime import date, datetime

import pandas as pd
import pytest

from core.clock import IST
from core.config import load_config
from execution.contract_selector import choose_expiry, itm_strike, select_contract, validate_quote
from tests import synthetic as syn

CFG = load_config()


def test_itm_strikes():
    assert itm_strike(25044.65, "CE", 50) == 25000
    assert itm_strike(25044.65, "PE", 50) == 25050
    assert itm_strike(25000.0, "CE", 50) == 24950
    assert itm_strike(25000.0, "PE", 50) == 25050
    assert itm_strike(25044.65, "CE", 50, 2) == 24950


def test_expiry_choice_avoids_expiry_day():
    ex = [date(2026, 11, 9), date(2026, 11, 17), date(2026, 11, 23)]
    assert choose_expiry(ex, date(2026, 11, 17), True) == date(2026, 11, 23)
    assert choose_expiry(ex, date(2026, 11, 17), False) == date(2026, 11, 17)
    assert choose_expiry(ex, date(2026, 11, 11), True) == date(2026, 11, 17)
    assert choose_expiry(ex, date(2026, 12, 1), True) is None


def _select(q_bid=250.0, q_ask=251.0, now=None, **qkw):
    now = now or datetime(2026, 11, 11, 9, 50, tzinfo=IST)
    exp = date(2026, 11, 17)
    ch = syn.chain(exp, 25044.65, now)
    inst = syn.instruments(exp, 25044.65)
    look = lambda s: (inst[inst.trading_symbol == s].iloc[0].to_dict() if (inst.trading_symbol == s).any() else None)
    return select_contract("BULLISH", 25044.65, now, [date(2026, 11, 9), exp], ch, look,
                           lambda s: syn.quote(s, q_bid, q_ask, now, **qkw), CFG["instruments"], 20)


def test_contract_selection_ok():
    sel = _select()
    assert sel.ok, sel.reason
    assert sel.contract.strike == 25000 and sel.contract.option_type == "CE" and sel.contract.lot_size == 65


@pytest.mark.parametrize("kw,reason", [
    ({"q_bid": 240.0, "q_ask": 251.0}, "spread_too_wide"),
    ({"volume": 10}, "low_volume"),
    ({"oi": 1}, "low_open_interest"),
    ({"q_bid": 0.0}, "no_two_sided_quote"),
])
def test_contract_selection_rejections(kw, reason):
    sel = _select(**kw)
    assert not sel.ok and sel.reason.startswith(reason)


def test_stale_quote_rejected():
    now = datetime(2026, 11, 11, 9, 50, tzinfo=IST)
    q = syn.quote("X", 100, 100.5, now - pd.Timedelta(minutes=5))
    ok, why, _ = validate_quote(q, now, CFG["instruments"]["contract_selection"], 20)
    assert not ok and why.startswith("stale_quote")


def test_atm_when_itm_steps_zero():
    assert itm_strike(25044.65, "CE", 50, 0) == 25050 and itm_strike(25020.0, "PE", 50, 0) == 25000
