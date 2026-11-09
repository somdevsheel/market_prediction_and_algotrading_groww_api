from datetime import date, datetime, time

import pandas as pd
import pytest

from core.clock import IST
from core.config import load_config
from execution.contract_selector import choose_expiry, itm_strike, select_contract, validate_quote
from strategies.base import MarketContext
from strategies.registry import build_strategy
from tests import synthetic as syn

CFG = load_config()
D = date(2026, 11, 11)


def strat(**ov):
    return build_strategy("breakout_vwap_v1", CFG["strategies"]["strategies"]["breakout_vwap_v1"], ov)


def ctx_for(idx, fut, upto: int):
    vis = idx.iloc[: upto + 1]
    now = vis["timestamp"].iloc[-1] + pd.Timedelta(minutes=5)
    fv = fut[fut.timestamp <= vis.timestamp.iloc[-1]]
    return MarketContext(now=now.to_pydatetime(), index_5m=vis, futures_5m=fv, regime={"regime": "trending_bullish"})


def test_breakout_signal_generated_and_reproducible():
    idx, fut = syn.breakout_day(D)
    s = strat(use_context_filter=False)
    sig = s.evaluate(ctx_for(idx, fut, 6))
    assert sig is not None, s.last_reasons
    assert sig.direction == "BULLISH"
    assert sig.invalidation_level < sig.reference_price < sig.target_level
    assert "futures_vwap_confirm" in sig.reason_codes and "futures_volume_expansion" in sig.reason_codes
    sig2 = strat(use_context_filter=False).evaluate(ctx_for(idx, fut, 6))
    assert sig.signal_id == sig2.signal_id  # deterministic


def test_no_signal_inside_range_or_before_or_complete():
    idx, fut = syn.breakout_day(D)
    s = strat(use_context_filter=False)
    assert s.evaluate(ctx_for(idx, fut, 1)) is None and "opening_range_incomplete" in s.last_reasons
    assert s.evaluate(ctx_for(idx, fut, 5)) is None


def test_volume_filter_blocks():
    idx, fut = syn.breakout_day(D)
    fut.loc[6, "volume"] = 100
    s = strat(use_context_filter=False)
    assert s.evaluate(ctx_for(idx, fut, 6)) is None
    assert any(r.startswith("volume_filter_failed") for r in s.last_reasons)


def test_signal_does_not_use_future_candles():
    idx, fut = syn.breakout_day(D)
    s = strat(use_context_filter=False)
    a = s.evaluate(ctx_for(idx, fut, 6))
    more = pd.concat([idx, syn.day_candles(D, [24000] * 3).assign(
        timestamp=lambda x: x.timestamp + pd.Timedelta(minutes=35))], ignore_index=True)
    b = s.evaluate(ctx_for(more, fut, 6))
    assert a.signal_id == b.signal_id


def test_context_filter_needs_history():
    idx, fut = syn.breakout_day(D)
    s = strat(use_context_filter=True)
    assert s.evaluate(ctx_for(idx, fut, 6)) is None
    assert "context_insufficient_history" in s.last_reasons


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
