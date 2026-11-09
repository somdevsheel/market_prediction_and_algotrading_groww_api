"""Archived tests for retired strategies A (sr_breakout) and C (breakout_retest). Not collected by default."""
from datetime import date

import pandas as pd

from tests import synthetic as syn
from tests.test_phase2 import _ctx

D = date(2026, 11, 11)


def test_sr_breakout_prior_day_high():
    from strategies.breakout.sr_breakout import SRBreakoutStrategy

    prev = syn.day_candles(date(2026, 11, 9), [25000, 25010, 25040, 25020, 25000], wiggle=1)  # pdh ~25041
    today = syn.day_candles(D, [25010, 25020, 25015, 25018, 25020, 25075], wiggle=1)
    today.loc[5, ["open", "low", "high"]] = [25020, 25019, 25076]
    idx = pd.concat([prev, today], ignore_index=True)
    s = SRBreakoutStrategy({"breakout_buffer_points": 5, "min_body_ratio": 0.5, "max_extension_points": 60,
                            "opening_range_minutes": 15, "use_swings": False})
    sig = s.evaluate(_ctx(idx, pd.DataFrame(columns=["timestamp"]), len(idx) - 1))
    assert sig and sig.direction == "BULLISH" and any(r.startswith("break_") for r in sig.reason_codes), s.last_reasons


def test_breakout_retest_sequence():
    from strategies.breakout.breakout_retest import BreakoutRetestStrategy

    # OR high ~25032; breakout close 25060; retest low touches 25035 closing 25045; confirm close > retest high
    closes = [25010, 25030, 25020, 25060, 25045, 25070]
    idx = syn.day_candles(D, closes, wiggle=1)
    idx.loc[4, ["low", "high"]] = [25030, 25062]
    idx.loc[5, ["open", "high", "low"]] = [25045, 25071, 25044]
    s = BreakoutRetestStrategy({"opening_range_minutes": 15, "breakout_buffer_points": 5,
                                "retest_tolerance_points": 8, "max_bars_after_breakout": 12})
    assert s.evaluate(_ctx(idx, pd.DataFrame(columns=["timestamp"]), 4)) is None  # retest bar, no confirmation yet
    sig = s.evaluate(_ctx(idx, pd.DataFrame(columns=["timestamp"]), 5))
    assert sig and sig.direction == "BULLISH" and "retest_held" in sig.reason_codes, s.last_reasons
    assert sig.invalidation_level == 25030


