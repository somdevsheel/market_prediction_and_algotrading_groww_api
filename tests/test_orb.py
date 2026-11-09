from datetime import date, datetime, time, timedelta

import pandas as pd
import pytest

from core.clock import IST
from strategies.orb.orb_core import find_entry, passes_filters, simulate_spot

D = date(2026, 11, 11)


def minutes(path, start=time(9, 15)):
    rows, prev = [], path[0]
    for i, c in enumerate(path):
        ts = datetime.combine(D, start, IST) + timedelta(minutes=i)
        rows.append([ts, prev, max(prev, c) + 1, min(prev, c) - 1, c, 0])
        prev = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def test_breakout_first_close_and_stop():
    # 5-min OR: values 100..104 -> H=105 (high +1), L=99; breakout close 107 at minute 7
    path = [100, 102, 104, 103, 101, 103, 104, 107, 110, 98, 95]
    day = minutes(path)
    e = find_entry(day, "breakout", 5)
    assert e.direction == "BULLISH" and e.entry_idx == 7 and e.stop_px == 99
    assert e.entry_ts == day.timestamp[7] + pd.Timedelta(minutes=1)
    r = simulate_spot(day, e)
    assert r["exit_reason"] == "stop" and r["points"] == pytest.approx(99 - 107)


def test_entry_is_causal():
    path = [100, 102, 104, 103, 101, 103, 104, 107, 110, 120]
    full = find_entry(minutes(path), "breakout", 5)
    partial = find_entry(minutes(path[:8]), "breakout", 5)
    assert full.entry_idx == partial.entry_idx and full.entry_px == partial.entry_px
    assert find_entry(minutes(path[:7]), "breakout", 5) is None


def test_retest_and_reversal():
    # break up (107), back inside (103), break up again (108) -> retest long; reversal needs close < L
    path = [100, 102, 104, 103, 101, 103, 104, 107, 103, 108, 109]
    day = minutes(path)
    rt = find_entry(day, "retest", 5)
    assert rt.direction == "BULLISH" and rt.entry_idx == 9 and rt.gap_minutes == 2
    assert find_entry(day, "reversal", 5) is None
    path2 = [100, 102, 104, 103, 101, 103, 104, 107, 103, 97, 96]
    rv = find_entry(minutes(path2), "reversal", 5)
    assert rv.direction == "BEARISH" and rv.stop_px == 105  # stop at the original breakout boundary (H)
    assert find_entry(minutes(path2), "retest", 5) is None  # opposite side first -> retest void


def test_filters():
    path = [100, 102, 104, 103, 101, 103, 104, 107]
    e = find_entry(minutes(path), "breakout", 5)
    assert e.range_pct == pytest.approx((105 - 99) / 100 * 100)
    assert passes_filters(e, {"min_range_pct": 0.5, "entry_before": "12:00"})
    assert not passes_filters(e, {"max_range_pct": 0.5})
    assert not passes_filters(e, {"entry_before": "09:20"})


def test_orb_strategy_signal_only_on_entry_candle():
    from strategies.base import MarketContext
    from strategies.orb.orb_strategy import ORBStrategy

    s = ORBStrategy({"mode": "breakout", "or_minutes": 5, "orb_filters": {}})
    path = [100, 102, 104, 103, 101, 103, 104, 107, 110]
    day = minutes(path)
    ctx = lambda n: MarketContext(now=(day.timestamp[n - 1] + pd.Timedelta(minutes=1)).to_pydatetime(),  # noqa: E731
                                  index_5m=day.iloc[:n])
    sig = s.evaluate(ctx(8))
    assert sig and sig.direction == "BULLISH" and sig.invalidation_level == 99
    assert sig.diagnostics["stop_mode"] == "touch"
    assert s.evaluate(ctx(9)) is None and "entry_already_passed" in s.last_reasons
