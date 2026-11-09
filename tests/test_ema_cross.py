from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

from backtesting.engine.ema_options import EMAOptionSim, contract_for, metrics
from core.clock import IST
from strategies.ema_cross.core import EMAConfig, day_trades, resample, spot_backtest, with_emas


def minute_days(paths: dict) -> pd.DataFrame:
    rows = []
    for d, path in paths.items():
        prev = path[0]
        for i, c in enumerate(path):
            ts = datetime.combine(d, time(9, 15), IST) + timedelta(minutes=i)
            rows.append([ts, prev, max(prev, c) + 0.5, min(prev, c) - 0.5, c])
            prev = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])


def test_resample_anchored_and_complete_only():
    d = date(2026, 11, 11)
    df = minute_days({d: list(np.linspace(100, 110, 375))})
    b = resample(df, 15)
    assert b.timestamp.iloc[0].time() == time(9, 15) and b.timestamp.iloc[1].time() == time(9, 30)
    assert len(b) == 25
    b5 = resample(df.drop(index=3), 5)  # missing minute -> first 5m bin incomplete -> dropped
    assert b5.timestamp.iloc[0].time() == time(9, 20)


def test_cross_detection_and_tie():
    b = pd.DataFrame({"timestamp": pd.date_range("2026-11-11 09:15", periods=60, freq="5min", tz=IST),
                      "close": [100.0] * 60})
    b = with_emas(b)
    assert (b.cross == 0).all()  # flat: all ties, never a cross
    up = np.r_[np.linspace(110, 100, 40), np.linspace(100, 120, 40)]
    b2 = with_emas(pd.DataFrame({"timestamp": pd.date_range("2026-11-10 09:15", periods=80, freq="5min", tz=IST),
                                 "close": up}))
    assert (b2.cross == 1).sum() == 1 and (b2.cross == -1).sum() == 0


def test_rules_entry_window_reversal_and_squareoff():
    d0, d = date(2026, 11, 10), date(2026, 11, 11)
    warm = list(np.linspace(100, 90, 375))
    # today: fall, then rise (bullish cross ~late morning), then fall again (bearish cross -> reversal)
    today = list(np.linspace(90, 85, 60)) + list(np.linspace(85, 110, 120)) + list(np.linspace(110, 80, 195))
    df = minute_days({d0: warm, d: today})
    trades, why = spot_backtest(df, EMAConfig(5, "09:30", "14:30"))
    t = trades[trades.date == str(d)]
    assert list(t.direction) == ["BULLISH", "BEARISH"]
    assert t.iloc[0].exit_reason == "opposite_cross" and bool(t.iloc[1].reversal)
    assert t.iloc[0].exit_ts == t.iloc[1].entry_ts  # reversal at the same fill bar
    assert t.iloc[1].exit_reason == "square_off_15:00" and t.iloc[1].exit_ts.time() == time(15, 0)
    # fill bar = 1-minute bar starting at the decision time (bar close)
    assert (t.entry_ts.dt.minute % 5 == 0).all()
    e = trades[trades.date == str(d)].iloc[0]
    assert e.spot_points == pytest.approx(e.exit_spot - e.entry_spot, abs=0.01)


def test_cutoff_blocks_new_entries_but_allows_exit():
    d0, d = date(2026, 11, 10), date(2026, 11, 11)
    df = minute_days({d0: list(np.linspace(100, 90, 375)),
                      d: list(np.linspace(90, 85, 60)) + list(np.linspace(85, 110, 120)) + list(np.linspace(110, 80, 195))})
    t, _ = spot_backtest(df, EMAConfig(5, "09:30", "11:30"))
    t = t[t.date == str(d)]
    assert len(t) == 1 and t.iloc[0].exit_reason == "opposite_cross"  # bearish cross after cutoff: exit, no reverse


def test_contract_modes():
    assert contract_for("ITM5_BUY", 1, 25012) == (24750, "CE", "BUY")
    assert contract_for("ITM5_BUY", -1, 25012) == (25250, "PE", "BUY")
    assert contract_for("ATM_BUY", 1, 25030) == (25050, "CE", "BUY")
    assert contract_for("ATM_SELL", 1, 25030) == (25050, "PE", "SELL")
    assert contract_for("ATM_SELL", -1, 25030) == (25050, "CE", "SELL")


def test_friction_and_sell_floor():
    sim = EMAOptionSim(None, None, None, {"brokerage_per_order": 20, "stt_sell_pct": 0.1, "exchange_txn_pct": 0.03503,
                                          "sebi_pct": 0.0001, "stamp_buy_pct": 0.003, "ipft_pct": 0.0005, "gst_pct": 18})
    assert sim.buy_fill(100.0) == pytest.approx(101.0)       # 100*1.005 + 0.5
    assert sim.sell_fill(100.0) == pytest.approx(99.0)
    assert sim.sell_fill(0.3) == 0.05                         # floored at one tick
    m = metrics(pd.DataFrame({"net_pnl": [100.0, -50.0, 200.0, -50.0], "gross_pnl": [150.0, 0, 250, 0],
                              "friction": [50.0] * 4}))
    assert m["profit_factor"] == pytest.approx(3.0) and m["payoff_ratio"] == pytest.approx(3.0)
    assert m["win_rate"] == 50.0 and m["expectancy"] == 50.0
