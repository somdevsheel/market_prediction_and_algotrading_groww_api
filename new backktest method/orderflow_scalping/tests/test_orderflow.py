"""No-look-ahead, formula and execution tests (synthetic data only)."""
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import yaml

from orderflow_scalping._paths import EXP_ROOT
from orderflow_scalping.backtesting import engine
from orderflow_scalping.backtesting.costs import futures_charges
from orderflow_scalping.indicators import ofi, regime, volume_delta, vwap
from orderflow_scalping.indicators.microprice import deviation, microprice
from orderflow_scalping.strategies import (hybrid_strategy, ofi_strategy, volume_delta_strategy, vwap_mean_reversion,
                                           vwap_momentum)

S = yaml.safe_load((EXP_ROOT / "config" / "settings.yaml").read_text())
P = yaml.safe_load((EXP_ROOT / "config" / "strategy_params.yaml").read_text())


def synth(days=6, seed=0):
    rng = np.random.default_rng(seed)
    rows, px = [], 20000.0
    for k in range(days):
        d = date(2026, 1, 5) + timedelta(days=k)
        for i in range(375):
            ts = pd.Timestamp(datetime(d.year, d.month, d.day, 9, 15)).tz_localize("Asia/Kolkata") + pd.Timedelta(minutes=i)
            o = px
            px = px + rng.normal(0, 4)
            h, l = max(o, px) + abs(rng.normal(0, 2)), min(o, px) - abs(rng.normal(0, 2))
            rows.append([ts, o, h, l, px, float(rng.integers(5000, 50000))])
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = df.timestamp.dt.date
    df["minute"] = df.timestamp.dt.hour * 60 + df.timestamp.dt.minute
    df["atr"] = regime.atr(df)
    df["vwap"] = vwap.session_vwap(df)
    df["vwap_z"] = vwap.deviation_z(df, df["vwap"])
    df["vwap_slope"] = vwap.slope(df["vwap"], 10)
    df["ofi_proxy"] = ofi.ofi_proxy(df)
    df["delta_bvc"] = volume_delta.delta_bvc(df)
    df["vol_ma"] = df["volume"].rolling(60, min_periods=20).mean().shift(1)
    df["rel_volume"] = df["volume"] / df["vol_ma"]
    return regime.add_regime(df, df["vwap_slope"])


def test_cks_ofi_known_case():
    # bid up with qty 10 (prev 5): +10 ; ask unchanged qty 8 -> 6: -(6) + 8
    e = ofi.ofi_cks([100, 100.05], [100.1, 100.1], [5, 10], [8, 6])
    assert e[1] == pytest.approx(10 - 6 + 8)


def test_microprice_formula():
    assert microprice(100, 101, 300, 100) == pytest.approx((101 * 300 + 100 * 100) / 400)
    assert deviation(100, 101, 300, 100) > 0  # more bid size -> microprice above mid


def test_bvc_sign_and_proxy_clv():
    df = pd.DataFrame({"close": [100, 101, 102, 101, 100] * 20, "volume": [1000] * 100})
    d = volume_delta.delta_bvc(df).iloc[25:]  # needs 20 bars of past volatility first (0 before that)
    up, dn = df.close.diff().iloc[25:] > 0, df.close.diff().iloc[25:] < 0
    assert (d[up] > 0).all() and (d[dn] < 0).all()
    bar = pd.DataFrame({"high": [10.0], "low": [0.0], "close": [10.0], "volume": [100.0]})
    assert ofi.ofi_proxy(bar).iloc[0] == 100.0


def test_futures_costs():
    c = futures_charges(20000 * 65, 20010 * 65, S["costs"])
    assert c["brokerage"] == 40 and c["stt"] == pytest.approx(20010 * 65 * 0.0002)
    assert c["total"] == pytest.approx(420.18, abs=0.5)  # ~Rs 420 per round trip at NIFTY 20,000 x 65


@pytest.mark.parametrize("mod,params", [(ofi_strategy, P["ofi"]["default"]), (volume_delta_strategy, P["volume_delta"]["default"]),
                                        (vwap_momentum, P["vwap_momentum"]["default"]),
                                        (vwap_mean_reversion, P["vwap_mean_reversion"]["default"])])
def test_no_lookahead_in_signals(mod, params):
    df = synth()
    a = mod.signals(df, params)[0]
    df2 = df.copy()
    cut = len(df) - 200
    for c in ("open", "high", "low", "close"):
        df2.loc[cut:, c] *= 1.05  # shock the future
    df2.loc[cut:, "volume"] *= 7
    df2 = synth.__wrapped__(df2) if hasattr(synth, "__wrapped__") else df2
    # recompute features on the shocked frame exactly as the pipeline does
    for k, f in (("atr", lambda d: regime.atr(d)), ("vwap", lambda d: vwap.session_vwap(d))):
        df2[k] = f(df2)
    df2["vwap_z"] = vwap.deviation_z(df2, df2["vwap"])
    df2["vwap_slope"] = vwap.slope(df2["vwap"], 10)
    df2["ofi_proxy"] = ofi.ofi_proxy(df2)
    df2["delta_bvc"] = volume_delta.delta_bvc(df2)
    df2["vol_ma"] = df2["volume"].rolling(60, min_periods=20).mean().shift(1)
    df2["rel_volume"] = df2["volume"] / df2["vol_ma"]
    df2 = regime.add_regime(df2.drop(columns=["adx", "rv", "rv_z", "regime"]), df2["vwap_slope"])
    b = mod.signals(df2, params)[0]
    assert (a[:cut] == b[:cut]).all()


def test_engine_fills_next_open_and_squares_off():
    df = synth(days=2)
    sig = np.zeros(len(df), dtype=np.int8)
    i = int(np.where(df.minute == 10 * 60)[0][0])
    sig[i] = 1
    s = yaml.safe_load(yaml.safe_dump(S))
    s["risk"].update(stop_atr_mult=1000, target_atr_mult=1000, max_holding_bars=10_000)
    t = engine.run(df, sig, s, "base")
    assert len(t) == 1
    assert t.iloc[0]["entry"] == pytest.approx(df.open.iloc[i + 1] + 1.0)  # next open + half-spread + slippage
    assert t.iloc[0]["exit_reason"] == "square_off"
    assert pd.Timestamp(t.iloc[0]["exit_time"]).strftime("%H:%M") == "15:20"


def test_limit_nonfill_counted():
    df = synth(days=1)
    df.loc[:, ["open", "high", "low", "close"]] = 20000.0  # flat: price never trades through the limit
    df["atr"] = 5.0
    sig = np.zeros(len(df), dtype=np.int8)
    sig[100] = 1
    s = yaml.safe_load(yaml.safe_dump(S))
    s["execution"]["order_type"] = "limit"
    t = engine.run(df, sig, s, "base")
    assert t.empty or len(t) == 0


def test_daily_loss_limit_halts():
    df = synth(days=1)
    sig = np.zeros(len(df), dtype=np.int8)
    sig[30:300:3] = 1
    s = yaml.safe_load(yaml.safe_dump(S))
    s["risk"].update(daily_loss_limit=500, stop_atr_mult=0.01, max_consecutive_losses=99)
    t = engine.run(df, sig, s, "conservative")
    assert len(t) <= 3 and t.net_pnl.cumsum().iloc[-1] <= -500


def test_hybrid_fit_uses_train_only():
    df = synth(days=6)
    mask = (df.date < df.date.unique()[3]).to_numpy()
    m1 = hybrid_strategy.fit(df, mask, P["hybrid"]["default"])
    df2 = df.copy()
    df2.loc[~mask, ["open", "high", "low", "close"]] *= 1.2  # change only the test period
    m2 = hybrid_strategy.fit(df2, mask, P["hybrid"]["default"])
    assert m1["weights"] == pytest.approx(m2["weights"], rel=1e-6)
