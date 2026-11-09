import math

import numpy as np
import pandas as pd
import pytest

from quant.features import indicators as ind
from quant.options import bsm
from quant.volatility import ewma, garch, hv


def test_hv_matches_manual():
    close = pd.Series(100 * np.exp(np.cumsum(np.r_[0, np.random.default_rng(0).normal(0, 0.01, 99)])))
    out = hv.historical_volatility(close, 20)
    r = np.log(close / close.shift(1)).iloc[-20:]
    assert out.iloc[-1] == pytest.approx(r.std(ddof=1) * math.sqrt(252) * 100)
    assert set(hv.hv_table(close).columns) == {"HV5", "HV10", "HV20", "HV30", "HV60"}


def test_intraday_annualization_factor():
    assert hv.periods_per_year(5) == 252 * 75
    assert hv.periods_per_year(None) == 252


def test_ewma_recursion_no_lookahead():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 60))
    v = ewma.ewma_variance(r, 0.94, seed_window=20)
    seed = np.var(r.iloc[:20], ddof=1)
    assert v.iloc[20] == pytest.approx(seed)
    assert v.iloc[21] == pytest.approx(0.94 * seed + 0.06 * r.iloc[20] ** 2)
    r2 = r.copy(); r2.iloc[40] = 1.0  # shock at 40 must not affect forecast at 40
    assert ewma.ewma_variance(r2).iloc[40] == pytest.approx(v.iloc[40])


def _simulate_garch(n=3000, w=0.02, a=0.08, b=0.9, seed=3):
    rng = np.random.default_rng(seed)
    r, s2 = np.zeros(n), w / (1 - a - b)
    for t in range(n):
        r[t] = rng.normal() * math.sqrt(s2)
        s2 = w + a * r[t] ** 2 + b * s2
    return pd.Series(r / 100)


def test_garch_recovers_parameters_roughly():
    f = garch.fit(_simulate_garch(), "garch")
    assert f.converged
    assert 0.03 < f.params["alpha"] < 0.15
    assert 0.8 < f.params["beta"] < 0.97
    fc = garch.forecast_variance(f, 50)
    assert fc[-1] == pytest.approx(f.params["omega"] / (1 - f.persistence()), rel=0.3)


@pytest.mark.parametrize("m", ["gjr", "egarch"])
def test_other_garch_models_fit(m):
    f = garch.fit(_simulate_garch(1500), m)
    assert np.isfinite(f.loglik)
    assert garch.forecast_annual_vol(f, 5) > 0


def test_bsm_put_call_parity_and_known_value():
    s, k, t, r, v = 100, 100, 1.0, 0.05, 0.2
    c, p = bsm.price(s, k, t, r, v, "CE"), bsm.price(s, k, t, r, v, "PE")
    assert c == pytest.approx(10.4506, abs=1e-3)  # Hull textbook value
    assert c - p == pytest.approx(s - k * math.exp(-r * t), abs=1e-9)


def test_iv_roundtrip_and_invalid_prices():
    s, k, t, r = 25000, 25100, 7 / 365, 0.065
    for v in (0.08, 0.15, 0.4):
        for o in ("CE", "PE"):
            px = bsm.price(s, k, t, r, v, o)
            assert bsm.implied_vol(px, s, k, t, r, o) == pytest.approx(v, abs=1e-5)
    assert bsm.implied_vol(0, s, k, t, r, "CE") is None
    assert bsm.implied_vol(1.0, s, 24000, t, r, "CE") is None  # below intrinsic
    assert bsm.implied_vol(100, s, k, 0, r, "CE") is None


def test_greeks_vs_finite_difference():
    s, k, t, r, v = 25000, 25000, 10 / 365, 0.065, 0.15
    for o in ("CE", "PE"):
        g = bsm.greeks(s, k, t, r, v, o)
        h = 1.0
        d_fd = (bsm.price(s + h, k, t, r, v, o) - bsm.price(s - h, k, t, r, v, o)) / (2 * h)
        g_fd = (bsm.price(s + h, k, t, r, v, o) - 2 * bsm.price(s, k, t, r, v, o) + bsm.price(s - h, k, t, r, v, o)) / h ** 2
        vega_fd = (bsm.price(s, k, t, r, v + 0.005, o) - bsm.price(s, k, t, r, v - 0.005, o)) / 1.0
        theta_fd = (bsm.price(s, k, t - 1 / 365, r, v, o) - bsm.price(s, k, t, r, v, o))
        assert g.delta == pytest.approx(d_fd, abs=1e-4)
        assert g.gamma == pytest.approx(g_fd, rel=1e-3)
        assert g.vega == pytest.approx(vega_fd, rel=1e-3)
        assert g.theta == pytest.approx(theta_fd, rel=0.05)
    assert bsm.greeks(s, k, t, r, v, "CE").delta - bsm.greeks(s, k, t, r, v, "PE").delta == pytest.approx(1, abs=1e-9)


def test_indicators_basic():
    ts = pd.date_range("2026-11-09 09:15", periods=30, freq="5min", tz="Asia/Kolkata")
    df = pd.DataFrame({"timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10.0})
    assert ind.session_vwap(df).iloc[-1] == pytest.approx(100.0)
    assert ind.atr(df, 14).iloc[-1] == pytest.approx(2.0)
    up = pd.Series(np.arange(1, 40, dtype=float))
    assert ind.rsi(up).iloc[-1] == pytest.approx(100.0)
    assert ind.ema(pd.Series([1.0] * 30), 9).iloc[-1] == pytest.approx(1.0)
    assert ind.opening_range(df, 15) == (101.0, 99.0)


def test_vwap_resets_daily():
    a = pd.date_range("2026-11-09 09:15", periods=3, freq="5min", tz="Asia/Kolkata")
    b = pd.date_range("2026-11-11 09:15", periods=3, freq="5min", tz="Asia/Kolkata")
    df = pd.DataFrame({"timestamp": a.append(b), "high": [10, 10, 10, 20, 20, 20], "low": [10, 10, 10, 20, 20, 20],
                       "close": [10, 10, 10, 20, 20, 20], "volume": [1] * 6})
    assert ind.session_vwap(df).iloc[3] == pytest.approx(20)


def test_volume_expansion_excludes_current_bar():
    v = pd.Series([10.0] * 6 + [30.0])
    assert ind.volume_expansion(v, 6).iloc[-1] == pytest.approx(3.0)
