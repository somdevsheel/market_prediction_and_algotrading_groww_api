"""Leakage tests for the ML lab (synthetic data; no network)."""
import numpy as np
import pandas as pd

from ml_lab.models import deep


def test_external_merge_uses_strictly_earlier_dates():
    d = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"])})
    x = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "v": [1.0, 2.0]})
    m = pd.merge_asof(d, x, on="date", allow_exact_matches=False)
    assert np.isnan(m.v.iloc[0]) and m.v.iloc[1] == 1.0  # same-day value never used


def test_sequences_end_at_decision_row():
    X = np.arange(30, dtype=float).reshape(30, 1)
    s = deep.sequences(X, np.array([25]), 5)
    assert s[0, -1, 0] == 25 and s[0, 0, 0] == 21  # rows 21..25, nothing after 25


def test_deep_model_ignores_test_targets_and_future_features():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = (X[:, 0] > 0).astype(float)
    tr, te = np.arange(0, 300), np.arange(300, 350)
    a = deep.fit_predict("mlp", X, y, tr, te, "clf", L=5, seed=0, epochs=5)
    X2, y2 = X.copy(), y.copy()
    X2[350:] = 99.0      # change rows after the test window
    y2[300:] = 1 - y2[300:]  # flip test targets
    b = deep.fit_predict("mlp", X2, y2, tr, te, "clf", L=5, seed=0, epochs=5)
    assert np.allclose(a, b, atol=1e-5)


def test_options_external_join_uses_only_earlier_sessions():
    import pandas as pd
    from ml_lab.options_ml import ext_prev
    us = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]))
    india = pd.DatetimeIndex(pd.to_datetime(["2025-01-02", "2025-01-03"]))
    # the US session of the same date closes after India's close, so it must not be visible
    assert ext_prev(us, india).tolist() == [1.0, 2.0]


def test_bs_vec_put_call_parity():
    import numpy as np
    from ml_lab.options_ml import R, bs_vec
    S, K, T, s = np.array([25000.0]), np.array([25100.0]), np.array([0.02]), np.array([0.15])
    c, p = bs_vec(S, K, T, s, np.array([True])), bs_vec(S, K, T, s, np.array([False]))
    assert abs((c - p)[0] - (S - K * np.exp(-R * T))[0]) < 1e-6
