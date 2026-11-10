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


def test_stock_order_cost_is_realistic_for_small_orders():
    from ml_lab.stocks_ml import order_cost
    buy, sell = order_cost(10_000, "buy"), order_cost(10_000, "sell")
    # Rs 10k delivery: brokerage 10 + STT 10 + stamp 1.5 + GST + slippage 10; sell adds DP ~16
    assert 30 < buy < 35 and 45 < sell < 52


def test_stock_seq_net_shapes():
    import torch
    from ml_lab.stocks_ml import SEQ_L, SeqNet
    for kind in ("lstm", "gru", "cnn", "transformer"):
        out = SeqNet(7, kind)(torch.zeros(4, 7), torch.zeros(4, SEQ_L, 3))
        assert out.shape == (4,)
