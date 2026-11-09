"""EWMA (RiskMetrics) variance: sigma2(t) = lam*sigma2(t-1) + (1-lam)*r(t-1)^2."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_variance(returns: pd.Series, lam: float = 0.94, seed_window: int = 20) -> pd.Series:
    """One-step-ahead variance forecast: value at t uses information up to t-1 only."""
    r = returns.dropna().to_numpy()
    n = len(r)
    out = np.full(n, np.nan)
    if n <= seed_window:
        return pd.Series(out, index=returns.dropna().index)
    var = np.var(r[:seed_window], ddof=1)
    for t in range(seed_window, n):
        out[t] = var
        var = lam * var + (1 - lam) * r[t] ** 2
    return pd.Series(out, index=returns.dropna().index)


def ewma_vol(returns: pd.Series, lam: float = 0.94, ppy: float = 252) -> pd.Series:
    return np.sqrt(ewma_variance(returns, lam) * ppy) * 100


def compare_lambdas(returns: pd.Series, lambdas=(0.90, 0.94, 0.97), oos_fraction: float = 0.3) -> pd.DataFrame:
    """Out-of-sample forecast error (MSE / QLIKE vs realized r^2) for each decay factor."""
    r = returns.dropna()
    split = int(len(r) * (1 - oos_fraction))
    rows = []
    for lam in lambdas:
        f = ewma_variance(r, lam).iloc[split:]
        realized = (r ** 2).iloc[split:]
        m = f.notna() & (f > 0)
        mse = float(((f[m] - realized[m]) ** 2).mean())
        qlike = float((np.log(f[m]) + realized[m] / f[m]).mean())
        rows.append({"lambda": lam, "oos_mse": mse, "oos_qlike": qlike, "n": int(m.sum())})
    return pd.DataFrame(rows).sort_values("oos_qlike").reset_index(drop=True)
