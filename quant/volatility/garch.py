"""GARCH(1,1), GJR-GARCH(1,1) and EGARCH(1,1) by Gaussian MLE (SciPy).

Implemented directly (no ``arch`` dependency) so the maths is auditable.
Returns are scaled by 100 internally for numerical stability.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class GarchFit:
    model: str
    params: dict
    loglik: float
    aic: float
    converged: bool
    last_variance: float      # in (pct return)^2
    last_return: float
    n: int

    def persistence(self) -> float:
        p = self.params
        if self.model == "garch":
            return p["alpha"] + p["beta"]
        if self.model == "gjr":
            return p["alpha"] + p["beta"] + 0.5 * p["gamma"]
        return p["beta"]


def _filter(model: str, theta: np.ndarray, r: np.ndarray) -> np.ndarray:
    n = len(r)
    s2 = np.empty(n)
    s2[0] = np.var(r)
    if model == "garch":
        w, a, b = theta
        for t in range(1, n):
            s2[t] = w + a * r[t - 1] ** 2 + b * s2[t - 1]
    elif model == "gjr":
        w, a, g, b = theta
        for t in range(1, n):
            s2[t] = w + (a + g * (r[t - 1] < 0)) * r[t - 1] ** 2 + b * s2[t - 1]
    else:  # egarch
        w, a, g, b = theta
        ez = np.sqrt(2 / np.pi)
        ls = np.log(s2[0])
        for t in range(1, n):
            z = r[t - 1] / np.sqrt(np.exp(ls))
            ls = float(np.clip(w + a * (abs(z) - ez) + g * z + b * ls, -30, 30))
            s2[t] = np.exp(ls)
    return s2


def _nll(theta, model, r):
    s2 = _filter(model, theta, r)
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        return 1e10
    return 0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + r ** 2 / s2)


def fit(returns: pd.Series, model: str = "garch") -> GarchFit:
    r = (returns.dropna().to_numpy() * 100.0)
    r = r - r.mean()
    v = np.var(r)
    if model == "garch":
        x0, bounds = [v * 0.05, 0.08, 0.9], [(1e-8, None), (0, 1), (0, 1)]
        cons = [{"type": "ineq", "fun": lambda t: 0.9999 - t[1] - t[2]}]
        names = ["omega", "alpha", "beta"]
    elif model == "gjr":
        x0, bounds = [v * 0.05, 0.03, 0.08, 0.88], [(1e-8, None), (0, 1), (0, 1), (0, 1)]
        cons = [{"type": "ineq", "fun": lambda t: 0.9999 - t[1] - t[3] - 0.5 * t[2]}]
        names = ["omega", "alpha", "gamma", "beta"]
    elif model == "egarch":
        x0, bounds = [np.log(v) * 0.05, 0.1, -0.05, 0.95], [(None, None), (-2, 2), (-2, 2), (-0.9999, 0.9999)]
        cons = []
        names = ["omega", "alpha", "gamma", "beta"]
    else:
        raise ValueError(model)
    res = minimize(_nll, x0, args=(model, r), method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500})
    s2 = _filter(model, res.x, r)
    ll = -float(res.fun)
    return GarchFit(model=model, params=dict(zip(names, map(float, res.x))), loglik=ll,
                    aic=2 * len(names) - 2 * ll, converged=bool(res.success),
                    last_variance=float(s2[-1]), last_return=float(r[-1]), n=len(r))


def forecast_variance(f: GarchFit, horizon: int = 1) -> np.ndarray:
    """h-step-ahead variance forecasts in (pct return)^2."""
    p = f.params
    out = np.empty(horizon)
    if f.model in ("garch", "gjr"):
        a_eff = p["alpha"] + (0.5 * p["gamma"] if f.model == "gjr" else 0)
        lev = p.get("gamma", 0) * (f.last_return < 0) if f.model == "gjr" else 0
        nxt = p["omega"] + (p["alpha"] + lev) * f.last_return ** 2 + p["beta"] * f.last_variance
        out[0] = nxt
        for h in range(1, horizon):
            out[h] = p["omega"] + (a_eff + p["beta"]) * out[h - 1]
    else:
        ez = np.sqrt(2 / np.pi)
        z = f.last_return / np.sqrt(f.last_variance)
        ls = p["omega"] + p["alpha"] * (abs(z) - ez) + p["gamma"] * z + p["beta"] * np.log(f.last_variance)
        out[0] = np.exp(ls)
        for h in range(1, horizon):  # E[|z|-ez]=0, E[z]=0 -> log-variance recursion (approximation)
            ls = p["omega"] + p["beta"] * ls
            out[h] = np.exp(ls)
    return out


def forecast_annual_vol(f: GarchFit, horizon: int = 1, ppy: float = 252) -> float:
    """Average annualized vol (%) over the next ``horizon`` periods."""
    return float(np.sqrt(forecast_variance(f, horizon).mean() * ppy))


def rolling_oos_evaluation(returns: pd.Series, models=("garch", "gjr", "egarch"), train: int = 500,
                           step: int = 20) -> pd.DataFrame:
    """Refit every ``step`` bars on a trailing window; score 1-step forecasts (QLIKE) out of sample."""
    r = returns.dropna()
    rows = []
    for m in models:
        errs = []
        for end in range(train, len(r) - 1, step):
            f = fit(r.iloc[end - train:end], m)
            fc = forecast_variance(f, step)
            real = (r.iloc[end:end + step].to_numpy() * 100) ** 2
            k = min(len(fc), len(real))
            errs.extend(np.log(fc[:k]) + real[:k] / fc[:k])
        rows.append({"model": m, "oos_qlike": float(np.mean(errs)) if errs else float("nan"), "n": len(errs)})
    return pd.DataFrame(rows).sort_values("oos_qlike").reset_index(drop=True)
