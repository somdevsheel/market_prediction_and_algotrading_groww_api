"""Robustness tools: chronological IS/OOS split, Monte Carlo trade-sequence and bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.reports.metrics import compute_metrics


def split_is_oos(trades: pd.DataFrame, days: list, oos_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    days = sorted(days)
    if not days:
        return trades, trades.iloc[0:0], ""
    cut = days[int(len(days) * (1 - oos_fraction))] if len(days) > 1 else days[-1]
    cut_s = str(cut)
    return trades[trades.trading_day < cut_s], trades[trades.trading_day >= cut_s], cut_s


def monte_carlo_drawdown(pnls: np.ndarray, capital: float, n: int = 2000, seed: int = 7) -> dict:
    """Shuffle trade order to estimate the distribution of max drawdown / final P&L."""
    if len(pnls) < 2:
        return {}
    rng = np.random.default_rng(seed)
    dds, finals = [], []
    for _ in range(n):
        s = rng.permutation(pnls)
        eq = capital + np.cumsum(s)
        dds.append(float((eq - np.maximum.accumulate(np.r_[capital, eq])[1:]).min()))
        finals.append(float(s.sum()))
    return {"max_dd_p50": round(float(np.percentile(dds, 50)), 2), "max_dd_p95": round(float(np.percentile(dds, 5)), 2),
            "worst_dd": round(float(min(dds)), 2), "runs": n}


def bootstrap_expectancy(pnls: np.ndarray, n: int = 2000, seed: int = 11) -> dict:
    """Resample trades with replacement; CI for mean P&L per trade."""
    if len(pnls) < 2:
        return {}
    rng = np.random.default_rng(seed)
    means = [rng.choice(pnls, len(pnls), replace=True).mean() for _ in range(n)]
    return {"mean": round(float(np.mean(pnls)), 2), "ci05": round(float(np.percentile(means, 5)), 2),
            "ci95": round(float(np.percentile(means, 95)), 2),
            "prob_mean_positive": round(float((np.array(means) > 0).mean()), 3)}


def robustness_report(trades: pd.DataFrame, days: list, capital: float, oos_fraction: float) -> dict:
    ins, oos, cut = split_is_oos(trades, days, oos_fraction)
    n_is = len([d for d in days if str(d) < cut])
    return {"oos_start": cut,
            "in_sample": compute_metrics(ins, capital, n_is),
            "out_of_sample": compute_metrics(oos, capital, len(days) - n_is),
            "monte_carlo": monte_carlo_drawdown(trades.net_pnl.to_numpy(), capital) if len(trades) else {},
            "bootstrap": bootstrap_expectancy(trades.net_pnl.to_numpy()) if len(trades) else {}}
