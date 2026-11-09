"""Bootstrap confidence intervals / p-values and trade-count sufficiency; multiple-testing correction."""
from __future__ import annotations

import numpy as np


def bootstrap_mean(x: np.ndarray, runs: int = 2000, seed: int = 7) -> dict:
    x = np.asarray(x, float)
    if len(x) < 2:
        return {"n": int(len(x))}
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (runs, len(x)), replace=True).mean(axis=1)
    return {"n": int(len(x)), "mean": round(float(x.mean()), 1), "ci05": round(float(np.percentile(means, 5)), 1),
            "ci95": round(float(np.percentile(means, 95)), 1),
            "p_mean_le_0": round(float((means <= 0).mean()), 4)}


def verdict(test_boot: dict, n_tests: int, min_trades: int, alpha: float = 0.05) -> str:
    if test_boot.get("n", 0) < min_trades:
        return f"INSUFFICIENT TRADES ({test_boot.get('n', 0)} < {min_trades})"
    p = test_boot["p_mean_le_0"]
    if p <= alpha / max(n_tests, 1):
        return f"EDGE (OOS p={p:.4f} ≤ Bonferroni {alpha}/{n_tests})"
    if p <= alpha:
        return f"NOT ROBUST (OOS p={p:.4f} passes 5% but fails multiple-testing correction)"
    return f"NO EDGE (OOS p={p:.3f})"
