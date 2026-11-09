"""Walk-forward parameter selection, parameter sensitivity, cost and slippage sensitivity.

Efficient design: each parameter combination is simulated ONCE over the full period (signals are causal and
days are independent apart from capital, so research capital is used). Walk-forward then selects, for each
fold, the combination with the best *training-window* score and records its *following test-window* trades.
Only the concatenated test-window trades are reported as walk-forward out-of-sample performance.
"""
from __future__ import annotations

import pandas as pd

from backtesting.reports.metrics import compute_metrics


def score(trades: pd.DataFrame, min_trades: int = 15) -> float:
    if trades is None or len(trades) < min_trades:
        return float("-inf")
    return float(trades.net_pnl.mean())  # expectancy per trade


def walk_forward(trades_by_combo: dict[str, pd.DataFrame], days: list, train_months: int = 12,
                 test_months: int = 3, capital: float = 20000.0, min_trades: int = 15) -> dict:
    months = sorted({pd.Timestamp(d).to_period("M") for d in days})
    folds, oos = [], []
    i = train_months
    while i < len(months):
        tr = (months[i - train_months], months[i - 1])
        te = months[i:i + test_months]
        best, best_s = None, float("-inf")
        for k, t in trades_by_combo.items():
            if t is None or t.empty:
                continue
            m = pd.to_datetime(t.trading_day).dt.to_period("M")
            s = score(t[(m >= tr[0]) & (m <= tr[1])], min_trades)
            if s > best_s:
                best, best_s = k, s
        test_tr = pd.DataFrame()
        if best is not None:
            t = trades_by_combo[best]
            m = pd.to_datetime(t.trading_day).dt.to_period("M")
            test_tr = t[m.isin(te)]
            oos.append(test_tr)
        folds.append({"train": f"{tr[0]}..{tr[1]}", "test": f"{te[0]}..{te[-1]}", "chosen": best,
                      "train_expectancy": None if best is None else round(best_s, 2),
                      "test_trades": len(test_tr), "test_net": round(float(test_tr.net_pnl.sum()), 2) if len(test_tr) else 0.0})
        i += test_months
    allo = pd.concat(oos, ignore_index=True) if oos else pd.DataFrame()
    return {"folds": folds, "oos_metrics": compute_metrics(allo, capital) if len(allo) else {"trades": 0}}


def sensitivity_table(results: dict[str, pd.DataFrame], capital: float) -> list[dict]:
    rows = []
    for k, t in results.items():
        m = compute_metrics(t, capital) if t is not None and len(t) else {"trades": 0}
        rows.append({"combo": k, **{x: m.get(x) for x in ("trades", "net_pnl", "win_rate_pct", "profit_factor",
                                                          "expectancy", "max_drawdown", "sharpe")}})
    return rows


def cost_sensitivity(trades: pd.DataFrame, capital: float, multipliers=(0.0, 0.5, 1.0, 1.5, 2.0)) -> list[dict]:
    rows = []
    for m in multipliers:
        t = trades.copy()
        t["net_pnl"] = t.gross_pnl - t.charges * m
        mm = compute_metrics(t, capital)
        rows.append({"charges_x": m, "net_pnl": mm["net_pnl"], "expectancy": mm["expectancy"],
                     "profit_factor": mm["profit_factor"]})
    return rows
