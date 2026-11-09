"""Performance metrics (all net of charges; spread/slippage are inside fill prices and also reported)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def daily(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    return trades.groupby(pd.to_datetime(trades["exit_time"]).dt.date)["net_pnl"].sum()


def metrics(trades: pd.DataFrame, capital: float, n_sessions: int) -> dict:
    if trades is None or trades.empty:
        return {"trades": 0}
    n = trades["net_pnl"]
    w, l = n[n > 0], n[n <= 0]
    d = daily(trades)
    dr = np.zeros(max(n_sessions, len(d)))
    dr[:len(d)] = d.to_numpy() / capital
    eq = capital + n.cumsum()
    peak = np.maximum.accumulate(np.r_[capital, eq.to_numpy()])[1:]
    dd = eq.to_numpy() - peak
    sd = dr.std(ddof=1) if len(dr) > 1 else 0
    neg = dr[dr < 0]
    notional = (trades["entry"] * trades["qty"] + trades["exit"] * trades["qty"]).sum()
    return {
        "trades": int(len(n)), "net_pnl": round(float(n.sum()), 0), "roi_pct": round(float(n.sum() / capital * 100), 2),
        "win_rate_pct": round(float((n > 0).mean() * 100), 1),
        "profit_factor": round(float(w.sum() / -l.sum()), 3) if l.sum() < 0 else None,
        "sharpe": round(float(dr.mean() / sd * math.sqrt(252)), 2) if sd > 0 else None,
        "sortino": round(float(dr.mean() / neg.std(ddof=1) * math.sqrt(252)), 2) if len(neg) > 1 and neg.std(ddof=1) > 0 else None,
        "max_dd": round(float(dd.min()), 0), "max_dd_pct": round(float((dd / peak).min() * 100), 2),
        "avg_hold_min": round(float(trades["hold_minutes"].mean()), 1),
        "expectancy": round(float(n.mean()), 1),
        "gross_before_charges": round(float(trades["gross_pnl"].sum()), 0),
        "charges": round(float(trades["charges"].sum()), 0), "slippage": round(float(trades["slippage_cost"].sum()), 0),
        "avg_mae_pts": round(float(trades["mae_pts"].mean()), 2), "avg_mfe_pts": round(float(trades["mfe_pts"].mean()), 2),
        "turnover_cr": round(float(notional / 1e7), 1), "trades_per_day": round(len(n) / max(n_sessions, 1), 2),
    }


def monthly(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    return trades.groupby(pd.to_datetime(trades["exit_time"]).dt.to_period("M"))["net_pnl"].sum()
