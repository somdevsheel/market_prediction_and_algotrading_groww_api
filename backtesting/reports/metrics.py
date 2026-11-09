"""Backtest / ledger performance metrics. Definitions are in docs/BACKTESTING_METHODOLOGY.md."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _streaks(wins: list[bool]) -> tuple[int, int]:
    best_w = best_l = cur_w = cur_l = 0
    for w in wins:
        cur_w, cur_l = (cur_w + 1, 0) if w else (0, cur_l + 1)
        best_w, best_l = max(best_w, cur_w), max(best_l, cur_l)
    return best_w, best_l


def equity_curve(trades: pd.DataFrame, capital: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["date", "pnl", "equity", "drawdown"])
    d = trades.groupby("trading_day")["net_pnl"].sum().sort_index()
    eq = capital + d.cumsum()
    peak = eq.cummax().clip(lower=capital)
    return pd.DataFrame({"date": d.index.astype(str), "pnl": d.values, "equity": eq.values,
                         "drawdown": (eq - peak).values})


def compute_metrics(trades: pd.DataFrame, capital: float, trading_days: int | None = None) -> dict:
    """``trades`` needs columns: trading_day, net_pnl, gross_pnl, charges, option_type, entry_time, exit_time."""
    if trades is None or trades.empty:
        return {"trades": 0, "note": "no trades"}
    t = trades.copy()
    n = len(t)
    wins, losses = t[t.net_pnl > 0], t[t.net_pnl <= 0]
    net, gross, charges = float(t.net_pnl.sum()), float(t.gross_pnl.sum()), float(t.charges.sum())
    gp, gl = float(wins.net_pnl.sum()), float(-losses.net_pnl.sum())
    eq = equity_curve(t, capital)
    max_dd = float(eq.drawdown.min()) if len(eq) else 0.0
    peak_eq = float(np.maximum.accumulate(np.r_[capital, eq.equity.values]).max())
    days = trading_days or len(eq)
    # daily return series over all days in sample (days without trades = 0)
    daily = np.zeros(max(days, len(eq)))
    daily[:len(eq)] = eq.pnl.values / capital
    sd = daily.std(ddof=1) if len(daily) > 1 else 0
    dd_sd = daily[daily < 0].std(ddof=1) if (daily < 0).sum() > 1 else 0
    years = days / 252 if days else 0
    final = capital + net
    cagr = (final / capital) ** (1 / years) - 1 if years >= 0.5 and final > 0 else None
    bw, bl = _streaks(list(t.net_pnl > 0))
    hold = None
    if "entry_time" in t and "exit_time" in t:
        h = (pd.to_datetime(t.exit_time) - pd.to_datetime(t.entry_time)).dt.total_seconds() / 60
        hold = float(h.mean())
    by = lambda col: {str(k): {"trades": int(len(g)), "net_pnl": round(float(g.net_pnl.sum()), 2),  # noqa: E731
                               "win_rate": round(float((g.net_pnl > 0).mean() * 100), 1)} for k, g in t.groupby(col)}
    t["month"] = pd.to_datetime(t.trading_day).dt.strftime("%Y-%m")
    t["year"] = pd.to_datetime(t.trading_day).dt.year
    out = {
        "trades": n, "net_pnl": round(net, 2), "gross_pnl": round(gross, 2), "charges": round(charges, 2),
        "roi_pct": round(net / capital * 100, 2), "cagr_pct": round(cagr * 100, 2) if cagr is not None else None,
        "win_rate_pct": round(len(wins) / n * 100, 2), "loss_rate_pct": round(len(losses) / n * 100, 2),
        "avg_win": round(float(wins.net_pnl.mean()), 2) if len(wins) else 0.0,
        "avg_loss": round(float(losses.net_pnl.mean()), 2) if len(losses) else 0.0,
        "expectancy": round(net / n, 2),
        "profit_factor": round(gp / gl, 3) if gl > 0 else None,
        "max_drawdown": round(max_dd, 2), "max_drawdown_pct": round(max_dd / peak_eq * 100, 2) if peak_eq else None,
        "sharpe": round(daily.mean() / sd * math.sqrt(252), 3) if sd > 0 else None,
        "sortino": round(daily.mean() / dd_sd * math.sqrt(252), 3) if dd_sd > 0 else None,
        "calmar": round((cagr or 0) / abs(max_dd / peak_eq), 3) if max_dd < 0 and cagr is not None else None,
        "max_consecutive_wins": bw, "max_consecutive_losses": bl,
        "avg_holding_minutes": round(hold, 1) if hold is not None else None,
        "cost_impact_pct_of_gross": round(charges / abs(gross) * 100, 2) if gross else None,
        "trading_days_in_sample": days,
        "by_option_type": by("option_type"), "by_month": by("month"), "by_year": by("year"),
    }
    if "regime" in t:
        out["by_regime"] = by("regime")
    if "expiry_day" in t:
        out["by_expiry_day"] = by("expiry_day")
    if "exit_reason" in t:
        out["by_exit_reason"] = by("exit_reason")
    if "hmm_label" in t and t["hmm_label"].notna().any():
        out["by_hmm"] = by("hmm_label")
    if "days_to_expiry" in t:
        t["dte_bucket"] = pd.cut(t["days_to_expiry"], [-1, 0, 1, 3, 7, 100], labels=["0", "1", "2-3", "4-7", ">7"])
        out["by_days_to_expiry"] = by("dte_bucket")
    return out
