"""Breakdowns: by regime, time of day, direction, exit reason; cost breakdown."""
from __future__ import annotations

import pandas as pd


def _agg(g):
    return pd.Series({"trades": len(g), "net_pnl": round(g.net_pnl.sum(), 0), "win_rate": round((g.net_pnl > 0).mean() * 100, 1),
                      "expectancy": round(g.net_pnl.mean(), 1)})


def breakdowns(t: pd.DataFrame) -> dict:
    if t.empty:
        return {}
    t = t.copy()
    t["hour"] = pd.to_datetime(t["entry_time"]).dt.hour
    t["direction"] = t["side"].map({1: "long", -1: "short"})
    return {k: t.groupby(col).apply(_agg, include_groups=False).reset_index().to_dict("records")
            for k, col in (("by_regime", "regime"), ("by_hour", "hour"), ("by_direction", "direction"),
                           ("by_exit_reason", "exit_reason"))}


def cost_breakdown(t: pd.DataFrame) -> dict:
    if t.empty:
        return {}
    cols = [c for c in t.columns if c.startswith("ch_")]
    out = {c[3:]: round(float(t[c].sum()), 0) for c in cols}
    out["charges_total"] = round(float(t["charges"].sum()), 0)
    out["spread_and_slippage"] = round(float(t["slippage_cost"].sum()), 0)
    out["per_trade_all_in"] = round(float((t["charges"] + t["slippage_cost"]).mean()), 1)
    return out
