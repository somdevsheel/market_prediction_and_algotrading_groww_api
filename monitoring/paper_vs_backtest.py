"""Compare paper execution with backtest assumptions (ledger only).

The backtest charges, per side, half-spread% x price + slippage ticks relative to the candle price (~mid).
Paper fills are measured the same way: entry cost = fill - mid(bid, ask at decision). The scenario whose
assumption is closest to the observed cost is reported, so backtests can use the realistic scenario.
"""
from __future__ import annotations

import json

import pandas as pd


def compare(trades: list[dict], scenarios: dict, tick: float = 0.05) -> dict:
    t = pd.DataFrame([x for x in trades if x.get("status") == "CLOSED"])
    if t.empty:
        return {"trades": 0, "note": "no closed paper trades yet"}
    det = t["details"].apply(lambda d: json.loads(d) if isinstance(d, str) and d else (d or {}))
    q = det.apply(lambda d: d.get("entry_quote") or {})
    mid = q.apply(lambda x: (x["bid"] + x["ask"]) / 2 if x.get("bid") and x.get("ask") else None)
    t["entry_cost_vs_mid"] = t["entry_price"] - mid
    t["exit_slip_vs_expected"] = t["expected_exit"] - t["exit_price"]
    px = float(t.entry_price.mean())
    assumed = {k: round(px * v["spread_pct"] / 200 + v["slippage_ticks"] * tick, 3) for k, v in scenarios.items()}
    obs = t["entry_cost_vs_mid"].dropna()
    out = {"trades": len(t), "net_pnl": round(float(t.net_pnl.sum()), 2),
           "avg_entry_cost_vs_mid_pts": round(float(obs.mean()), 3) if len(obs) else None,
           "avg_exit_slip_vs_expected_pts": round(float(t.exit_slip_vs_expected.dropna().mean()), 3)
           if t.exit_slip_vs_expected.notna().any() else None,
           "avg_charges": round(float(t.charges.mean()), 2), "assumed_cost_per_side_pts": assumed}
    if len(obs):
        out["closest_scenario"] = min(assumed, key=lambda k: abs(assumed[k] - float(obs.mean())))
    out["by_exit_reason"] = {k: {"count": int(g.shape[0]), "net": round(float(g.net_pnl.sum()), 2)}
                             for k, g in t.groupby("exit_reason")}
    return out
