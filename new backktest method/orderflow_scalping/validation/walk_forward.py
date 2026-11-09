"""Walk-forward: on each fold pick the parameter set with the best TRAIN expectancy (min trades), score it on
the following TEST window only. Parameter sets are simulated once over the full period (bars are independent
across days apart from the daily risk state, which resets every session)."""
from __future__ import annotations

import pandas as pd


def walk_forward(trades_by_key: dict[str, pd.DataFrame], sessions: list, train_m: int, test_m: int,
                 min_trades: int = 30) -> tuple[pd.DataFrame, list[dict]]:
    months = sorted({pd.Timestamp(d).to_period("M") for d in sessions})
    oos, folds = [], []
    i = train_m
    while i < len(months):
        tr0, tr1, te = months[i - train_m], months[i - 1], months[i:i + test_m]
        best, best_s = None, float("-inf")
        for k, t in trades_by_key.items():
            if t.empty:
                continue
            m = pd.to_datetime(t["exit_time"]).dt.tz_localize(None).dt.to_period("M")
            s = t[(m >= tr0) & (m <= tr1)]
            if len(s) >= min_trades and s.net_pnl.mean() > best_s:
                best, best_s = k, s.net_pnl.mean()
        tt = pd.DataFrame()
        if best is not None:
            t = trades_by_key[best]
            m = pd.to_datetime(t["exit_time"]).dt.tz_localize(None).dt.to_period("M")
            tt = t[m.isin(te)]
            oos.append(tt)
        folds.append({"train": f"{tr0}..{tr1}", "test": f"{te[0]}..{te[-1]}", "chosen": best,
                      "train_expectancy": None if best is None else round(best_s, 1),
                      "test_trades": len(tt), "test_net": round(float(tt.net_pnl.sum()), 0) if len(tt) else 0})
        i += test_m
    return (pd.concat(oos, ignore_index=True) if oos else pd.DataFrame()), folds
