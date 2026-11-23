"""Earnings tilt check on saved out-of-sample predictions (no retraining).

final score = rank(P(BUY) - P(SELL)) - 0.5 + W * mean(rank of SUE, revenue YoY, profit YoY, margin change)
The BUY/HOLD/SELL signal is unchanged; the tilt only re-orders which BUY stocks fill the top 10.
W = 0.5 was fixed on 2019-21; every later year is an unseen check.

    python3 -m ml_lab.fundamentals_tilt results/fund_<run>      (the run's predictions.parquet)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml_lab.fundamentals_test import add_fundamentals
from ml_lab.stocks_bhs import backtest, build, load_panels, perf

W = 0.5
GROWTH = ["f_sue", "f_rev_yoy", "f_np_yoy", "f_margin_chg"]
PERIODS = {"2019-21 (weight chosen)": (2019, 2021), "2022-24 (unseen)": (2022, 2024),
           "2025-26 (new data, unseen)": (2025, 2026), "all 2019-26": (2019, 2026)}


def main() -> int:
    run = Path(sys.argv[1])
    T0 = pd.read_parquet(run / "predictions.parquet")
    T0 = T0[T0.variant == "price_only"].drop(columns="variant")
    P, jump, nifty = load_panels()
    L, _, seq = build(P, jump, nifty)
    del seq
    L = add_fundamentals(L, P["close"])
    T0 = T0.merge(L[["date", "sym"] + GROWTH], on=["date", "sym"], how="left")
    T0[GROWTH] = T0[GROWTH].fillna(0.0)
    base = T0.groupby("date").score.rank(pct=True) - 0.5
    T1 = T0.assign(score=base + W * T0[GROWTH].mean(axis=1))
    opens = P["open"].ffill(limit=10)
    rows = []
    for label, (a, b) in PERIODS.items():
        for name, T in (("model only", T0), ("model + earnings tilt", T1)):
            t = T[T.date.dt.year.between(a, b)]
            eq, _, ch = backtest(t, opens)
            s = perf(eq.equity)
            k = t.fwd.notna()
            top = t[k].groupby("date").apply(lambda g: g[g.signal == 2].nlargest(10, "score").fwd.mean())
            rows.append({"period": label, "variant": name, "months": t.date.nunique(),
                         "fund_coverage_pct": round((t[GROWTH].abs().sum(axis=1) > 0).mean() * 100),
                         "top10_BUY_avg_month_pct": round(top.mean() * 100, 2),
                         "cagr_net_pct": s["cagr_pct"], "sharpe": s["sharpe"], "max_dd_pct": s["max_dd_pct"]})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    out.to_csv(run / "tilt_check.csv", index=False)
    print(f"-> {run / 'tilt_check.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
