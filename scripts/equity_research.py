"""Swing / positional equity research: pre-declared grids, selection on train (2021-2023), out-of-sample 2024+.

    python3 scripts/equity_research.py [--capital 100000]

Writes data/backtests/equity_<ts>/ (report.json, equity curves, trades). Universe = stocks downloaded into the
data lake (see scripts/ingest_equities.py); survivorship bias applies (currently-listed stocks only).
"""
import _bootstrap  # noqa: F401

import argparse
import itertools
import json
from datetime import datetime

import pandas as pd

from core.clock import IST
from core.config import load_config, resolve_path
from equity.backtest import metrics, run
from equity.data import load_panel, nifty_daily
from equity.strategies import STRATEGIES, _eligible

START, TRAIN_END, END = "2021-01-01", "2023-12-31", "2026-11-06"
GRIDS = {
    "momentum": {"lookback": [126, 252], "skip": [21], "n": [5, 10], "market_filter": [False, True]},
    "breakout_52w": {"vol_mult": [1.5, 2.0], "n": [5, 10], "market_filter": [False, True]},
    "pullback_rsi2": {"rsi": [5, 10], "max_days": [10], "n": [5, 10], "market_filter": [False, True]},
}


def seg(eq: pd.Series, a: str, b: str) -> pd.Series:
    return eq[(eq.index >= pd.Timestamp(a)) & (eq.index <= pd.Timestamp(b))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=100_000)
    a = ap.parse_args()
    cfg = load_config()
    panel = load_panel()
    nifty = nifty_daily()
    ca = panel.pop("_corporate_actions")
    n_stocks = panel["close"].shape[1]
    print(f"corporate actions: {len(ca['adjusted'])} split/bonus adjustments, {len(ca['dropped'])} stocks dropped "
          f"for unexplained jumps")
    out = resolve_path(cfg, "backtests") / f"equity_{datetime.now(IST):%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"universe: {n_stocks} stocks, {panel['close'].index.min().date()} -> {panel['close'].index.max().date()}")
    (out / "corporate_actions.json").write_text(json.dumps(ca, indent=1))
    rep = {"universe_stocks": n_stocks, "corporate_actions": {"adjusted": len(ca["adjusted"]), "dropped": len(ca["dropped"])}, "capital": a.capital, "period": [START, END], "train_end": TRAIN_END,
           "survivorship_bias": "currently-listed stocks only; delisted companies missing -> results optimistic",
           "strategies": {}, "baselines": {}}
    # baselines
    nb = nifty[(nifty.index >= START) & (nifty.index <= END)]
    nb = nb / nb.iloc[0] * a.capital
    elig = _eligible(panel)
    ew_r = panel["close"].pct_change(fill_method=None).where(elig.shift(1).fillna(False).astype(bool)).mean(axis=1)
    ew = (1 + ew_r[(ew_r.index >= START) & (ew_r.index <= END)].fillna(0)).cumprod() * a.capital
    for name, e in (("nifty_buy_hold", nb), ("equal_weight_universe", ew)):
        rep["baselines"][name] = {"full": metrics(e), "train": metrics(seg(e, START, TRAIN_END)), "test": metrics(seg(e, "2024-01-01", END))}
    n_sets = 0
    for sname, grid in GRIDS.items():
        strat = STRATEGIES[sname]
        rows = []
        for combo in itertools.product(*grid.values()):
            p = dict(zip(grid, combo), nifty=nifty)
            res = run(panel, strat, p, a.capital, 10, p["n"], START, END)
            n_sets += 1
            e = res.equity
            rows.append({"params": {k: v for k, v in p.items() if k != "nifty"}, "full": metrics(e, res.trades),
                         "train": metrics(seg(e, START, TRAIN_END)), "test": metrics(seg(e, "2024-01-01", END)),
                         "_res": res})
            print(f"  {sname} {rows[-1]['params']}: train CAGR {rows[-1]['train']['cagr_pct']}% | test CAGR "
                  f"{rows[-1]['test']['cagr_pct']}% maxDD {rows[-1]['test']['max_dd_pct']}% | trades {rows[-1]['full'].get('trades')}", flush=True)
        best = max(rows, key=lambda r: r["train"]["sharpe"] or -9)
        p = dict(best["params"], nifty=nifty)
        cons = run(panel, strat, p, a.capital, 25, p["n"], START, END)
        best["_res"].equity.to_csv(out / f"{sname}_equity.csv", header=["value"])
        best["_res"].trades.to_csv(out / f"{sname}_trades.csv", index=False)
        rep["strategies"][sname] = {"selected_on_train": best["params"], "full": best["full"], "train": best["train"],
                                    "test": best["test"], "test_conservative_slippage": metrics(seg(cons.equity, "2024-01-01", END)),
                                    "grid": [{k: v for k, v in r.items() if k != "_res"} for r in rows]}
    rep["parameter_sets_tested"] = n_sets
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print("\nOUT-OF-SAMPLE 2024-01-01 -> 2026-11-06 (selected on 2021-2023):")
    print(f"{'strategy':24s} {'CAGR%':>6s} {'maxDD%':>7s} {'Sharpe':>6s} {'Calmar':>6s}   train CAGR%   full trades  win%   conservative CAGR%")
    for k, v in rep["baselines"].items():
        t = v["test"]
        print(f"{k:24s} {t['cagr_pct']:6} {t['max_dd_pct']:7} {str(t['sharpe']):>6} {str(t['calmar']):>6}   {v['train']['cagr_pct']:>10}")
    for k, v in rep["strategies"].items():
        t = v["test"]
        print(f"{k:24s} {t['cagr_pct']:6} {t['max_dd_pct']:7} {str(t['sharpe']):>6} {str(t['calmar']):>6}   {v['train']['cagr_pct']:>10}   "
              f"{v['full'].get('trades'):>10} {v['full'].get('win_rate_pct'):>5}   {v['test_conservative_slippage']['cagr_pct']:>8}   {v['selected_on_train']}")
    print("report ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
