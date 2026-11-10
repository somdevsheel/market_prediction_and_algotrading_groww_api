"""Intraday equity research ("stocks in play" ORB / opening drive) on the liquid universe's 5-minute candles.

    python3 scripts/equity_intraday_research.py [--capital 100000]

Selection: each day the top-K stocks by first-bar relative volume (rel_vol >= 1, price >= 50). Size: risk = 0.5%
of capital per trade (stop distance), notional capped at capital x leverage / K. Charges: Groww intraday schedule.
Universe chosen by CURRENT liquidity (mild look-ahead) and currently-listed stocks only (survivorship) -> optimistic.
"""
import _bootstrap  # noqa: F401

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

from core.clock import IST
from core.config import PROJECT_ROOT, load_config, resolve_path
from equity.costs import INTRADAY, order_charges
from equity.intraday import day_table

TRAIN_END = "2023-12-31"
GRID = {"setup": ["orb5", "drive15"], "k": [10, 20], "stop_mult": [0.05, 0.10, 0.20], "leverage": [1, 3]}


def build_table(syms: list[str]) -> pd.DataFrame:
    cache = resolve_path(load_config(), "data_lake") / "gold" / "equity" / "intraday_day_table.parquet"
    if cache.exists():
        t = pd.read_parquet(cache)
        if set(syms) <= set(t["sym"].unique()):
            return t[t["sym"].isin(syms)]
    with ProcessPoolExecutor(8) as ex:
        parts = [p for p in ex.map(day_table, syms) if len(p)]
    t = pd.concat(parts, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(cache, index=False)
    return t


def simulate(t: pd.DataFrame, p: dict, capital: float, slip_bps: float) -> pd.DataFrame:
    x = t[(t.setup == p["setup"]) & (t.stop_mult == p["stop_mult"]) & (t.rel_vol >= 1) & (t.price >= 50)]
    x = x.sort_values(["date", "rel_vol"], ascending=[True, False]).groupby("date").head(p["k"]).copy()
    slip = slip_bps / 1e4
    x["entry_f"] = x.entry * (1 + x.side * slip)
    x["exit_f"] = x.exit * (1 - x.side * slip)
    risk_pts = (x.stop_mult * x.atr14).clip(lower=0.01)
    qty_risk = np.floor(capital * 0.005 / risk_pts)
    qty_cap = np.floor(capital * p["leverage"] / p["k"] / x.entry_f)
    x["qty"] = np.minimum(qty_risk, qty_cap)
    x = x[x.qty >= 1]
    buy_v = np.where(x.side > 0, x.entry_f, x.exit_f) * x.qty
    sell_v = np.where(x.side > 0, x.exit_f, x.entry_f) * x.qty
    x["charges"] = [order_charges(b, "buy", INTRADAY) + order_charges(s, "sell", INTRADAY) for b, s in zip(buy_v, sell_v)]
    x["gross"] = x.side * (x.exit_f - x.entry_f) * x.qty
    x["net"] = x.gross - x.charges
    x["pre_cost_bps"] = x.side * (x.exit / x.entry - 1) * 1e4
    return x


def stats(x: pd.DataFrame, capital: float) -> dict:
    if x.empty:
        return {"trades": 0}
    d = x.groupby("date").net.sum()
    eq = capital + d.cumsum()
    dd = (eq - eq.cummax()) / eq.cummax()
    r = d / capital
    w, l = x.net[x.net > 0], x.net[x.net <= 0]
    return {"trades": int(len(x)), "days": int(len(d)), "net": round(float(x.net.sum())), "gross": round(float(x.gross.sum())),
            "charges": round(float(x.charges.sum())), "return_pct": round(float(x.net.sum() / capital * 100), 1),
            "win_rate": round(float((x.net > 0).mean() * 100), 1),
            "profit_factor": round(float(w.sum() / -l.sum()), 3) if l.sum() < 0 else None,
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2) if r.std() > 0 else None,
            "max_dd_pct": round(float(dd.min() * 100), 1), "pre_cost_bps_per_trade": round(float(x.pre_cost_bps.mean()), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=100_000)
    a = ap.parse_args()
    syms = (PROJECT_ROOT / "data" / "universe_intraday.txt").read_text().split()
    root = resolve_path(load_config(), "data_lake") / "silver" / "equity"
    have = [s for s in syms if (root / f"symbol={s}" / "interval=5minute").exists()]
    t = build_table(have)
    print(f"universe {len(have)}/{len(syms)} stocks with 5-min data | day-table rows {len(t):,} | "
          f"{t.date.min()} -> {t.date.max()}")
    rows = []
    for combo in itertools.product(*GRID.values()):
        p = dict(zip(GRID, combo))
        x = simulate(t, p, a.capital, 3)
        ds = x.date.astype(str)
        tr, te = x[ds <= TRAIN_END], x[ds > TRAIN_END]
        cons = simulate(t, p, a.capital, 8)
        rows.append({"params": p, "train": stats(tr, a.capital), "test": stats(te, a.capital),
                     "test_conservative": stats(cons[cons.date.astype(str) > TRAIN_END], a.capital)})
    sel = {}
    for setup in GRID["setup"]:
        c = [r for r in rows if r["params"]["setup"] == setup and r["train"].get("trades", 0) >= 200]
        if c:
            b = max(c, key=lambda r: r["train"].get("sharpe") or -99)
            b["selected"] = True
            sel[setup] = b
    out = resolve_path(load_config(), "backtests") / f"equity_intraday_{datetime.now(IST):%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps({"universe": have, "rows": rows, "train_end": TRAIN_END,
        "caveats": ["universe by current liquidity (look-ahead)", "survivorship bias", "5-min bars: intrabar path unknown"]},
        indent=1, default=str))
    print(f"\n{'setup':8s} {'K':>3s} {'stop':>5s} {'lev':>3s} sel | {'pre-cost bps tr/te':>18s} | {'train net':>10s} {'TEST net':>10s} {'test PF':>7s} {'test DD%':>8s} | {'cons. test net':>14s}")
    for r in rows:
        p, tr, te = r["params"], r["train"], r["test"]
        print(f"{p['setup']:8s} {p['k']:>3} {p['stop_mult']:>5} {p['leverage']:>3} {'*' if r.get('selected') else ' ':>3} | "
              f"{str(tr.get('pre_cost_bps_per_trade')):>8} / {str(te.get('pre_cost_bps_per_trade')):<8} | {tr.get('net', 0):>10,} "
              f"{te.get('net', 0):>10,} {str(te.get('profit_factor')):>7} {str(te.get('max_dd_pct')):>8} | {r['test_conservative'].get('net', 0):>14,}")
    print("report ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
