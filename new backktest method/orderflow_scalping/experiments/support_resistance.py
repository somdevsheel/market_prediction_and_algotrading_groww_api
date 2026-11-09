"""Pre-registered support/resistance test on 1-minute NIFTY futures (same engine, costs, train/test split).

Levels (known before the bar): previous day high/low (PDH/PDL), opening range 09:15-09:45 high/low (after 09:45),
or both. Styles: BREAKOUT (close beyond level + buffer -> follow) and BOUNCE (bar touches level within tolerance and
closes back on the near side with a rejection candle -> fade). One signal per level per day, max 3 trades/day.
    cd "new backktest method" && python3 -m orderflow_scalping.experiments.support_resistance
"""
from __future__ import annotations

import copy
import itertools
import json
from datetime import date

import numpy as np
import pandas as pd
import yaml

from orderflow_scalping._paths import EXP_ROOT, RESULTS
from orderflow_scalping.analytics.performance import metrics
from orderflow_scalping.backtesting import engine
from orderflow_scalping.data.preprocessing.features import build
from orderflow_scalping.validation.statistical_tests import bootstrap_mean, verdict


def levels(df: pd.DataFrame, which: str) -> list[tuple[pd.Series, str]]:
    d = df["date"]
    out = []
    if which in ("prev_day", "both"):
        hi, lo = df.groupby("date")["high"].max().shift(1), df.groupby("date")["low"].min().shift(1)
        out += [(d.map(hi), "res"), (d.map(lo), "sup")]
    if which in ("opening_range", "both"):
        orm = df["minute"] < 9 * 60 + 45
        oh = df["high"].where(orm).groupby(d).transform("max").where(~orm)
        ol = df["low"].where(orm).groupby(d).transform("min").where(~orm)
        out += [(oh, "res"), (ol, "sup")]
    return out


def first_per_day(cond: pd.Series, d: pd.Series) -> pd.Series:
    c = cond.fillna(False)
    return c & (c.astype(int).groupby(d).cumsum() == 1)


def signals(df: pd.DataFrame, p: dict) -> np.ndarray:
    d = df["date"]
    sig = np.zeros(len(df), np.int8)
    for lvl, kind in levels(df, p["levels"]):
        b = p["buffer"]
        if p["style"] == "breakout":
            cond = (df["close"] > lvl + b) if kind == "res" else (df["close"] < lvl - b)
            side = 1 if kind == "res" else -1
        else:  # bounce: touch within tolerance and close back on the near side with a rejection body
            if kind == "sup":
                cond = (df["low"] <= lvl + b) & (df["close"] > lvl) & (df["close"] > df["open"])
                side = 1
            else:
                cond = (df["high"] >= lvl - b) & (df["close"] < lvl) & (df["close"] < df["open"])
                side = -1
        f = first_per_day(cond, d).to_numpy()
        sig[f & (sig == 0)] = side
    return sig


GRID = {"style": ["breakout", "bounce"], "levels": ["prev_day", "opening_range", "both"], "buffer": [0, 5, 10]}


def main() -> int:
    S = yaml.safe_load((EXP_ROOT / "config" / "settings.yaml").read_text())
    df, _ = build(date.fromisoformat(S["data"]["start"]), date.fromisoformat(S["data"]["end"]))
    tr_end = S["validation"]["train_end"]
    sessions = sorted(df["date"].unique())
    n_te = sum(1 for x in sessions if str(x) > tr_end)
    s = copy.deepcopy(S)
    s["risk"].update(max_holding_bars=60, stop_atr_mult=2.0, target_atr_mult=4.0, max_trades_per_day=3)
    n_tests = 98 + len(list(itertools.product(*GRID.values())))
    rows = []
    for combo in itertools.product(*GRID.values()):
        p = dict(zip(GRID, combo))
        t = engine.run(df, signals(df, p), s, "base", label="sr")
        dte = pd.to_datetime(t.exit_time).dt.strftime("%Y-%m-%d")
        a, b = t[dte <= tr_end], t[dte > tr_end]
        pre = (t.gross_pnl + t.slippage_cost) / t.qty
        cons = engine.run(df, signals(df, p), s, "conservative", label="sr")
        rows.append({**p, "trades": len(t), "pre_pts_train": round(pre[dte <= tr_end].mean(), 2),
                     "pre_pts_test": round(pre[dte > tr_end].mean(), 2), "train_net": round(a.net_pnl.sum()),
                     "test_net": round(b.net_pnl.sum()), "test_pf": metrics(b, S["risk"]["capital"], n_te).get("profit_factor"),
                     "test_win": round((b.net_pnl > 0).mean() * 100, 1),
                     "test_net_conservative": round(cons[pd.to_datetime(cons.exit_time).dt.strftime("%Y-%m-%d") > tr_end].net_pnl.sum()),
                     "train_exp": a.net_pnl.mean(), "boot": bootstrap_mean(b.net_pnl.to_numpy())})
    for style in GRID["style"]:
        cand = [r for r in rows if r["style"] == style]
        max(cand, key=lambda r: r["train_exp"])["selected"] = True
    (RESULTS / "support_resistance.json").write_text(json.dumps({"n_tests_total": n_tests, "rows": rows}, indent=1, default=str))
    print(f"parameter sets so far (all studies): {n_tests}\n")
    print(f"{'style':9s} {'levels':14s} {'buf':>3s} {'sel':3s} {'trades':>6s} {'pre-cost pts tr/te':>19s} {'train net':>11s} {'TEST net':>11s} {'PF':>6s} {'win%':>5s}  verdict")
    for r in rows:
        print(f"{r['style']:9s} {r['levels']:14s} {r['buffer']:>3} {'*' if r.get('selected') else '':3s} {r['trades']:6d} "
              f"{r['pre_pts_train']:>8} / {r['pre_pts_test']:<8} {r['train_net']:>11,} {r['test_net']:>11,} {str(r['test_pf']):>6s} "
              f"{r['test_win']:>5}  {verdict(r['boot'], n_tests, 100) if r.get('selected') else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
