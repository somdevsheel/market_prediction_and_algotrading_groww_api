"""Pre-registered time-of-day scalping ideas on 1-minute NIFTY futures (same engine, costs and validation).

Rules and grids are fixed here BEFORE looking at results; parameters are chosen on train (<= 2023) only.
    cd "new backktest method" && python3 -m orderflow_scalping.experiments.scalp_ideas
"""
from __future__ import annotations

import copy
import itertools
import json
from datetime import date

import numpy as np
import pandas as pd
import yaml

from orderflow_scalping._paths import EXP_ROOT, MAIN_ROOT, RESULTS
from orderflow_scalping.analytics.performance import metrics
from orderflow_scalping.backtesting import engine
from orderflow_scalping.data.preprocessing.features import build
from orderflow_scalping.validation.statistical_tests import bootstrap_mean, verdict


def at_minute(df, hhmm):  # boolean mask of the bar whose CLOSE is at hh:mm (bar start = hh:mm - 1)
    h, m = map(int, hhmm.split(":"))
    return (df["minute"] == h * 60 + m - 1).to_numpy()


def day_open(df):
    return df.groupby("date")["open"].transform("first")


def prev_close(df):
    c = df.groupby("date")["close"].last()
    return df["date"].map(c.shift(1))


def opening_drive(df, p):  # first 5 minutes' move vs ATR -> follow it at 09:20 close
    t = at_minute(df, "09:20")
    mv = df["close"] - day_open(df)
    sig = np.zeros(len(df), np.int8)
    sig[t & (mv > p["k"] * df["atr"]).to_numpy()] = 1
    sig[t & (mv < -p["k"] * df["atr"]).to_numpy()] = -1
    return sig, None, None


def orb5(df, p):  # first 1-minute close beyond the 09:15-09:20 range (09:20-10:30)
    d = df["date"]
    first5 = df["minute"] < 9 * 60 + 20
    hi = df["high"].where(first5).groupby(d).transform("max")
    lo = df["low"].where(first5).groupby(d).transform("min")
    win = (df["minute"] >= 9 * 60 + 20) & (df["minute"] < 10 * 60 + 30)
    up = (df["close"] > hi + p["buf"]) & win
    dn = (df["close"] < lo - p["buf"]) & win
    first_up = up & (up.astype(int).groupby(d).cumsum() == 1)
    first_dn = dn & (dn.astype(int).groupby(d).cumsum() == 1)
    taken = (first_up | first_dn).astype(int).groupby(d).cumsum()
    sig = np.zeros(len(df), np.int8)
    sig[(first_up & (taken == 1)).to_numpy()] = 1
    sig[(first_dn & (taken == 1)).to_numpy()] = -1
    return sig, None, None


def gap_fade(df, p):  # gap > x% vs previous close -> fade at 09:20 close
    t = at_minute(df, "09:20")
    gap = (day_open(df) / prev_close(df) - 1) * 100
    sig = np.zeros(len(df), np.int8)
    sig[t & (gap > p["gap_pct"]).to_numpy()] = -1
    sig[t & (gap < -p["gap_pct"]).to_numpy()] = 1
    return sig, None, None


def last_hour(df, p):  # day's direction at 14:30 (vs open, in ATR units) continues to 15:20
    t = at_minute(df, "14:30")
    mv = df["close"] - day_open(df)
    sig = np.zeros(len(df), np.int8)
    sig[t & (mv > p["k"] * df["atr"] * 10).to_numpy()] = 1
    sig[t & (mv < -p["k"] * df["atr"] * 10).to_numpy()] = -1
    return sig, None, None


def expiry_afternoon(df, p, expiries):  # expiry days only: 13:30 direction vs VWAP continues
    t = at_minute(df, "13:30") & df["date"].isin(expiries).to_numpy()
    z = df["vwap_z"]
    sig = np.zeros(len(df), np.int8)
    sig[t & (z > p["z"]).to_numpy()] = 1
    sig[t & (z < -p["z"]).to_numpy()] = -1
    return sig, None, None


IDEAS = {  # name: (fn, grid, max_holding_bars, stop_atr, target_atr)
    "opening_drive": (opening_drive, {"k": [1.0, 2.0, 3.0]}, 30),
    "orb_5min_futures": (orb5, {"buf": [0, 5, 10]}, 20),
    "gap_fade": (gap_fade, {"gap_pct": [0.3, 0.5, 0.8]}, 30),
    "last_hour_trend": (last_hour, {"k": [0.5, 1.0, 1.5]}, 50),
    "expiry_afternoon": (expiry_afternoon, {"z": [0.5, 1.0, 1.5]}, 100),
}


def main() -> int:
    S = yaml.safe_load((EXP_ROOT / "config" / "settings.yaml").read_text())
    S["session"]["first_entry"] = "09:19"  # allow the 09:20 decision bar
    df, _ = build(date.fromisoformat(S["data"]["start"]), date.fromisoformat(S["data"]["end"]))
    tr_end = S["validation"]["train_end"]
    sessions = sorted(df["date"].unique())
    n_tr = sum(1 for d in sessions if str(d) <= tr_end)
    cache = json.loads((MAIN_ROOT / "data_lake" / "manifests" / "expiries_cache.json").read_text())
    expiries = {date.fromisoformat(x) for v in cache.values() for x in v}
    rows, n_tests = [], 83 + sum(len(list(itertools.product(*g.values()))) for _, g, _ in IDEAS.values())
    for name, (fn, grid, hold) in IDEAS.items():
        s = copy.deepcopy(S)
        s["risk"].update(max_holding_bars=hold, stop_atr_mult=3.0, target_atr_mult=6.0, max_trades_per_day=1)
        best = None
        for combo in itertools.product(*grid.values()):
            p = dict(zip(grid, combo))
            sig, xl, xs = fn(df, p, expiries) if name == "expiry_afternoon" else fn(df, p)
            t = engine.run(df, sig, s, "base", xl, xs, name)
            if t.empty:
                continue
            dte = pd.to_datetime(t.exit_time).dt.strftime("%Y-%m-%d")
            a, b = t[dte <= tr_end], t[dte > tr_end]
            pre = (t.gross_pnl + t.slippage_cost) / t.qty
            r = {"idea": name, "params": p, "trades": len(t), "train_net": round(a.net_pnl.sum()), "test_net": round(b.net_pnl.sum()),
                 "test_trades": len(b), "pre_cost_pts_train": round(pre[dte <= tr_end].mean(), 2),
                 "pre_cost_pts_test": round(pre[dte > tr_end].mean(), 2), "train_exp": a.net_pnl.mean() if len(a) else -1e9,
                 "test_pf": metrics(b, S["risk"]["capital"], len(sessions) - n_tr).get("profit_factor"),
                 "test_boot": bootstrap_mean(b.net_pnl.to_numpy())}
            rows.append(r)
            if len(a) >= 50 and (best is None or r["train_exp"] > best["train_exp"]):
                best = r
        if best:
            best["selected_on_train"] = True
    out = RESULTS / "scalp_ideas.json"
    out.write_text(json.dumps({"n_tests_total": n_tests, "rows": rows}, indent=1, default=str))
    print(f"total parameter sets incl. previous study: {n_tests}\n")
    print(f"{'idea':18s} {'params':16s} {'sel':3s} {'trades':>6s} {'pre-cost pts (train/test)':>26s} {'train net':>11s} {'TEST net':>11s} {'test PF':>7s}  verdict")
    for r in rows:
        print(f"{r['idea']:18s} {str(r['params']):16s} {'*' if r.get('selected_on_train') else '':3s} {r['trades']:6d} "
              f"{r['pre_cost_pts_train']:>12} / {r['pre_cost_pts_test']:<11} {r['train_net']:>11,} {r['test_net']:>11,} "
              f"{str(r['test_pf']):>7s}  {verdict(r['test_boot'], n_tests, 100) if r.get('selected_on_train') else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
