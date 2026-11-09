"""Order-flow scalping research runner (historical research / paper simulation only — no live orders).

    cd "new backktest method" && python3 -m orderflow_scalping.main

For every strategy: default parameters under 3 cost scenarios (+ limit-order execution), a pre-declared parameter
grid (sensitivity), parameters selected on TRAIN only and scored out-of-sample, walk-forward, bootstrap
significance with a multiple-testing correction, regime / time-of-day breakdowns and baseline comparison.
Everything is written to results/<timestamp>/ with a config snapshot and reproducibility metadata.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time as _time
from datetime import date, datetime

import numpy as np
import pandas as pd
import yaml

from orderflow_scalping._paths import EXP_ROOT, MAIN_ROOT, RESULTS
from orderflow_scalping.analytics.performance import metrics
from orderflow_scalping.analytics.report_generator import write_run
from orderflow_scalping.analytics.trade_analysis import breakdowns, cost_breakdown
from orderflow_scalping.backtesting import engine
from orderflow_scalping.data.preprocessing.features import build
from orderflow_scalping.strategies import (baselines, hybrid_strategy, microprice_strategy, ofi_strategy,
                                           volume_delta_strategy, vwap_mean_reversion, vwap_momentum)
from orderflow_scalping.validation.sensitivity import expand, key, stability
from orderflow_scalping.validation.statistical_tests import bootstrap_mean, verdict
from orderflow_scalping.validation.walk_forward import walk_forward

STRATS = {
    "ofi": ofi_strategy, "volume_delta": volume_delta_strategy, "vwap_momentum": vwap_momentum,
    "vwap_mean_reversion": vwap_mean_reversion,
}


def split(t: pd.DataFrame, train_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if t.empty:
        return t, t
    d = pd.to_datetime(t["exit_time"]).dt.strftime("%Y-%m-%d")
    return t[d <= train_end], t[d > train_end]


def main() -> int:
    t0 = _time.time()
    S = yaml.safe_load((EXP_ROOT / "config" / "settings.yaml").read_text())
    P = yaml.safe_load((EXP_ROOT / "config" / "strategy_params.yaml").read_text())
    V = S["validation"]
    start, end = date.fromisoformat(S["data"]["start"]), date.fromisoformat(S["data"]["end"])
    df, vrep = build(start, end)
    sessions = sorted(df["date"].unique())
    tr_end = V["train_end"]
    n_tr = sum(1 for d in sessions if str(d) <= tr_end)
    n_te = len(sessions) - n_tr
    cap = S["risk"]["capital"]
    train_mask = (df["date"].astype(str) <= tr_end).to_numpy()
    run_dir = RESULTS / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"data: {len(df):,} bars, {len(sessions)} sessions (train {n_tr}, test {n_te}) -> {run_dir}", flush=True)

    def sim(sig, xl, xs, scenario="base", settings=None, label=""):
        return engine.run(df, sig, settings or S, scenario, xl, xs, label)

    report: dict = {"strategies": {}, "data_validation": vrep}
    n_tests = 0
    selected_trades: dict[str, pd.DataFrame] = {}

    def evaluate(name: str, label: str, gen, grid_params: list[dict], default: dict, proxy: bool):
        nonlocal n_tests
        print(f"\n== {label}: {len(grid_params)} parameter sets", flush=True)
        by_key, rows = {}, []
        for p in grid_params:
            sig, xl, xs = gen(p)
            t = sim(sig, xl, xs, label=label)
            by_key[key(p)] = t
            a, b = split(t, tr_end)
            rows.append({"params": key(p), "train_trades": len(a), "train_net": round(float(a.net_pnl.sum()), 0) if len(a) else 0,
                         "train_exp": round(float(a.net_pnl.mean()), 1) if len(a) else None,
                         "test_trades": len(b), "test_net": round(float(b.net_pnl.sum()), 0) if len(b) else 0})
        n_tests += len(grid_params)
        cand = [r for r in rows if r["train_trades"] >= V["min_trades"] and r["train_exp"] is not None]
        sel = max(cand, key=lambda r: r["train_exp"])["params"] if cand else rows[0]["params"]
        sel_p = next(p for p in grid_params if key(p) == sel)
        t_sel = by_key[sel]
        a, b = split(t_sel, tr_end)
        sig, xl, xs = gen(default)
        scen = {}
        for sc in S["scenarios"]:
            tt = sim(sig, xl, xs, sc, label=label)
            scen[sc] = {"full": metrics(tt, cap, len(sessions)), "test": metrics(split(tt, tr_end)[1], cap, n_te)}
        lim_s = json.loads(json.dumps(S))
        lim_s["execution"]["order_type"] = "limit"
        tl = sim(sig, xl, xs, "base", lim_s, label)
        wf_t, folds = walk_forward(by_key, sessions, V["wf_train_months"], V["wf_test_months"])
        boot = bootstrap_mean(b.net_pnl.to_numpy(), V["bootstrap_runs"]) if len(b) else {"n": 0}
        write_run(run_dir / name, t_sel, cap, label, tr_end)
        selected_trades[name] = t_sel
        res = {"label": label, "mode": "PROXY (candle-based)" if proxy else "OHLCV (genuine for this signal)",
               "selected_params_on_train": sel_p, "full": metrics(t_sel, cap, len(sessions)),
               "train": metrics(a, cap, n_tr), "test": metrics(b, cap, n_te), "test_bootstrap": boot,
               "scenarios_default_params": scen,
               "limit_order_execution_default_params": {**metrics(tl, cap, len(sessions)), "nonfills": tl.attrs.get("nonfills", 0) if len(tl) else None},
               "walk_forward": {"oos": metrics(wf_t, cap, None or len(sessions)), "folds": folds},
               "sensitivity": rows, "stability": stability(rows), "breakdowns_test": breakdowns(b),
               "costs_test": cost_breakdown(b)}
        report["strategies"][name] = res
        m = res["test"]
        print(f"   selected {sel} | TRAIN net {res['train'].get('net_pnl')} | TEST net {m.get('net_pnl')} "
              f"PF {m.get('profit_factor')} trades {m.get('trades')} | WF OOS net {res['walk_forward']['oos'].get('net_pnl')}", flush=True)

    for name, mod in STRATS.items():
        cfg = P[name]
        evaluate(name, mod.LABEL, lambda p, mod=mod: mod.signals(df, p), expand(cfg["default"], cfg["grid"]),
                 cfg["default"], proxy=name in ("ofi", "volume_delta"))

    # hybrid: weights & thresholds fitted on TRAIN only, per grid point
    hp = P["hybrid"]
    models = {}

    def hyb(p):
        k = key(p)
        if k not in models:
            models[k] = hybrid_strategy.fit(df, train_mask, p)
        return hybrid_strategy.signals(df, p, models[k])
    evaluate("hybrid", hybrid_strategy.LABEL, hyb, expand(hp["default"], hp["grid"]), hp["default"], proxy=True)
    report["strategies"]["hybrid"]["fitted_models"] = models

    # microprice: not testable on candles — reported, never invented
    report["strategies"]["microprice"] = {"label": microprice_strategy.LABEL, "mode": "NOT TESTABLE",
                                          "reason": "requires historical L2 bid/ask quantities; Groww provides none. "
                                                    "Record live depth with data/ingestion/record_depth.py to test later."}

    # baselines
    for bname, gen in (("baseline_vwap_cross", lambda p: baselines.vwap_cross(df, p)),
                       ("baseline_momentum", lambda p: baselines.momentum(df, p))):
        bp = P["baselines"]["vwap_cross" if "vwap" in bname else "momentum"]
        evaluate(bname, bname.replace("_", " "), gen, [bp], bp, proxy=False)

    report["n_parameter_sets_tested"] = n_tests
    for name, r in report["strategies"].items():
        if "test_bootstrap" in r:
            r["verdict"] = verdict(r["test_bootstrap"], n_tests, V["min_trades"])
    # reproducibility metadata
    feat = next(iter(sorted((EXP_ROOT / "data" / "storage").glob("features_*.parquet"))), None)
    try:
        commit = subprocess.run(["git", "-C", str(MAIN_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = None
    report["metadata"] = {"generated_at": datetime.now().isoformat(), "runtime_s": round(_time.time() - t0, 1),
                          "bars": len(df), "sessions": len(sessions), "train_sessions": n_tr, "test_sessions": n_te,
                          "train_end": tr_end, "features_sha256": hashlib.sha256(feat.read_bytes()).hexdigest() if feat else None,
                          "main_repo_commit": commit, "python": platform.python_version(), "pandas": pd.__version__,
                          "numpy": np.__version__, "data_mode": S["data"]["mode"]}
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump({"settings": S, "strategy_params": P}, sort_keys=False))
    (run_dir / "summary.json").write_text(json.dumps(report, indent=1, default=str))
    from orderflow_scalping.analytics.consolidated import write_markdown

    write_markdown(run_dir, report, S)
    print(f"\nreport -> {run_dir / 'REPORT.md'}  ({report['metadata']['runtime_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
