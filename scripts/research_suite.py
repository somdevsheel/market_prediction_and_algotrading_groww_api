"""Full research suite (experiments A–I, cost scenarios, ML walk-forward, parameter walk-forward,
sensitivity, data-quality sensitivity). Writes a report to data/backtests/research_<ts>/.

    python scripts/research_suite.py --start 2021-01-01 --end 2026-11-06 [--workers 8]

Phase 1 (parallel, offline): causal signal replays of base strategies A, B, C and the B parameter grid.
Phase 2 (sequential, rate-limited downloads of option candles): trade simulation for every experiment/scenario.
Experiments D, E, H, I = B's signals passed through their configured filters (exact: filters act post-signal).
"""
import _bootstrap  # noqa: F401

import argparse
import itertools
import json
import logging
import pickle
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from types import SimpleNamespace

import pandas as pd

from backtesting.engine.backtester import Backtester
from backtesting.reports.metrics import compute_metrics
from backtesting.validation.walk_forward import cost_sensitivity, sensitivity_table, walk_forward
from core.clock import IST
from core.config import load_config, resolve_path
from core.log import setup_logging
from ingestion.historical.datalake import DataLake
from ingestion.instruments.calendar import ExpiryCalendar
from ingestion.providers.groww import GrowwProvider
from quant.machine_learning import pipeline as ml
from strategies.filters import FilteredStrategy

log = logging.getLogger("research")
GRID = {"breakout_buffer_points": [3, 5, 10], "volume_multiplier": [1.0, 1.2, 1.5]}
RESEARCH_CAPITAL = 1_000_000.0


def _replay(job):
    sid, overrides, start, end, preload, feats = job
    cfg = load_config()
    lake = DataLake(resolve_path(cfg, "data_lake"))
    bt = Backtester(cfg, None, lake, "base", overrides, strategy_id=sid, features=feats,
                    calendar_preload=preload, offline=True)
    sigs, days = bt.signals(start, end)
    return sid, overrides, sigs, days


def combo_key(o: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(o.items())) or "default"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-11-06")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    cfg = load_config()
    setup_logging(resolve_path(cfg, "logs"))
    lake = DataLake(resolve_path(cfg, "data_lake"))
    provider = GrowwProvider(cfg["settings"])
    out_dir = resolve_path(cfg, "backtests") / f"research_{datetime.now(IST):%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = lake.root / "gold" / "volatility" / "daily_features.parquet"
    feats = pd.read_parquet(fpath) if fpath.exists() else None

    # prewarm expiry calendar (API) so workers run offline
    cal = ExpiryCalendar(provider, "NIFTY", lake.root / "manifests" / "expiries_cache.json")
    y, m = start.year, start.month
    while (y, m) <= (end.year + (end.month == 12), (end.month % 12) + 1):
        cal.month(y, m)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    preload = dict(cal._c)

    # ---------------- phase 1: parallel causal replays
    jobs = [("sr_breakout_v1", {}), ("breakout_vwap_v1", {}), ("breakout_retest_v1", {})]
    for combo in itertools.product(*GRID.values()):
        o = dict(zip(GRID.keys(), combo))
        if o != {"breakout_buffer_points": 5, "volume_multiplier": 1.2}:
            jobs.append(("breakout_vwap_v1", o))
    log.info("phase 1: %d replays on %d workers", len(jobs), a.workers)
    replays = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for sid, o, sigs, days in ex.map(_replay, [(s, o, start, end, preload, feats) for s, o in jobs]):
            replays[(sid, combo_key(o))] = (sigs, days)
            log.info("replay %s %s: %d candidate signals", sid, combo_key(o), len(sigs))
    with open(out_dir / "signals.pkl", "wb") as f:
        pickle.dump(replays, f)

    B_sigs, days = replays[("breakout_vwap_v1", "default")]

    # ---------------- ML walk-forward on B's candidate signals
    sq = datetime.strptime(cfg["settings"]["session"]["square_off"], "%H:%M").time()
    rows = []
    for d, dec, sig, f, day_idx in B_sigs:
        r = ml.sample_row(d, dec, sig, f)
        r["label"] = ml.label_signal(sig, day_idx, dec, sq)
        rows.append(r)
    ds = pd.DataFrame(rows)
    P, M = ml.walk_forward(ds) if len(ds) else (pd.DataFrame(), pd.DataFrame())
    lake.write_gold("ml", "dataset_breakout_vwap_v1", ds)
    if len(P):
        lake.write_gold("ml", "oos_predictions_breakout_vwap_v1", P)
        lake.write_gold("ml", "metrics_breakout_vwap_v1", M)
    model_choice = (cfg["strategies"]["strategies"]["ml_confirmed_v1"].get("ml_model") or "logistic")
    if len(ds) >= 100 and ds["label"].nunique() == 2:
        mp = resolve_path(cfg, "ledger_db").parent / "models" / f"ml_breakout_vwap_v1_{model_choice}.pkl"
        ml.fit_final(ds, model_choice, mp, {"base_strategy": "breakout_vwap_v1", "model": model_choice,
                                            "samples": len(ds), "train_start": str(start), "train_end": str(end),
                                            "trained_at": datetime.now(IST).isoformat(),
                                            "oos_metrics": M.to_dict("records") if len(M) else []})
    ml_probs = dict(zip(P[P.model == model_choice].decision_ts, P[P.model == model_choice].prob)) if len(P) else {}
    log.info("ML: %d samples, label rate %.3f\n%s", len(ds), ds["label"].mean() if len(ds) else 0, M)

    # ---------------- filtered experiments from B's signals
    def filtered(sid: str):
        scfg = cfg["strategies"]["strategies"][sid]
        fs = FilteredStrategy(SimpleNamespace(params={}, last_reasons=[]), scfg["filters"], sid, scfg["version"])
        out = []
        for d, dec, sig, f, di in B_sigs:
            f2 = dict(f)
            f2["ml_prob"] = ml_probs.get(dec.isoformat())
            ok, R, codes = fs.check(SimpleNamespace(features=f2, regime={"regime": sig.regime}), sig.direction)
            if ok:
                s2 = pickle.loads(pickle.dumps(sig))
                s2.strategy_id, s2.reason_codes = sid, sig.reason_codes + codes
                out.append((d, dec, s2, f2, di))
        return out, days

    experiments = {"A_sr_breakout": ("sr_breakout_v1", replays[("sr_breakout_v1", "default")]),
                   "B_breakout_vwap": ("breakout_vwap_v1", (B_sigs, days)),
                   "C_breakout_retest": ("breakout_retest_v1", replays[("breakout_retest_v1", "default")])}
    for name, sid in (("D_vol_filtered", "vol_filtered_v1"), ("E_regime_filtered", "regime_filtered_v1"),
                      ("H_ml_confirmed", "ml_confirmed_v1"), ("I_combined", "combined_v1")):
        experiments[name] = (sid, filtered(sid))

    # ---------------- phase 2: simulations
    report = {"period": [str(start), str(end)], "generated_at": datetime.now(IST).isoformat(),
              "ml_metrics": M.to_dict("records") if len(M) else [], "ml_samples": len(ds),
              "experiments": [], "research_capital_experiments": []}
    trades_store = {}
    for name, (sid, pre) in experiments.items():
        for scn in cfg["settings"]["backtest"]["scenarios"]:
            bt = Backtester(cfg, provider, lake, scn, strategy_id=sid, experiment=name, features=feats)
            res = bt.run(start, end, precomputed=pre)
            res.save(resolve_path(cfg, "backtests"))
            mm, rob = res.metrics, res.robustness or {}
            report["experiments"].append({"experiment": name, "scenario": scn, "run_id": res.run_id,
                                          **{k: mm.get(k) for k in ("trades", "net_pnl", "roi_pct", "win_rate_pct",
                                                                    "profit_factor", "expectancy", "max_drawdown",
                                                                    "sharpe")},
                                          "oos_net": (rob.get("out_of_sample") or {}).get("net_pnl"),
                                          "oos_trades": (rob.get("out_of_sample") or {}).get("trades"),
                                          "skipped": len(res.skipped)})
            log.info("%s/%s: %s", name, scn, {k: mm.get(k) for k in ("trades", "net_pnl", "profit_factor")})
        cfg_r = json.loads(json.dumps(cfg))
        cfg_r["risk"]["capital"] = cfg_r["risk"]["max_premium_outlay"] = RESEARCH_CAPITAL
        bt = Backtester(cfg_r, provider, lake, "base", strategy_id=sid, experiment=name + "_rc", features=feats)
        res = bt.run(start, end, precomputed=pre)
        trades_store[name] = res.trades
        mm = res.metrics
        report["research_capital_experiments"].append({"experiment": name, **{k: mm.get(k) for k in (
            "trades", "net_pnl", "win_rate_pct", "profit_factor", "expectancy", "by_year", "by_exit_reason",
            "by_hmm", "by_days_to_expiry", "by_option_type")}, "bootstrap": (res.robustness or {}).get("bootstrap")})

    # ---------------- parameter grid: sensitivity + walk-forward (research capital, base scenario)
    cfg_r = json.loads(json.dumps(cfg))
    cfg_r["risk"]["capital"] = cfg_r["risk"]["max_premium_outlay"] = RESEARCH_CAPITAL
    grid_trades = {}
    for (sid, key), pre in replays.items():
        if sid != "breakout_vwap_v1":
            continue
        bt = Backtester(cfg_r, provider, lake, "base", strategy_id=sid, experiment=f"grid[{key}]", features=feats)
        grid_trades[key] = bt.run(start, end, precomputed=pre).trades
    for rr in (1.5, 3.0):  # exit-parameter sensitivity on default signals (simulation only)
        c2 = json.loads(json.dumps(cfg_r))
        c2["risk"]["reward_risk"] = rr
        bt = Backtester(c2, provider, lake, "base", strategy_id="breakout_vwap_v1", experiment=f"rr{rr}", features=feats)
        grid_trades[f"default,reward_risk={rr}"] = bt.run(start, end, precomputed=(B_sigs, days)).trades
    report["parameter_sensitivity"] = sensitivity_table(grid_trades, RESEARCH_CAPITAL)
    report["walk_forward"] = walk_forward(grid_trades, days, 12, 3, RESEARCH_CAPITAL)
    report["cost_sensitivity_B"] = cost_sensitivity(trades_store["B_breakout_vwap"], RESEARCH_CAPITAL) \
        if len(trades_store["B_breakout_vwap"]) else []
    dq = []
    for frac in (0.05, 0.10):
        bt = Backtester(cfg_r, provider, lake, "base", strategy_id="breakout_vwap_v1", experiment=f"dq{frac}",
                        features=feats, drop_option_candle_frac=frac, seed=1)
        mm = bt.run(start, end, precomputed=(B_sigs, days)).metrics
        dq.append({"drop_option_candles": frac, **{k: mm.get(k) for k in ("trades", "net_pnl", "expectancy")}})
    report["data_quality_sensitivity_B"] = dq
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    lake.write_gold("backtest_results", f"research_{out_dir.name}", pd.DataFrame(report["experiments"]))
    print(json.dumps({k: report[k] for k in ("experiments", "ml_metrics", "walk_forward")}, indent=1, default=str)[:6000])
    print("report ->", out_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
