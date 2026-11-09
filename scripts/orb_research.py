"""ORB research: replicate the brief's spot-point study, validate its filters honestly, then test real options.

    python scripts/orb_research.py spot     [--start 2020-01-01 --end 2026-09-30]   (offline, 1-minute lake data)
    python scripts/orb_research.py options  [--start 2020-01-01 --end 2026-11-06]   (downloads 1-minute option candles)

Spot study: 4 opening-range lengths x {breakout, retest, reversal}; the brief's filters; year-by-year; and a
train/test check where filter thresholds are chosen on 2020-2023 only and scored on 2024+.
Options study: the 3 filtered setups with real 1-ITM option candles, costs, and (a) the brief's stop with a
research risk budget, (b) the user's Rs 1,000 risk cap.
"""
import _bootstrap  # noqa: F401

import argparse
import itertools
import json
from datetime import date, datetime, time

import numpy as np
import pandas as pd

from core.clock import IST
from core.config import load_config, resolve_path
from core.log import setup_logging
from ingestion.historical.datalake import DataLake
from strategies.orb.orb_core import find_entry, passes_filters, simulate_spot

MODES = ["breakout", "retest", "reversal"]
LENGTHS = [5, 15, 30, 60]
SETUPS = {  # the brief's filtered setups
    "orb_breakout60_v1": ("breakout", 60, {"min_range_pct": 0.5, "entry_before": "12:00"}),
    "orb_retest30_v1": ("retest", 30, {"max_range_pct": 0.5, "max_gap_minutes": 60}),
    "orb_reversal5_v1": ("reversal", 5, {"entry_before": "12:00"}),
}


def stats(t: pd.DataFrame) -> dict:
    if t is None or t.empty:
        return {"trades": 0}
    p = t["points"]
    w, l = p[p > 0], p[p <= 0]
    eq = p.cumsum()
    return {"trades": len(t), "points": round(float(p.sum()), 1), "win_rate": round(float((p > 0).mean() * 100), 1),
            "profit_factor": round(float(w.sum() / -l.sum()), 3) if l.sum() < 0 else None,
            "avg_win": round(float(w.mean()), 1) if len(w) else 0, "avg_loss": round(float(l.mean()), 1) if len(l) else 0,
            "max_dd_points": round(float((eq - eq.cummax()).min()), 1),
            "avg_stop_distance": round(float(t["stop_distance"].mean()), 1)}


def all_entries(days: dict, mode: str, orm: int) -> pd.DataFrame:
    rows = []
    for d, day in days.items():
        e = find_entry(day, mode, orm)
        if e is not None:
            r = simulate_spot(day, e)
            r["_entry"] = e
            rows.append(r)
    return pd.DataFrame(rows)


def filt(t: pd.DataFrame, f: dict) -> pd.DataFrame:
    return t[t["_entry"].apply(lambda e: passes_filters(e, f))] if len(t) else t


def spot(cfg, lake, start: date, end: date) -> dict:
    df = lake.read_silver("nifty", "NSE-NIFTY", "1minute", datetime.combine(start, time.min, IST),
                          datetime.combine(end, time.max, IST))
    df = df[(df.timestamp.dt.time >= time(9, 15)) & (df.timestamp.dt.time < time(15, 30))]
    days = {d: g.reset_index(drop=True) for d, g in df.groupby(df.timestamp.dt.date) if len(g) >= 300}
    out = {"sessions": len(days), "period": [str(min(days)), str(max(days))], "unfiltered": [], "filtered": {},
           "year_by_year": {}, "train_test": {}}
    cache = {}
    for mode, orm in itertools.product(MODES, LENGTHS):
        t = all_entries(days, mode, orm)
        cache[(mode, orm)] = t
        out["unfiltered"].append({"mode": mode, "or_minutes": orm, **stats(t)})
    for sid, (mode, orm, f) in SETUPS.items():
        t = filt(cache[(mode, orm)], f)
        out["filtered"][sid] = stats(t)
        t = t.assign(year=pd.to_datetime(t["date"]).dt.year)
        out["year_by_year"][sid] = {int(y): stats(g) for y, g in t.groupby("year")}
    # honest check: choose thresholds on 2020-2023, score on 2024+
    grids = {
        "orb_breakout60_v1": [{"min_range_pct": r, "entry_before": tb} for r in (None, 0.3, 0.4, 0.5, 0.6, 0.7)
                              for tb in (None, "11:00", "12:00", "13:00")],
        "orb_retest30_v1": [{"max_range_pct": r, "max_gap_minutes": g} for r in (None, 0.3, 0.4, 0.5, 0.6, 0.8)
                            for g in (None, 30, 60, 90, 120)],
        "orb_reversal5_v1": [{"entry_before": tb} for tb in (None, "10:30", "11:00", "12:00", "13:00", "14:00")],
    }
    for sid, grid in grids.items():
        mode, orm, _ = SETUPS[sid]
        t = cache[(mode, orm)]
        yr = pd.to_datetime(t["date"]).dt.year
        tr, te = t[yr <= 2023], t[yr >= 2024]
        scored = []
        for f in grid:
            a, b = filt(tr, f), filt(te, f)
            if len(a) < 200:  # minimum sample in train (brief used 500 over the full period)
                continue
            scored.append({"filters": f, "train": stats(a), "test": stats(b)})
        best = max(scored, key=lambda x: x["train"]["points"])
        out["train_test"][sid] = {"chosen_on_train": best["filters"], "train": best["train"], "test": best["test"],
                                  "unfiltered_test": stats(te), "brief_filters_test": stats(filt(te, SETUPS[sid][2])),
                                  "grid": scored}
    return out


def options(cfg, lake, start: date, end: date) -> dict:
    from backtesting.engine.backtester import Backtester
    from ingestion.instruments.calendar import ExpiryCalendar
    from ingestion.providers.groww import GrowwProvider
    from strategies.registry import build_strategy

    p = GrowwProvider(cfg["settings"])
    ExpiryCalendar(p, "NIFTY", lake.root / "manifests" / "expiries_cache.json")  # warm cache on demand
    df = lake.read_silver("nifty", "NSE-NIFTY", "1minute", datetime.combine(start, time.min, IST),
                          datetime.combine(end, time.max, IST))
    df = df[(df.timestamp.dt.time >= time(9, 15)) & (df.timestamp.dt.time < time(15, 30))]
    days = {d: g.reset_index(drop=True) for d, g in df.groupby(df.timestamp.dt.date) if len(g) >= 300}
    variants = {
        "brief_stop_research_budget": {"capital": 1_000_000, "max_premium_outlay": 1_000_000, "risk_per_trade_max": 25_000},
        "user_account_1000_cap": {"capital": 20_000, "max_premium_outlay": 20_000, "risk_per_trade_max": 1_000},
    }
    out = {}
    for sid, (mode, orm, f) in SETUPS.items():
        strat = build_strategy(sid, cfg["strategies"]["strategies"][sid])
        sigs = []
        for d, day in days.items():
            e = find_entry(day, mode, orm)
            if e is None or not passes_filters(e, f):
                continue
            from strategies.base import MarketContext

            ctx = MarketContext(now=e.entry_ts.to_pydatetime(), index_5m=day.iloc[:e.entry_idx + 1])
            s = strat.evaluate(ctx)
            if s:
                sigs.append((d, e.entry_ts, s, {}, day[["timestamp", "high", "low", "close"]]))
        out[sid] = {"signals": len(sigs)}
        for vname, rv in variants.items():
            for scn in ("base", "conservative"):
                c = json.loads(json.dumps(cfg))
                c["risk"].update(rv)
                c["settings"]["session"]["square_off"] = "15:25"
                bt = Backtester(c, p, lake, scn, strategy_id=sid, experiment=f"{sid}-{vname}", interval=1)
                res = bt.run(start, end, precomputed=(sigs, sorted(days)))
                res.save(resolve_path(cfg, "backtests"))
                m = res.metrics
                rob = res.robustness or {}
                skipped = pd.Series([s.get("reason", "")[:30] for s in res.skipped]).value_counts().head(4).to_dict()
                out[sid][f"{vname}/{scn}"] = {
                    **{k: m.get(k) for k in ("trades", "net_pnl", "gross_pnl", "charges", "win_rate_pct",
                                             "profit_factor", "expectancy", "max_drawdown", "by_year",
                                             "by_exit_reason")},
                    "oos_net": (rob.get("out_of_sample") or {}).get("net_pnl"),
                    "bootstrap": rob.get("bootstrap"), "skipped": skipped, "run_id": res.run_id}
                print(sid, vname, scn, {k: m.get(k) for k in ("trades", "net_pnl", "profit_factor", "win_rate_pct")},
                      "skipped", skipped, flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["spot", "options"])
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-09-30")
    a = ap.parse_args()
    cfg = load_config()
    setup_logging(resolve_path(cfg, "logs"))
    lake = DataLake(resolve_path(cfg, "data_lake"))
    s, e = date.fromisoformat(a.start), date.fromisoformat(a.end)
    res = spot(cfg, lake, s, e) if a.what == "spot" else options(cfg, lake, s, e)
    od = resolve_path(cfg, "backtests") / f"orb_{a.what}_{datetime.now(IST):%Y%m%d-%H%M%S}"
    od.mkdir(parents=True, exist_ok=True)
    (od / "report.json").write_text(json.dumps(res, indent=1, default=str))
    print("report ->", od / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
