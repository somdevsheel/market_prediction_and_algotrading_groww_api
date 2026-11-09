"""Run backtests on real Groww historical data (downloads into the data lake, cached & idempotent).

    python scripts/run_backtest.py --start 2026-08-01 --end 2026-11-09
    python scripts/run_backtest.py --start 2026-05-01 --end 2026-11-09 --experiments all --scenarios all
"""
import _bootstrap  # noqa: F401

import argparse
import json
from datetime import date

from backtesting.engine.backtester import Backtester
from backtesting.validation.experiments import EXPERIMENTS
from core.config import load_config, resolve_path
from core.log import setup_logging
from ingestion.historical.datalake import DataLake
from ingestion.providers.groww import GrowwProvider

KEYS = ["trades", "net_pnl", "roi_pct", "win_rate_pct", "profit_factor", "expectancy", "max_drawdown", "sharpe",
        "charges"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--experiments", default="default", help="'default', 'all' or comma list of " + ",".join(EXPERIMENTS))
    ap.add_argument("--scenarios", default="base", help="'all' or comma list of optimistic,base,conservative")
    ap.add_argument("--capital", type=float, help="research override of risk.capital/max_premium_outlay "
                    "(measures per-trade edge without the small-account funding constraint)")
    a = ap.parse_args()
    cfg = load_config()
    if a.capital:
        cfg["risk"]["capital"] = cfg["risk"]["max_premium_outlay"] = a.capital
    setup_logging(resolve_path(cfg, "logs"))
    provider = GrowwProvider(cfg["settings"])
    lake = DataLake(resolve_path(cfg, "data_lake"))
    exps = {"default": {}} if a.experiments == "default" else (
        EXPERIMENTS if a.experiments == "all" else {k: EXPERIMENTS[k] for k in a.experiments.split(",")})
    scns = list(cfg["settings"]["backtest"]["scenarios"]) if a.scenarios == "all" else a.scenarios.split(",")
    rows = []
    for en, ov in exps.items():
        for sc in scns:
            bt = Backtester(cfg, provider, lake, sc, ov, en)
            res = bt.run(date.fromisoformat(a.start), date.fromisoformat(a.end))
            out = res.save(resolve_path(cfg, "backtests"))
            m = res.metrics
            oos = (res.robustness or {}).get("out_of_sample", {})
            rows.append({"experiment": en, "scenario": sc, **{k: m.get(k) for k in KEYS},
                         "oos_trades": oos.get("trades"), "oos_net": oos.get("net_pnl"), "dir": out.name})
            print(f"\n== {en} / {sc} -> {out}")
            print(json.dumps({k: m.get(k) for k in KEYS}, indent=1))
            print(f"skipped signals: {len(res.skipped)}  data issues: {len(res.data_issues)}")
    print("\nSUMMARY (simulated fills; not actual historical fills)")
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
