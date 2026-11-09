"""Backtest an ORB strategy from config/strategies.yaml on real 1-minute option candles (fast path).

    python3 scripts/backtest_strategy.py                                   # active strategy, 2021 -> today
    python3 scripts/backtest_strategy.py --start 2024-01-01 --scenarios base,conservative --capital 20000 --risk 1000

Signals come from strategies/orb/orb_core.py (identical to live), contract-level filters (cheap IV) and the
risk engine are applied by the backtester, fills are simulated on real option candles with the scenario's
spread/slippage and the full charges. Results go to data/backtests/<run_id>/ (Dashboard -> Backtests).
"""
import _bootstrap  # noqa: F401

import argparse
import json
from datetime import date, datetime

from backtesting.engine.orb_runner import run_orb_backtest
from core.clock import IST
from core.config import load_config, resolve_path
from core.log import setup_logging
from ingestion.providers.groww import GrowwProvider


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default=datetime.now(IST).date().isoformat())
    ap.add_argument("--scenarios", default="base,conservative")
    ap.add_argument("--capital", type=float)
    ap.add_argument("--risk", type=float, help="max planned loss per trade (Rs)")
    ap.add_argument("--square-off", help="HH:MM override (default: config/settings.yaml)")
    a = ap.parse_args()
    cfg = load_config()
    setup_logging(resolve_path(cfg, "logs"))
    sid = a.strategy or cfg["strategies"]["active_strategy"]
    scfg = cfg["strategies"]["strategies"][sid]
    if scfg.get("class") != "orb":
        raise SystemExit(f"{sid}: this fast backtester supports ORB strategies only")
    if a.capital:
        cfg["risk"]["capital"] = cfg["risk"]["max_premium_outlay"] = a.capital
    if a.risk:
        cfg["risk"]["risk_per_trade_max"] = a.risk
    if a.square_off:
        cfg["settings"]["session"]["square_off"] = a.square_off
    print(f"{sid}: backtesting {a.start} -> {a.end}")
    provider = GrowwProvider(cfg["settings"])
    s, e = date.fromisoformat(a.start), date.fromisoformat(a.end)
    for scn in a.scenarios.split(","):
        res = run_orb_backtest(cfg, provider, s, e, scn, sid)
        out = resolve_path(cfg, "backtests") / res.run_id
        m, r = res.metrics, res.robustness or {}
        print(f"\n[{scn}] -> {out.name}")
        print(json.dumps({k: m.get(k) for k in ("trades", "net_pnl", "gross_pnl", "charges", "win_rate_pct", "profit_factor",
                                                "expectancy", "max_drawdown", "avg_win", "avg_loss")}, indent=1))
        print("by year:", {k: v["net_pnl"] for k, v in (m.get("by_year") or {}).items()})
        print("last 30% (OOS):", (r.get("out_of_sample") or {}).get("net_pnl"), "| bootstrap P(mean>0):",
              (r.get("bootstrap") or {}).get("prob_mean_positive"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
