"""Change a strategy's governance status with evidence (append-only history in state/governance.json).

    python scripts/promote_strategy.py --strategy breakout_vwap_v1 --to BACKTESTED --backtest-run <run_id>
    python scripts/promote_strategy.py --strategy X --to OUT_OF_SAMPLE_VALIDATED --research-report <report.json> --experiment B_breakout_vwap
    python scripts/promote_strategy.py --strategy X --to PAPER_TRADING
    python scripts/promote_strategy.py --strategy X --to APPROVED_FOR_LIVE      # uses paper trades in the ledger
    python scripts/promote_strategy.py --list
"""
import _bootstrap  # noqa: F401

import argparse
import json

from core.config import load_config, resolve_path
from governance.registry import registry_table, transition
from ledger.store import Ledger


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy")
    ap.add_argument("--to")
    ap.add_argument("--backtest-run")
    ap.add_argument("--research-report")
    ap.add_argument("--experiment")
    ap.add_argument("--note", default="")
    ap.add_argument("--force-reason", help="owner override of unmet evidence requirements (recorded permanently)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    if a.list:
        for r in registry_table(cfg):
            print(f"{r['strategy_id']:22s} v{r['version']:6s} {r['status']:24s} hash={r['config_hash']} "
                  f"{'(active)' if r['active'] else ''}")
        return 0
    ev: dict = {}
    if a.backtest_run:
        ev["backtest_run_id"] = a.backtest_run
    if a.research_report:
        rep = json.loads(open(a.research_report).read())
        rows = [e for e in rep["experiments"] if e["experiment"] == a.experiment]
        cons = [e for e in rows if e["scenario"] == "conservative"]
        ev["oos"] = {"trades": max((e.get("oos_trades") or 0) for e in rows) if rows else 0,
                     "net_pnl_conservative": cons[0].get("oos_net") if cons else None,
                     "report": a.research_report, "experiment": a.experiment}
    if a.to == "APPROVED_FOR_LIVE":
        led = Ledger(resolve_path(cfg, "ledger_db"))
        pts = [t for t in led.trades("paper") if t["strategy_id"] == a.strategy and t["status"] == "CLOSED"]
        ev["paper"] = {"trades": len(pts), "net_pnl": sum(t["net_pnl"] or 0 for t in pts)}
    try:
        if a.force_reason:
            phrase = f"OVERRIDE {a.strategy}"
            if input(f"Overriding governance evidence for {a.strategy} -> {a.to}.\nType '{phrase}' to confirm: ").strip() != phrase:
                print("not confirmed")
                return 1
        e = transition(cfg, a.strategy, a.to, ev, a.note, force_reason=a.force_reason)
    except ValueError as err:
        print("REFUSED:", err)
        return 1
    print("recorded:", json.dumps(e, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
