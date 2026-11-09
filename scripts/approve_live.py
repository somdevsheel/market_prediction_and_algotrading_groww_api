"""Owner approval for LIVE trading. Writes state/live_approval.json.

    python3 scripts/approve_live.py              # today only
    python3 scripts/approve_live.py --days 10    # standing approval for the next 10 trading days (max 20)
    python3 scripts/approve_live.py --revoke     # remove approval immediately
    python3 scripts/approve_live.py --status

A standing approval lets the scheduler start the bot live every morning without typing anything. It is tied
to the strategy's config hash: any change to the strategy voids it. Live trading also requires
settings.mode=live and the strategy status APPROVED_FOR_LIVE.
"""
import _bootstrap  # noqa: F401

import argparse
import getpass
import json
from datetime import datetime

from core.clock import IST, MarketCalendar
from core.config import load_config, resolve_path
from governance.registry import config_hash

GATES = [
    "Backtest reviewed incl. the 2024+ weakness at small risk budgets (runbook section 2.6)",
    "Live smoke test passed (scripts/live_smoke_test.py printed PASS)",
    "Funds in Groww cover one lot plus the configured risk",
    "TOTP auto-login configured (needed for unattended mornings) or I will refresh the token daily",
    "Static IP / Groww API requirements satisfied; current SEBI/NSE algo rules reviewed",
    "I accept that I can lose the full premium and more than the planned stop, and that bugs or outages can cause losses",
    "I will monitor alerts and know how to use the kill switch and close positions in the Groww app",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1, help="trading days covered, 1-20")
    ap.add_argument("--revoke", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    p = resolve_path(cfg, "live_approval_file")
    if a.status:
        print(p.read_text() if p.exists() else "no live approval")
        return 0
    if a.revoke:
        if p.exists():
            p.unlink()
        print("live approval revoked")
        return 0
    days = max(1, min(20, a.days))
    cal = MarketCalendar(cfg["settings"], cfg["instruments"])
    today = datetime.now(IST).date()
    first = today if cal.is_trading_day(today) else cal.next_trading_day(today)
    until = first
    for _ in range(days - 1):
        until = cal.next_trading_day(until)
    sid = cfg["strategies"]["active_strategy"]
    print(f"LIVE TRADING APPROVAL for {sid} — {first} .. {until} ({days} trading day(s)). Answer each with 'yes'\n")
    for g in GATES:
        if input(f"  {g}? ").strip().lower() != "yes":
            print("Not approved.")
            return 1
    phrase = f"APPROVE LIVE {first} TO {until}"
    if input(f"\nType exactly: {phrase}\n> ").strip() != phrase:
        print("Not approved.")
        return 1
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"date": first.isoformat(), "valid_until": until.isoformat(), "strategy": sid,
                             "config_hash": config_hash(cfg["strategies"]["strategies"][sid]),
                             "approved_by": getpass.getuser(), "at": datetime.now(IST).isoformat(), "gates": GATES}, indent=1))
    print(f"Approved {first} .. {until} -> {p}\nRevoke any time: python3 scripts/approve_live.py --revoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
