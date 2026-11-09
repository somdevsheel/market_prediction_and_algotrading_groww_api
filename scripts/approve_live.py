"""Interactive, same-day live-trading approval. Writes state/live_approval.json valid for TODAY only.

Live trading also requires settings.mode=live and the strategy status APPROVED_FOR_LIVE.
"""
import _bootstrap  # noqa: F401

import getpass
import json
from datetime import datetime

from core.clock import IST
from core.config import load_config, resolve_path

GATES = [
    "Backtest reviewed incl. out-of-sample and conservative-cost scenario",
    "Paper trading run for a meaningful period and compared with backtest",
    "All tests pass (python -m pytest)",
    "Broker integration verified (scripts/check_connection.py) and funds sufficient",
    "Static IP / Groww API requirements satisfied",
    "Current SEBI/NSE algo-trading rules reviewed",
    "I accept that I can lose the full premium and more than the planned stop",
]

cfg = load_config()
print("LIVE TRADING APPROVAL — answer each gate with 'yes'\n")
for g in GATES:
    if input(f"  {g}? ").strip().lower() != "yes":
        print("Not approved.")
        raise SystemExit(1)
phrase = f"APPROVE LIVE {datetime.now(IST):%Y-%m-%d}"
if input(f"\nType exactly: {phrase}\n> ").strip() != phrase:
    print("Not approved.")
    raise SystemExit(1)
p = resolve_path(cfg, "live_approval_file")
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({"date": f"{datetime.now(IST):%Y-%m-%d}", "approved_by": getpass.getuser(),
                         "at": datetime.now(IST).isoformat(), "gates": GATES}))
print(f"Approved for today only -> {p}")
