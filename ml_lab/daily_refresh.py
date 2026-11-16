"""One-click daily refresh for the Stocks ML page: new prices -> retrain the main model -> new signals -> new forecasts.

Steps (each a separate process, logged to ml_lab/results/refresh.log, progress in ml_lab/results/refresh_status.json):
  1. index data (NIFTY etc.)                python3 -m ml_lab.data.yahoo
  2. stock prices, last few weeks           python3 -m ml_lab.data.all_stocks update
  3. retrain the main model (XGBoost) on    python3 -m ml_lab.stocks_bhs --current-only
     all data up to the latest close and write today's BUY/HOLD/SELL (snapshot kept in <run>/signals_history/<date>.csv)
     (CPU only, ~1-2 min; all 13 models: add --all-models, slow on CPU)
  4. price-range forecasts                  python3 -m ml_lab.stock_price_forecast

The yearly walk-forward backtest (python3 -m ml_lab.stocks_bhs, ~40+ min) is NOT part of this; it only changes
when you want to re-test the models, not for daily signals.

    python3 -m ml_lab.daily_refresh
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from core.clock import IST

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
STATUS = RESULTS / "refresh_status.json"
LOG = RESULTS / "refresh.log"
STEPS = [
    ("Index data (NIFTY, VIX, ...)", ["-m", "ml_lab.data.yahoo"]),
    ("Stock prices (all ~5,200 stocks)", ["-m", "ml_lab.data.all_stocks", "update"]),
    ("Retrain XGBoost (main model) + new signals", ["-m", "ml_lab.stocks_bhs", "--current-only"]),
    ("Price-range forecasts", ["-m", "ml_lab.stock_price_forecast"]),
]


def now() -> str:
    return datetime.now(IST).isoformat(timespec="seconds")


def read_status() -> dict:
    try:
        st = json.loads(STATUS.read_text())
    except (OSError, ValueError):
        return {"state": "idle"}
    if st.get("state") == "running" and not _alive(st.get("pid")):
        st.update(state="failed", error="the refresh process stopped unexpectedly", finished=st.get("finished") or now())
    return st


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (TypeError, ValueError, OSError):
        return False


def write_status(**kw) -> None:
    st = read_status() if STATUS.exists() else {}
    st.update(kw)
    RESULTS.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(STATUS)


def main() -> int:
    cwd = ROOT.parent
    steps = [{"name": n, "state": "pending"} for n, _ in STEPS]
    write_status(state="running", pid=os.getpid(), started=now(), finished=None, error=None, steps=steps, step=0)
    with LOG.open("w") as log:
        for i, (name, args) in enumerate(STEPS):
            steps[i].update(state="running", started=now())
            write_status(steps=steps, step=i + 1)
            log.write(f"\n===== {now()}  step {i + 1}/{len(STEPS)}: {name}\n")
            log.flush()
            rc = subprocess.call([sys.executable, *args], cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
            steps[i].update(state="done" if rc == 0 else "failed", finished=now(), exit_code=rc)
            if rc != 0:
                write_status(state="failed", steps=steps, finished=now(), error=f"step {i + 1} ({name}) exited with code {rc}")
                return rc
            write_status(steps=steps)
    write_status(state="done", steps=steps, finished=now())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
