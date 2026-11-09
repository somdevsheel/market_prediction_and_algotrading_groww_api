"""Daily job scheduler (simple, dependency-free loop; IST; trading days only).

    python apps/worker/scheduler.py [--mode paper]

08:50 pre-market check (token refresh via TOTP if configured, funds, features present)
09:05 start the trading worker (subprocess, paper by default)
15:45 end-of-day: incremental ingest (index, VIX, futures), today's ATM IV, rebuild gold features (next
      session row), backup with verification
Each job failure raises an alert. Jobs run at most once per day (state in the ledger).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402
import logging  # noqa: E402
import subprocess  # noqa: E402
import time as _t  # noqa: E402
from datetime import datetime, time, timedelta  # noqa: E402

from core.clock import IST, MarketCalendar  # noqa: E402
from core.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402
from core.log import setup_logging  # noqa: E402
from ledger.store import Ledger  # noqa: E402
from monitoring.alerts import AlertManager  # noqa: E402

log = logging.getLogger("scheduler")
PY = sys.executable


def run(cmd: list[str], timeout: int = 7200) -> tuple[int, str]:
    p = subprocess.run([PY, *cmd], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)[-3000:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="paper", choices=["paper", "live"])
    a = ap.parse_args()
    cfg = load_config()
    setup_logging(resolve_path(cfg, "logs"))
    cal = MarketCalendar(cfg["settings"], cfg["instruments"])
    led = Ledger(resolve_path(cfg, "ledger_db"))
    alerts = AlertManager(led)
    worker = None

    def once(name: str, at: time, fn):
        now = datetime.now(IST)
        key = f"job:{name}:{now.date()}"
        if now.time() >= at and key not in led.get_status():
            led.set_status(key, {"started": now.isoformat()})
            try:
                rc, out = fn()
                led.set_status(key, {"started": now.isoformat(), "rc": rc, "tail": out[-500:]})
                if rc != 0:
                    alerts.alert("ERROR", f"JOB_FAILED_{name}", out[-800:])
            except Exception as e:  # noqa: BLE001
                alerts.alert("ERROR", f"JOB_FAILED_{name}", repr(e))

    def start_worker():
        nonlocal worker
        if worker is None or worker.poll() is not None:
            worker = subprocess.Popen([PY, "scripts/run_trader.py", "--mode", a.mode], cwd=PROJECT_ROOT)
        return 0, f"worker pid {worker.pid}"

    def eod():
        today = datetime.now(IST).date()
        s = (today - timedelta(days=7)).isoformat()
        steps = [["scripts/bulk_ingest.py", "--start", s, "--end", today.isoformat(), "--intervals", "5minute,1day"],
                 ["scripts/build_gold.py", "iv", "--start", s, "--end", today.isoformat()],
                 ["scripts/build_gold.py", "features"],
                 ["scripts/backup.py"]]
        outs = []
        for st in steps:
            rc, out = run(st)
            outs.append(f"$ {' '.join(st)} -> {rc}")
            if rc != 0:
                return rc, "\n".join(outs) + "\n" + out
        return 0, "\n".join(outs)

    log.info("scheduler started (worker mode %s)", a.mode)
    while True:
        now = datetime.now(IST)
        if cal.is_trading_day(now.date()):
            once("premarket", time(8, 50), lambda: run(["scripts/check_connection.py"], 300))
            once("start_worker", time(9, 5), start_worker)
            once("eod", time(15, 45), eod)
        _t.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
