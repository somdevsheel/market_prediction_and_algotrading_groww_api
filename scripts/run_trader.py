"""Run the intraday trading worker.

    python scripts/run_trader.py              # paper (default, safe)
    python scripts/run_trader.py --mode live  # requires: settings.mode=live, strategy APPROVED_FOR_LIVE,
                                              # and today's approval from scripts/approve_live.py

Start any time before 09:15 IST; it waits for the session, trades at most one
trade, squares off by 15:15 and exits. Ctrl+C triggers a safe shutdown that
exits any open position first.
"""
import _bootstrap  # noqa: F401

import argparse
import logging
import signal
import time as _time
from datetime import datetime

from apps.worker.engine import TradingEngine
from core.clock import IST, Clock, MarketCalendar
from core.config import load_config, resolve_path
from core.log import setup_logging
from core.secrets import obtain_access_token
from execution.paper_adapter import PaperBroker
from ingestion.providers.groww import GrowwProvider
from ledger.store import Ledger
from risk.kill_switch import KillSwitch

log = logging.getLogger("run_trader")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    args = ap.parse_args()
    cfg = load_config()
    setup_logging(resolve_path(cfg, "logs"))
    clock = Clock()
    cal = MarketCalendar(cfg["settings"], cfg["instruments"])
    today = clock.today()
    if not cal.is_trading_day(today):
        log.warning("%s is not a trading day per config/instruments.yaml holidays. Next: %s", today,
                    cal.next_trading_day(today))
        return 0
    token = obtain_access_token()
    provider = GrowwProvider(cfg["settings"], token=token)
    ledger = Ledger(resolve_path(cfg, "ledger_db"))
    kill = KillSwitch(resolve_path(cfg, "kill_switch_file"))
    if args.mode == "live":
        from execution.groww_adapter import GrowwBroker

        broker = GrowwBroker(provider.client, cfg["settings"])
    else:
        broker = PaperBroker(lambda s: provider.get_quote(s, "FNO"), cfg["settings"]["paper"]["starting_capital"],
                             cfg["settings"]["paper"]["slippage_ticks"], cfg["instruments"]["tick_size"])
    engine = TradingEngine(cfg, provider, broker, ledger, clock, kill)
    engine.startup()
    if kill.active():
        log.warning("KILL SWITCH ACTIVE: %s — no new entries. Clear it with: python scripts/kill.py --off", kill.info())
    stop = {"flag": False}

    def handle(sig, frm):
        stop["flag"] = True
        log.warning("shutdown requested")

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)
    poll = cfg["settings"]["session"]["poll_seconds"]
    log.info("worker started in %s mode (strategy %s, status %s)", args.mode.upper(), engine.strategy.strategy_id,
             engine.strategy_status)
    while not stop["flag"]:
        try:
            engine.step()
        except Exception as e:  # noqa: BLE001
            log.exception("step failed")
            ledger.record_error("worker", repr(e))
        if engine.done_for_day and not engine.trade:
            log.info("done for the day")
            break
        _time.sleep(poll)
    if engine.trade:
        log.warning("exiting open position before shutdown")
        engine.exit_position("shutdown", datetime.now(IST))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
