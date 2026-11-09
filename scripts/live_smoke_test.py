"""Zero-cost LIVE broker verification (run during market hours, before enabling live trading).

Places ONE limit BUY order for 1 lot of the nearest non-expiry ATM NIFTY CE at ~30% of its bid (far from the
market, so it should not fill), then checks status by reference, cancels it, and confirms the cancellation.
This exercises: place_order, order_reference_id lookup, status mapping, cancel, positions and funds — with
real Groww responses — without taking a position. Everything is recorded in the ledger (role TEST, mode live)
and in state/live_smoke_test.json.

    python scripts/live_smoke_test.py

Requires typing a confirmation phrase. If the order unexpectedly fills, it immediately offers to sell it back.
"""
import _bootstrap  # noqa: F401

import json
import time as _t
from datetime import datetime

from core.clock import IST, MarketCalendar
from core.config import load_config, resolve_path
from core.log import setup_logging
from execution.broker_interface import OrderRequest
from execution.contract_selector import choose_expiry
from execution.groww_adapter import GrowwBroker
from execution.order_manager import OrderManager, make_order_ref
from execution.order_state import OrderState as S
from ingestion.instruments.calendar import ExpiryCalendar
from ingestion.providers.groww import GrowwProvider
from ledger.store import Ledger


def main() -> int:
    cfg = load_config()
    setup_logging(resolve_path(cfg, "logs"))
    cal = MarketCalendar(cfg["settings"], cfg["instruments"])
    now = datetime.now(IST)
    if not cal.in_session(now):
        print(f"Market is closed ({now:%a %H:%M} IST). Run between 09:20 and 15:00 on a trading day.")
        return 1
    p = GrowwProvider(cfg["settings"])
    broker = GrowwBroker(p.client, cfg["settings"])
    led = Ledger(resolve_path(cfg, "ledger_db"))
    om = OrderManager(broker, led, cfg["settings"])
    report: dict = {"started": now.isoformat(), "steps": []}

    def step(name, **kw):
        report["steps"].append({"step": name, **kw})
        print(f"- {name}: {json.dumps(kw, default=str)[:400]}")

    funds = broker.get_available_funds()
    step("funds", option_buy_balance=funds)
    pos_before = broker.get_net_positions()
    step("positions_before", positions=pos_before)
    exp = choose_expiry(ExpiryCalendar(p, "NIFTY").around(now.date()), now.date(), True)
    chain = p.get_option_chain("NIFTY", exp)
    k = round(chain.underlying_ltp / 50) * 50
    row = chain.get(k, "CE")
    inst = p.get_instruments()
    lot = int(inst[(inst.exchange == "NSE") & (inst.trading_symbol == row.trading_symbol)].iloc[0]["lot_size"])
    q = p.get_quote(row.trading_symbol, "FNO")
    tick = 0.05
    price = max(tick, round(round((q.bid or q.ltp) * 0.30 / tick) * tick, 2))
    step("contract", symbol=row.trading_symbol, expiry=str(exp), lot=lot, bid=q.bid, ask=q.ask, test_limit=price,
         value=round(price * lot, 2))
    if price * lot > funds:
        step("abort", reason="test order value exceeds available funds")
        return 1
    phrase = "PLACE TEST ORDER"
    if input(f"\nThis sends a REAL limit BUY: {lot} x {row.trading_symbol} @ {price} (far below market; should not "
             f"fill) and then cancels it.\nType '{phrase}' to continue: ").strip() != phrase:
        print("cancelled by user")
        return 1
    tid = f"SMOKE-{now:%Y%m%d%H%M%S}"
    req = OrderRequest(order_ref=make_order_ref(tid, "TEST"), trading_symbol=row.trading_symbol, side="BUY", qty=lot,
                       order_type="LIMIT", price=price, product=cfg["settings"]["execution"]["product"],
                       validity=cfg["settings"]["execution"]["validity"])
    o = om.submit(req, trade_id=tid, signal_id=None, role="TEST", mode="live", trading_day=now.date())
    step("placed", state=o["state"], broker_order_id=o["broker_order_id"], last_error=o["last_error"])
    _t.sleep(3)
    o = om.refresh(o["order_id"])
    raw = led.query("SELECT detail FROM order_events WHERE order_id=? ORDER BY id DESC LIMIT 1", (o["order_id"],))
    step("status_by_reference", state=o["state"], filled_qty=o["filled_qty"], last_event=raw[0]["detail"] if raw else None)
    if S(o["state"]) in (S.REJECTED,):
        step("result", verdict="ORDER REJECTED by broker/exchange — read last_error; adjust product/price rules before live")
    elif int(o["filled_qty"] or 0) > 0:
        step("WARNING", msg="test order FILLED unexpectedly — you now hold a position")
        if input("Type 'SELL BACK' to sell it with a marketable limit now: ").strip() == "SELL BACK":
            q2 = p.get_quote(row.trading_symbol, "FNO")
            sreq = OrderRequest(order_ref=make_order_ref(tid, "EXIT"), trading_symbol=row.trading_symbol, side="SELL",
                                qty=int(o["filled_qty"]), order_type="LIMIT",
                                price=max(tick, round(round(((q2.bid or q2.ltp) - 4 * tick) / tick) * tick, 2)),
                                product=req.product, validity=req.validity)
            so = om.submit(sreq, trade_id=tid, signal_id=None, role="EXIT", mode="live", trading_day=now.date())
            so = om.wait_terminal(so["order_id"], 20)
            step("sell_back", state=so["state"], filled=so["filled_qty"], avg=so["avg_price"])
    else:
        o = om.cancel(o["order_id"])
        step("cancel", state=o["state"])
        ok = S(o["state"]) == S.CANCELLED
        step("result", verdict="PASS: place -> status -> cancel works with real Groww responses" if ok else
             f"CHECK: final state {o['state']} — inspect order in the Groww app before going live")
    step("positions_after", positions=broker.get_net_positions())
    out = resolve_path(cfg, "ledger_db").parent / "live_smoke_test.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    print("\nreport ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
