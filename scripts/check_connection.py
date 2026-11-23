"""Pre-market check: credentials, token expiry, funds, market data, lot size, holiday status.

Usage: python scripts/check_connection.py
"""
import _bootstrap  # noqa: F401

import time
from datetime import datetime

from core.clock import IST, MarketCalendar
from core.config import load_config
from core.secrets import get_secret, jwt_expiry, obtain_access_token
from ingestion.providers.groww import GrowwProvider


def main() -> int:
    cfg = load_config()
    cal = MarketCalendar(cfg["settings"], cfg["instruments"])
    now = datetime.now(IST)
    nxt = now.date() if cal.is_trading_day(now.date()) and now.time() < cal.close else cal.next_trading_day(now.date())
    print(f"Now {now:%Y-%m-%d %H:%M} IST | next trading session: {nxt} ({nxt:%A})")
    try:
        tok = obtain_access_token()
    except Exception as e:
        print(f"[FAIL] credentials: {e}")
        return 1
    exp = jwt_expiry(tok)
    if exp:
        left = (exp - time.time()) / 3600
        print(f"[{'OK' if left > 0 else 'FAIL'}] token expires {datetime.fromtimestamp(exp, IST):%Y-%m-%d %H:%M} IST "
              f"({left:.1f} h left)")
        if datetime.fromtimestamp(exp, IST).date() < nxt or (datetime.fromtimestamp(exp, IST).date() == nxt and
                                                              datetime.fromtimestamp(exp, IST).time() < cal.close):
            if get_secret("GROWW_API_KEY") and (get_secret("GROWW_TOTP_SECRET") or get_secret("GROWW_API_SECRET")):
                print("       -> expires before the next session; a new token is generated automatically (API key + TOTP).")
            else:
                print("       -> token will NOT cover the next session. Refresh it that morning or set up TOTP (.env.example).")
    if not get_secret("GROWW_TOTP_SECRET") and not get_secret("GROWW_API_SECRET"):
        print("[WARN] no TOTP/secret configured; you must paste a fresh access token every morning")
    p = GrowwProvider(cfg["settings"], token=tok)
    ltp = p.get_ltp(["NSE_NIFTY", "NSE_INDIAVIX"], "CASH")
    print(f"[OK] NIFTY {ltp.get('NSE_NIFTY')}  INDIA VIX {ltp.get('NSE_INDIAVIX')}")
    funds = p.get_available_option_buy_balance()
    print(f"[{'OK' if funds > 0 else 'WARN'}] option-buy balance: Rs {funds:,.2f}")
    fut = p.futures_contracts("NIFTY")
    lot = int(fut.iloc[0]["lot_size"])
    spot = ltp.get("NSE_NIFTY") or 0
    print(f"[OK] NIFTY lot size {lot}; near futures {fut.iloc[0]['groww_symbol']}")
    est = 1.2 * 0.01 * spot * lot  # rough 1-ITM weekly premium ~1.2% of spot
    if funds < est:
        print(f"[BLOCK] live: a 1-ITM lot needs roughly Rs {est:,.0f}; live entries will be rejected by the risk engine"
              f" (sufficient_funds). Paper mode is unaffected.")
    print(f"[INFO] mode in config: {cfg['settings']['mode']}; strategy status: "
          f"{cfg['strategies']['strategies'][cfg['strategies']['active_strategy']]['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
