"""Option contract selection (Phase 6).

Direction comes from the signal; this module picks the contract and validates
it, or returns a rejection reason. It never picks by volume/OI alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from core.models import OptionChain, OptionContract, Quote


@dataclass
class Selection:
    ok: bool
    contract: Optional[OptionContract] = None
    quote: Optional[Quote] = None
    chain_row: Optional[dict] = None
    reason: str = ""
    checks: dict = field(default_factory=dict)


def itm_strike(spot: float, option_type: str, step: float, itm_steps: int = 1) -> float:
    """1-ITM CE = highest strike strictly below spot; 1-ITM PE = lowest strike strictly above spot.
    itm_steps <= 0 means ATM (nearest strike)."""
    if itm_steps <= 0:
        return round(spot / step) * step
    if option_type == "CE":
        k = math.floor(spot / step) * step
        if k >= spot:
            k -= step
        return k - step * (itm_steps - 1)
    k = math.ceil(spot / step) * step
    if k <= spot:
        k += step
    return k + step * (itm_steps - 1)


def choose_expiry(expiries: list[date], today: date, avoid_expiry_day: bool) -> Optional[date]:
    valid = sorted(e for e in expiries if e > today or (e == today and not avoid_expiry_day))
    return valid[0] if valid else None


def validate_quote(q: Quote, now: datetime, cfg: dict, max_age_s: float) -> tuple[bool, str, dict]:
    checks = {"bid": q.bid, "ask": q.ask, "ltp": q.ltp, "spread_pct": q.spread_pct, "volume": q.volume,
              "oi": q.open_interest}
    if not q.bid or not q.ask or q.bid <= 0 or q.ask <= 0:
        return False, "no_two_sided_quote", checks
    if q.ask < q.bid:
        return False, "crossed_quote", checks
    if q.spread_pct is None or q.spread_pct > cfg["max_spread_pct"]:
        return False, f"spread_too_wide:{q.spread_pct:.2f}%", checks
    if (q.volume or 0) < cfg["min_volume"]:
        return False, f"low_volume:{q.volume}", checks
    if (q.open_interest or 0) < cfg["min_open_interest"]:
        return False, f"low_open_interest:{q.open_interest}", checks
    if not (cfg["min_premium"] <= q.ask <= cfg["max_premium"]):
        return False, f"premium_out_of_bounds:{q.ask}", checks
    if q.last_trade_time is not None:
        age = (now - q.last_trade_time).total_seconds()
        checks["quote_age_s"] = age
        if age > max_age_s:
            return False, f"stale_quote:{age:.0f}s", checks
    return True, "", checks


def select_contract(direction: str, spot: float, now: datetime, expiries: list[date], chain: OptionChain,
                    instruments_lookup, quote_fn, inst_cfg: dict, max_quote_age_s: float) -> Selection:
    """``instruments_lookup(trading_symbol) -> dict`` (instrument master row);
    ``quote_fn(trading_symbol) -> Quote``."""
    cs = inst_cfg["contract_selection"]
    opt = "CE" if direction == "BULLISH" else "PE"
    expiry = choose_expiry(expiries, now.date(), cs.get("avoid_expiry_day", True))
    if expiry is None:
        return Selection(False, reason="no_valid_expiry")
    if chain.expiry != expiry:
        return Selection(False, reason=f"chain_expiry_mismatch:{chain.expiry}!={expiry}")
    strike = itm_strike(spot, opt, inst_cfg["strike_step"], cs.get("itm_steps", 1))
    row = chain.get(strike, opt)
    if row is None or not row.trading_symbol:
        return Selection(False, reason=f"strike_not_in_chain:{strike}{opt}")
    inst = instruments_lookup(row.trading_symbol)
    if not inst:
        return Selection(False, reason=f"instrument_not_found:{row.trading_symbol}")
    if str(inst.get("instrument_type")) != opt or float(inst.get("strike_price")) != strike:
        return Selection(False, reason="instrument_mismatch")
    if str(inst.get("expiry_date"))[:10] != expiry.isoformat():
        return Selection(False, reason="instrument_expiry_mismatch")
    if int(inst.get("buy_allowed", 1)) != 1:
        return Selection(False, reason="buy_not_allowed")
    lot = int(inst["lot_size"])
    contract = OptionContract(underlying=inst_cfg["underlying"], trading_symbol=row.trading_symbol,
                              groww_symbol=inst.get("groww_symbol"), expiry=expiry, strike=strike, option_type=opt,
                              lot_size=lot, tick_size=float(inst.get("tick_size") or inst_cfg["tick_size"]))
    q = quote_fn(row.trading_symbol)
    ok, why, checks = validate_quote(q, now, cs, max_quote_age_s)
    checks.update({"strike": strike, "expiry": expiry.isoformat(), "lot_size": lot,
                   "chain_delta": row.delta, "chain_iv": row.iv})
    if not ok:
        return Selection(False, contract=contract, quote=q, reason=why, checks=checks)
    return Selection(True, contract=contract, quote=q, chain_row=row.__dict__, checks=checks)
