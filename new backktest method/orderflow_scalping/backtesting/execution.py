"""Execution models (all prices are ones that were available at or after the decision time).

* market entry/exit: next bar OPEN +/- (half_spread + slippage)              -> pays the spread
* limit entry      : limit = signal-bar close; fills only if a later bar (within ttl) trades THROUGH the
                     limit by >= trade_through ticks (queue position unknown with candles); else NON-FILL
* stop             : stop-market: stop price (or the open if gapped through) minus spread+slippage
* target           : resting limit: fills only on trade-through; no spread paid
"""
from __future__ import annotations


def market_fill(open_px: float, side: int, half_spread: float, slippage: float) -> float:
    return open_px + side * (half_spread + slippage)  # side +1 buy pays up, -1 sell receives less


def stop_fill(stop_px: float, open_px: float, pos: int, half_spread: float, slippage: float) -> float:
    # closing a long sells: worst of stop and a gapped open; then cross the spread
    base = min(stop_px, open_px) if pos > 0 else max(stop_px, open_px)
    return base - pos * (half_spread + slippage)
