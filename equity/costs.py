"""Groww equity charges (VERIFY current rates). Delivery (CNC) and intraday (MIS)."""
from __future__ import annotations

DELIVERY = {"brokerage_pct": 0.1, "brokerage_max": 20.0, "brokerage_min": 5.0, "stt_buy_pct": 0.1, "stt_sell_pct": 0.1,
            "exchange_pct": 0.00297, "sebi_pct": 0.0001, "stamp_buy_pct": 0.015, "ipft_pct": 0.0001, "gst_pct": 18.0,
            "dp_per_sell": 15.93}   # DP charge per scrip per sell day (incl. GST)
INTRADAY = {"brokerage_pct": 0.1, "brokerage_max": 20.0, "brokerage_min": 5.0, "stt_buy_pct": 0.0, "stt_sell_pct": 0.025,
            "exchange_pct": 0.00297, "sebi_pct": 0.0001, "stamp_buy_pct": 0.003, "ipft_pct": 0.0001, "gst_pct": 18.0,
            "dp_per_sell": 0.0}


def order_charges(value: float, side: str, c: dict) -> float:
    """Charges for ONE order (buy or sell) of ``value`` rupees."""
    brk = min(c["brokerage_max"], max(c["brokerage_min"], value * c["brokerage_pct"] / 100))
    stt = value * (c["stt_buy_pct"] if side == "buy" else c["stt_sell_pct"]) / 100
    exch = value * c["exchange_pct"] / 100
    sebi = value * c["sebi_pct"] / 100
    ipft = value * c["ipft_pct"] / 100
    stamp = value * c["stamp_buy_pct"] / 100 if side == "buy" else 0.0
    gst = (brk + exch + sebi + ipft) * c["gst_pct"] / 100
    dp = c["dp_per_sell"] if side == "sell" else 0.0
    return brk + stt + exch + sebi + ipft + stamp + gst + dp
