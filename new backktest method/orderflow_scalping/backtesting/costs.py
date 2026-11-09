"""NIFTY futures transaction charges (rates in config/settings.yaml -> costs)."""
from __future__ import annotations


def futures_charges(buy_notional: float, sell_notional: float, c: dict) -> dict:
    turnover = buy_notional + sell_notional
    brokerage = c["brokerage_per_order"] * 2
    stt = sell_notional * c["stt_sell_pct"] / 100
    exch = turnover * c["exchange_txn_pct"] / 100
    sebi = turnover * c["sebi_pct"] / 100
    stamp = buy_notional * c["stamp_buy_pct"] / 100
    ipft = turnover * c["ipft_pct"] / 100
    gst = (brokerage + exch + sebi + ipft) * c["gst_pct"] / 100
    out = {"brokerage": brokerage, "stt": stt, "exchange": exch, "sebi": sebi, "stamp": stamp, "ipft": ipft, "gst": gst}
    out["total"] = sum(out.values())
    return out
