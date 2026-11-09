"""Instrument master snapshot + contract quality checks (lot sizes, expiries, symbol mapping)."""
from __future__ import annotations

import hashlib
import re
from datetime import date, datetime

import pandas as pd

from core.clock import IST

_GS = re.compile(r"^NSE-(?P<u>[A-Z]+)-(?P<d>\d{2}[A-Za-z]{3}\d{2})-(?:(?P<k>\d+(?:\.\d+)?)-(?P<t>CE|PE)|FUT)$")


def snapshot(lake, df: pd.DataFrame) -> dict:
    """Store the raw instrument master in bronze (csv + checksum) and the NIFTY F&O subset in silver."""
    d = datetime.now(IST)
    raw = df.to_csv(index=False).encode()
    p = lake.root / "bronze" / "instruments" / f"{d:%Y%m%d}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(raw)
    sub = df[(df.underlying_symbol == "NIFTY") & (df.segment == "FNO")].copy()
    sp = lake.root / "silver" / "instruments" / f"nifty_fno_{d:%Y%m%d}.parquet"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sub.astype(str).to_parquet(sp, index=False)
    return {"bronze": str(p), "silver": str(sp), "rows": len(df), "nifty_fno_rows": len(sub),
            "sha256": hashlib.sha256(raw).hexdigest()}


def check_contracts(df: pd.DataFrame, today: date, is_trading_day=None) -> dict:
    """Quality report for NIFTY F&O rows of the instrument master."""
    n = df[(df.underlying_symbol == "NIFTY") & (df.segment == "FNO")].copy()
    n["expiry"] = pd.to_datetime(n["expiry_date"], errors="coerce").dt.date
    issues: dict = {}
    bad_exp = n[n["expiry"].isna() | (n["expiry"] < today)]
    issues["invalid_or_past_expiry"] = len(bad_exp)
    if is_trading_day:
        issues["expiry_on_non_trading_day"] = int(sum(not is_trading_day(e) for e in n["expiry"].dropna().unique()))
    lots = n.groupby("expiry")["lot_size"].nunique()
    issues["expiries_with_mixed_lot_sizes"] = [str(k) for k, v in lots.items() if v > 1]
    issues["lot_sizes"] = sorted({int(float(x)) for x in n["lot_size"].dropna().unique()})
    mapping_errors = []
    for _, r in n.iterrows():
        m = _GS.match(str(r["groww_symbol"]))
        if not m:
            mapping_errors.append(r["groww_symbol"])
            continue
        if datetime.strptime(m["d"], "%d%b%y").date() != r["expiry"]:
            mapping_errors.append(r["groww_symbol"])
        elif m["t"] and (m["t"] != r["instrument_type"] or float(m["k"]) != float(r["strike_price"])):
            mapping_errors.append(r["groww_symbol"])
    issues["symbol_mapping_errors"] = len(mapping_errors)
    issues["mapping_error_examples"] = mapping_errors[:5]
    issues["contracts"] = len(n)
    issues["ok"] = (issues["invalid_or_past_expiry"] == 0 and not issues["expiries_with_mixed_lot_sizes"]
                    and issues["symbol_mapping_errors"] == 0)
    return issues
