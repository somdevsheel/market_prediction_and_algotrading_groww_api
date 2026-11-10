"""Download daily history from Yahoo Finance (yfinance) into the data lake (dataset 'yahoo').

yfinance is an unofficial Yahoo client: data is free, unaudited and for personal research. Each download keeps
provenance (ticker, timestamp, row count, checksum) in data_lake/manifests/yahoo.jsonl.

    python3 -m ml_lab.data.yahoo
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

import pandas as pd
import yfinance as yf

from core.clock import IST
from core.config import load_config, resolve_path

TICKERS = {
    # India (target + local context)
    "NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "INDIAVIX": "^INDIAVIX",
    # markets whose session t-1 closes before India opens on day t
    "SP500": "^GSPC", "NASDAQ": "^IXIC", "DOW": "^DJI", "USVIX": "^VIX", "FTSE": "^FTSE", "DAX": "^GDAXI",
    "NIKKEI": "^N225", "HANGSENG": "^HSI", "SHANGHAI": "000001.SS",
    # macro
    "USDINR": "USDINR=X", "CRUDE": "CL=F", "GOLD": "GC=F", "US10Y": "^TNX", "DXY": "DX-Y.NYB",
}


def download(start: str = "2007-01-01", end: str | None = None) -> dict:
    root = resolve_path(load_config(), "data_lake")
    out_dir = root / "silver" / "yahoo"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, tk in TICKERS.items():
        df = yf.download(tk, start=start, end=end, interval="1d", auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            report[name] = {"ticker": tk, "rows": 0, "error": "no data"}
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower).reset_index().rename(columns={"Date": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df = df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        df = df[~bad].drop_duplicates("date").sort_values("date")
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        chk = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:16]
        report[name] = {"ticker": tk, "rows": len(df), "from": str(df.date.min().date()), "to": str(df.date.max().date()),
                        "dropped_bad_prices": int(bad.sum()), "sha256_16": chk}
        with open(root / "manifests" / "yahoo.jsonl", "a") as f:
            f.write(json.dumps({"at": datetime.now(IST).isoformat(), "name": name, **report[name]}) + "\n")
        print(f"{name:10s} {tk:10s} {report[name].get('rows', 0):5d} rows  {report[name].get('from')} -> {report[name].get('to')}", flush=True)
    return report


if __name__ == "__main__":
    download()
