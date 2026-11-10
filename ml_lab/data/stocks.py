"""Download NIFTY 100 stocks (current constituents, niftyindices.com) from yfinance: split- & dividend-adjusted
daily OHLCV from 2010. Survivorship bias: today's constituents only (past members that fell out are missing).

    python3 -m ml_lab.data.stocks
"""
from __future__ import annotations

import io
import json
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf

from core.clock import IST
from core.config import load_config, resolve_path

LIST_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"


def main(start: str = "2010-01-01") -> int:
    root = resolve_path(load_config(), "data_lake")
    out = root / "silver" / "yahoo_stocks"
    out.mkdir(parents=True, exist_ok=True)
    lst = pd.read_csv(io.StringIO(requests.get(LIST_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text))
    lst.to_csv(out / "_constituents.csv", index=False)
    tickers = [f"{s}.NS" for s in lst["Symbol"]]
    raw = yf.download(tickers, start=start, interval="1d", auto_adjust=True, progress=False, threads=True, group_by="ticker")
    ok, missing = 0, []
    frames = []
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            missing.append(t)
            continue
        x = raw[t].dropna(subset=["Close"]).reset_index().rename(columns=str.lower)
        x = x[(x[["open", "high", "low", "close"]] > 0).all(axis=1)]
        if len(x) < 250:
            missing.append(t)
            continue
        x["sym"] = t.replace(".NS", "")
        frames.append(x[["date", "sym", "open", "high", "low", "close", "volume"]])
        ok += 1
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df.to_parquet(out / "nifty100_daily.parquet", index=False)
    rep = {"at": datetime.now(IST).isoformat(), "stocks": ok, "missing": missing, "rows": len(df),
           "from": str(df.date.min().date()), "to": str(df.date.max().date()), "adjusted": "auto_adjust=True"}
    (root / "manifests" / "yahoo_stocks.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
