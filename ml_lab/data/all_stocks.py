"""All Indian listed stocks: list from the Groww instrument master, daily prices from Yahoo Finance.

Step 1  `groww`  : Groww API instrument master -> company shares only (ISIN INE····01····, cash segment, type EQ);
                   one row per company (ISIN). NSE-listed companies use Yahoo `SYMBOL.NS`; BSE-only companies use
                   `<BSE symbol>.BO` (scrip-code tickers mostly return nothing on Yahoo; kept as fallback). Saved to data_lake/silver/stock_universe/groww_stocks_<date>.parquet.
Step 2  `yahoo`  : split/dividend-adjusted daily OHLCV from START (2000-01-01) in resumable chunks
                   (data_lake/bronze/yahoo_all/chunk_*.parquet), then combined to
                   data_lake/silver/yahoo_stocks/all_daily.parquet with a coverage + quality report.

    python3 -m ml_lab.data.all_stocks            # both steps
    python3 -m ml_lab.data.all_stocks yahoo      # resume download (+ combine)
    python3 -m ml_lab.data.all_stocks combine    # only rebuild all_daily.parquet + report from downloaded chunks
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import yfinance.shared  # noqa: F401  (per-ticker error registry, used to detect rate limits)

from core.clock import IST
from core.config import load_config, resolve_path

START = "2000-01-01"
CHUNK = 100
CFG = load_config()
LAKE = resolve_path(CFG, "data_lake")
UNI_DIR = LAKE / "silver" / "stock_universe"
RAW_DIR = LAKE / "bronze" / "yahoo_all"


def groww_universe() -> pd.DataFrame:
    from core.secrets import obtain_access_token
    from ingestion.providers.groww import GrowwProvider
    inst = GrowwProvider(CFG["settings"], token=obtain_access_token()).get_instruments()
    eq = inst[(inst.segment == "CASH") & (inst.instrument_type == "EQ")]
    eq = eq[eq["isin"].astype(str).str.match(r"^INE.{4}01")].copy()          # equity shares (not bonds/ETFs/REITs)
    nse = eq[eq.exchange == "NSE"].drop_duplicates("isin").set_index("isin")
    bse = eq[eq.exchange == "BSE"].drop_duplicates("isin").set_index("isin")
    u = pd.DataFrame(index=sorted(set(nse.index) | set(bse.index)))
    u.index.name = "isin"
    u["name"] = nse["name"].combine_first(bse["name"])
    u["nse_symbol"], u["nse_series"] = nse["trading_symbol"], nse["series"]
    u["bse_symbol"], u["bse_code"], u["bse_group"] = bse["trading_symbol"], bse["exchange_token"].astype(str), bse["series"]
    u["groww_symbol"] = nse["groww_symbol"].combine_first(bse["groww_symbol"])
    u["primary_exchange"] = np.where(u.nse_symbol.notna(), "NSE", "BSE")
    u["yahoo"] = np.where(u.nse_symbol.notna(), u.nse_symbol + ".NS", u.bse_symbol + ".BO")       # Yahoo knows BSE by symbol
    u["yahoo_alt"] = np.where(u.nse_symbol.notna(), u.nse_symbol + ".NS", u.bse_code + ".BO")   # scrip code: fallback
    u = u.reset_index()
    UNI_DIR.mkdir(parents=True, exist_ok=True)
    p = UNI_DIR / f"groww_stocks_{datetime.now(IST):%Y%m%d}.parquet"
    u.to_parquet(p, index=False)
    u.to_csv(UNI_DIR / "groww_stocks_latest.csv", index=False)
    print(f"Groww: {len(u)} companies ({(u.primary_exchange == 'NSE').sum()} NSE, {(u.primary_exchange == 'BSE').sum()} BSE-only) -> {p}")
    return u


def latest_universe() -> pd.DataFrame:
    return pd.read_parquet(sorted(UNI_DIR.glob("groww_stocks_2*.parquet"))[-1])


def _download(tickers: list[str]) -> pd.DataFrame:
    raw = yf.download(tickers, start=START, interval="1d", auto_adjust=True, progress=False, threads=True,
                      group_by="ticker")
    frames = []
    for t in tickers:
        if raw.empty or t not in raw.columns.get_level_values(0):
            continue
        x = raw[t].dropna(subset=["Close"]).reset_index().rename(columns=str.lower)
        x = x[(x[["open", "high", "low", "close"]] > 0).all(axis=1)]
        if x.empty:
            continue
        x["yahoo"] = t
        frames.append(x[["date", "yahoo", "open", "high", "low", "close", "volume"]])
    if not frames:
        return pd.DataFrame(columns=["date", "yahoo", "open", "high", "low", "close", "volume"])
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype("float32")
    df["volume"] = df["volume"].fillna(0).astype("int64")
    return df


def _have() -> set[str]:
    fs = list(RAW_DIR.glob("*.parquet"))
    return set(pd.concat(pd.read_parquet(p, columns=["yahoo"]) for p in fs).yahoo.unique()) if fs else set()


def _rate_limited(tickers: list[str]) -> bool:
    errs = getattr(yf.shared, "_ERRORS", {}) or {}
    return any("rate" in str(errs.get(t, "")).lower() or "too many" in str(errs.get(t, "")).lower() for t in tickers)


def wait_for_yahoo(probe: str = "RELIANCE.NS", every: int = 300, max_wait: int = 3 * 3600) -> None:
    """Yahoo blocks an IP for a while after heavy use. Probe every 5 minutes until one small request works."""
    waited = 0
    while waited < max_wait:
        print(f"  Yahoo rate limit: waiting {every // 60} min (waited {waited // 60} min so far)", flush=True)
        time.sleep(every)
        waited += every
        try:
            if len(yf.download(probe, period="5d", progress=False, auto_adjust=True)):
                print("  Yahoo is answering again", flush=True)
                return
        except Exception:
            pass
    raise RuntimeError("Yahoo still rate-limiting after 3 h; rerun later (progress is saved)")


def retry_missing(u: pd.DataFrame, rounds: int = 4, batch: int = 20, pause: float = 6.0) -> None:
    """Yahoo throttles (HTTP 429) after a few thousand tickers. Retry the missing ones slowly, in rounds; a batch
    that returns nothing at all is treated as throttled -> wait and redo once. BSE-only companies are tried by scrip
    symbol first, then by scrip code; either way the rows are stored under the canonical `yahoo` key."""
    nodata_file = RAW_DIR / "_nodata.json"
    nodata: dict[str, int] = json.loads(nodata_file.read_text()) if nodata_file.exists() else {}
    alt = dict(zip(u.yahoo, u.yahoo_alt))
    for rnd in range(rounds):
        have = {dict(zip(u.yahoo_alt, u.yahoo)).get(t, t) for t in _have()}
        miss = [t for t in u.yahoo if t not in have and nodata.get(t, 0) < 3]
        print(f"retry round {rnd + 1}: {len(miss)} missing", flush=True)
        if not miss:
            break
        for i in range(0, len(miss), batch):
            b = miss[i:i + batch]
            q = b if rnd % 2 == 0 else [alt[t] for t in b]                  # odd rounds: scrip-code fallback
            back = {qq: t for qq, t in zip(q, b)}
            while True:
                try:
                    df = _download(q)
                    limited = _rate_limited(q)
                except Exception as e:
                    df, limited = pd.DataFrame(), "rate" in str(e).lower() or "429" in str(e)
                if not limited:
                    break
                if len(df):                                                     # partial batch: keep what came
                    break
                wait_for_yahoo()
            if len(df):
                df["yahoo"] = df.yahoo.map(back)
                df.to_parquet(RAW_DIR / f"retry_{datetime.now():%Y%m%d%H%M%S%f}.parquet", index=False)
            got = set(df.yahoo) if len(df) else set()
            blocked = {back[x] for x in q if _rate_limited([x])}
            for t in b:
                if t not in got and t not in blocked:                           # never count a blocked ticker as "no data"
                    nodata[t] = nodata.get(t, 0) + 1
            nodata_file.write_text(json.dumps(nodata))
            if (i // batch) % 10 == 0:
                print(f"  {i + len(b)}/{len(miss)} tried, +{len(got)} in last batch", flush=True)
            time.sleep(pause)


def yahoo_all(u: pd.DataFrame, download: bool = True) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tickers = u.yahoo.tolist() if download else []
    chunks = [tickers[i:i + CHUNK] for i in range(0, len(tickers), CHUNK)]
    for k, ch in enumerate(chunks):
        p = RAW_DIR / f"chunk_{k:03d}.parquet"
        if p.exists():
            continue
        for attempt in range(3):
            try:
                df = _download(ch)
                break
            except Exception as e:                                              # network / rate limit: back off
                print(f"  chunk {k} attempt {attempt + 1} failed: {type(e).__name__}", flush=True)
                time.sleep(30 * (attempt + 1))
        else:
            continue
        df.to_parquet(p, index=False)
        print(f"  chunk {k + 1}/{len(chunks)}: {df.yahoo.nunique()}/{len(ch)} tickers, {len(df):,} rows", flush=True)
        time.sleep(2)
    if download:
        retry_missing(u)
    all_ = pd.concat([pd.read_parquet(p) for p in sorted(RAW_DIR.glob("*.parquet"))], ignore_index=True)
    alt2main = dict(zip(u.yahoo_alt, u.yahoo))
    all_["yahoo"] = all_.yahoo.map(alt2main).fillna(all_.yahoo)              # .map: Series.replace(dict) is very slow
    all_ = all_.drop_duplicates(["yahoo", "date"]).merge(
        u[["yahoo", "isin", "name", "nse_symbol", "bse_code", "primary_exchange"]], on="yahoo", how="left")
    all_["sym"] = all_.nse_symbol.fillna("BSE:" + all_.bse_code.astype(str))
    out = LAKE / "silver" / "yahoo_stocks" / "all_daily.parquet"
    all_.drop(columns=["nse_symbol", "bse_code"]).sort_values(["sym", "date"]).to_parquet(out, index=False)
    # coverage + quality report
    g = all_.sort_values("date").groupby("yahoo")
    first, last, n = g.date.min(), g.date.max(), g.size()
    big = all_.sort_values(["yahoo", "date"]).groupby("yahoo").close.pct_change().abs() > 0.5
    jumps = big.groupby(all_.loc[big.index, "yahoo"]).sum().astype(int)
    cov = pd.DataFrame({"first": first, "last": last, "rows": n, "jumps_gt_50pct": jumps}).join(
        u.set_index("yahoo")[["isin", "name", "primary_exchange"]])
    cov.to_csv(LAKE / "silver" / "yahoo_stocks" / "all_coverage.csv")
    recent = last >= last.max() - pd.Timedelta(days=10)
    rep = {"at": datetime.now(IST).isoformat(), "groww_companies": len(u), "yahoo_with_data": int(len(cov)),
           "missing": int(len(u) - len(cov)), "active_last_10d": int(recent.sum()), "rows": int(len(all_)),
           "from": str(all_.date.min().date()), "to": str(all_.date.max().date()),
           "by_exchange": cov.primary_exchange.value_counts().to_dict(),
           "history_start_counts": {"<=2005": int((first <= "2005-12-31").sum()), "2006-2015": int(((first > "2005-12-31") & (first <= "2015-12-31")).sum()),
                                    "2016+": int((first > "2015-12-31").sum())},
           "stocks_with_>50pct_daily_jumps": int((jumps > 0).sum()), "adjusted": "auto_adjust=True",
           "output": str(out)}
    (LAKE / "manifests" / "yahoo_all_stocks.json").write_text(json.dumps(rep, indent=1))
    return rep


def main(argv: list[str]) -> int:
    step = argv[0] if argv else "all"
    u = groww_universe() if step in ("all", "groww") else latest_universe()
    if step in ("all", "yahoo", "combine"):
        print(json.dumps(yahoo_all(u, download=step != "combine"), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
