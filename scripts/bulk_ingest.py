"""Bulk, resumable historical ingestion (idempotent; safe to re-run / interrupt).

    python scripts/bulk_ingest.py --start 2020-01-01 --end 2026-11-09 [--intervals 5minute,1day] [--skip-futures]

Downloads NIFTY index and INDIA VIX for each interval, and near-month NIFTY futures (5minute) contract by contract.
"""
import _bootstrap  # noqa: F401

import argparse
from datetime import date, timedelta

from core.config import load_config, resolve_path
from core.log import setup_logging
from ingestion.historical.datalake import DataLake
from ingestion.historical.downloader import ensure_candles
from ingestion.instruments.calendar import ExpiryCalendar
from ingestion.providers.groww import GrowwProvider


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--intervals", default="1minute,5minute,15minute,30minute,1hour,1day")
    ap.add_argument("--skip-futures", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    setup_logging(resolve_path(cfg, "logs"))
    p = GrowwProvider(cfg["settings"])
    lake = DataLake(resolve_path(cfg, "data_lake"))
    s, e = date.fromisoformat(a.start), date.fromisoformat(a.end)
    for iv in a.intervals.split(","):
        for ds, sym in (("nifty", "NSE-NIFTY"), ("india_vix", "NSE-INDIAVIX")):
            df = ensure_candles(p, lake, ds, sym, "CASH", iv, s, e)
            print(f"{ds} {iv}: {len(df)} rows")
    if not a.skip_futures:
        cal = ExpiryCalendar(p, "NIFTY", resolve_path(cfg, "data_lake") / "manifests" / "expiries_cache.json")
        by: dict[str, list[date]] = {}
        d = s
        while d <= e:
            if d.weekday() < 5:
                sym = cal.futures_symbol_for(d)
                if sym:
                    by.setdefault(sym, []).append(d)
            d += timedelta(days=1)
        for sym, ds in by.items():
            df = ensure_candles(p, lake, "futures", sym, "FNO", "5minute", min(ds), max(ds))
            print(f"futures {sym}: {len(df)} rows")
    print("failures:", len(p.failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
