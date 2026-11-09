"""Download 1-minute near-month NIFTY futures (2020 -> date) into THIS experiment's own data store.

Reuses the main project's Groww provider, idempotent downloader and expiry calendar (imported read-only).
The main project's data lake is only READ (expiry cache, 1-minute index); nothing is written there.

    python3 -m orderflow_scalping.data.ingestion.download --start 2020-01-01 --end 2026-11-09
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta

from orderflow_scalping._paths import MAIN_ROOT, STORE  # noqa: F401  (sets sys.path)
from core.config import load_config
from core.log import setup_logging
from ingestion.historical.datalake import DataLake
from ingestion.historical.downloader import ensure_candles
from ingestion.instruments.calendar import ExpiryCalendar
from ingestion.providers.groww import GrowwProvider


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-11-09")
    a = ap.parse_args()
    cfg = load_config()
    setup_logging(STORE / "logs")
    lake = DataLake(STORE / "lake")
    p = GrowwProvider(cfg["settings"])
    main_cache = MAIN_ROOT / "data_lake" / "manifests" / "expiries_cache.json"
    preload = json.loads(main_cache.read_text()) if main_cache.exists() else {}
    cal = ExpiryCalendar(p, "NIFTY", STORE / "expiries_cache.json", preload)
    s, e = date.fromisoformat(a.start), date.fromisoformat(a.end)
    by: dict[str, list[date]] = {}
    d = s
    while d <= e:
        if d.weekday() < 5:
            sym = cal.futures_symbol_for(d)
            if sym:
                by.setdefault(sym, []).append(d)
        d += timedelta(days=1)
    for sym, ds in by.items():
        df = ensure_candles(p, lake, "futures", sym, "FNO", "1minute", min(ds), max(ds))
        print(f"{sym}: {len(df)} rows", flush=True)
    print("failures:", len(p.failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
