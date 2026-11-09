"""Idempotent, resumable historical candle ingestion into the data lake.

``ensure_candles`` downloads only windows that are not yet recorded as complete
in the manifest (windows ending today or later are always refreshed), validates
them, writes Bronze + Silver, quarantines bad rows, and returns the Silver data.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta

import pandas as pd

from core.clock import IST
from ingestion.historical.datalake import DataLake
from ingestion.historical.quality import validate_candles
from ingestion.providers.groww import INTERVAL_MINUTES, GrowwProvider

log = logging.getLogger(__name__)


def day_windows(start: date, end: date, max_days: int) -> list[tuple[datetime, datetime]]:
    out, cur = [], start
    while cur <= end:
        w_end = min(end, cur + timedelta(days=max_days - 1))
        out.append((datetime.combine(cur, time(9, 15), IST), datetime.combine(w_end, time(15, 30), IST)))
        cur = w_end + timedelta(days=1)
    return out


def ensure_candles(provider: GrowwProvider, lake: DataLake, dataset: str, symbol: str, segment: str,
                   interval: str, start: date, end: date, today: date | None = None) -> pd.DataFrame:
    today = today or datetime.now(IST).date()
    done = lake.completed_windows(dataset, symbol, interval)
    max_days = int(provider.max_days.get(interval, 30))
    for ws, we in day_windows(start, end, max_days):
        key = (ws.isoformat(), we.isoformat())
        if key in done and we.date() < today:
            continue
        try:
            raw = provider._call("historical", provider.client.get_historical_candles, "NSE", segment, symbol,
                                 ws.strftime("%Y-%m-%d %H:%M:%S"), we.strftime("%Y-%m-%d %H:%M:%S"), interval)
        except Exception as e:  # recorded, never substituted
            provider.failures.append({"symbol": symbol, "interval": interval, "start": key[0], "end": key[1], "error": str(e)})
            lake.append_manifest(dataset, {"symbol": symbol, "interval": interval, "start": key[0], "end": key[1],
                                           "status": "failed", "error": str(e), "source": provider.name})
            log.error("download failed %s %s %s..%s: %s", symbol, interval, ws.date(), we.date(), e)
            continue
        lake.write_bronze(dataset, symbol, interval, ws, we, raw, provider.name,
                          {"segment": segment, "interval": interval})
        df = provider.candles_to_frame(raw.get("candles") or [])
        valid, bad, rep = validate_candles(df, INTERVAL_MINUTES.get(interval))
        lake.quarantine(dataset, symbol, interval, bad)
        n = lake.write_silver(dataset, symbol, interval, valid, provider.name)
        status = "complete" if we.date() < today else "partial"
        lake.append_manifest(dataset, {"symbol": symbol, "interval": interval, "segment": segment,
                                       "start": key[0], "end": key[1], "rows": len(valid), "silver_rows": n,
                                       "status": status, "quality": rep.to_dict(), "source": provider.name})
        log.info("ingested %s %s %s..%s rows=%d missing=%d quarantined=%d", symbol, interval, ws.date(), we.date(),
                 len(valid), rep.missing_intervals, len(bad))
    return lake.read_silver(dataset, symbol, interval,
                            datetime.combine(start, time.min, IST), datetime.combine(end, time.max, IST))
