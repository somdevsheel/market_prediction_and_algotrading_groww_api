"""Download historical candles into the data lake and print a coverage report.

    python scripts/download_history.py --dataset nifty --symbol NSE-NIFTY --segment CASH --interval 5minute \
        --start 2026-01-01 --end 2026-11-09
    python scripts/download_history.py --coverage-probe   # sample 2020..2026 incl. an expired option contract
"""
import _bootstrap  # noqa: F401

import argparse
from datetime import date

from core.config import load_config, resolve_path
from core.log import setup_logging
from ingestion.historical.datalake import DataLake
from ingestion.historical.downloader import ensure_candles
from ingestion.providers.groww import GrowwProvider, groww_option_symbol


def coverage_probe(p: GrowwProvider, lake: DataLake) -> None:
    """Measured (not assumed) availability per year for index 1m/5m and one expired ATM-ish option per year."""
    for y in range(2020, 2027):
        d = date(y, 3, 2) if y != 2026 else date(2026, 11, 3)
        while d.weekday() >= 5:
            d = d.replace(day=d.day + 1)
        for iv in ("1minute", "5minute"):
            df = ensure_candles(p, lake, "nifty", "NSE-NIFTY", "CASH", iv, d, d)
            print(f"{y} NIFTY {iv} {d}: {len(df)} candles")
        try:
            ex = [e for e in p.get_expiries("NIFTY", d.year, d.month) if e >= d]
        except Exception as e:  # noqa: BLE001
            print(f"{y} expiries error: {e}")
            continue
        idx = ensure_candles(p, lake, "nifty", "NSE-NIFTY", "CASH", "5minute", d, d)
        if not ex or idx.empty:
            print(f"{y} options: no expiries/index for {d}")
            continue
        k = round(float(idx.close.iloc[0]) / 50) * 50
        sym = groww_option_symbol("NIFTY", ex[0], k, "CE")
        o = ensure_candles(p, lake, "options", sym, "FNO", "5minute", d, d)
        vol = int(o.volume.fillna(0).sum()) if len(o) else 0
        oi = o.open_interest.notna().sum() if len(o) else 0
        print(f"{y} option {sym}: {len(o)} candles, volume sum {vol}, OI rows {oi}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--symbol")
    ap.add_argument("--segment", default="CASH")
    ap.add_argument("--interval", default="5minute")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--coverage-probe", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    setup_logging(resolve_path(cfg, "logs"))
    p = GrowwProvider(cfg["settings"])
    lake = DataLake(resolve_path(cfg, "data_lake"))
    if a.coverage_probe:
        coverage_probe(p, lake)
        return 0
    df = ensure_candles(p, lake, a.dataset, a.symbol, a.segment, a.interval, date.fromisoformat(a.start),
                        date.fromisoformat(a.end))
    print(f"{len(df)} candles in silver for {a.symbol} {a.interval}; failures: {p.failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
