"""Data completeness & quality report from manifests + instrument master checks.

    python scripts/data_report.py [--instruments]   (--instruments downloads + snapshots the master; needs token)
"""
import _bootstrap  # noqa: F401

import argparse
import json

from core.clock import MarketCalendar
from core.config import load_config, resolve_path
from ingestion.historical.datalake import DataLake
from monitoring.data_quality import completeness


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruments", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    lake = DataLake(resolve_path(cfg, "data_lake"))
    rep = completeness(lake)
    if a.instruments:
        from datetime import date

        from ingestion.instruments.master import check_contracts, snapshot
        from ingestion.providers.groww import GrowwProvider

        df = GrowwProvider(cfg["settings"]).get_instruments()
        rep["instrument_snapshot"] = snapshot(lake, df)
        cal = MarketCalendar(cfg["settings"], cfg["instruments"])
        rep["instrument_quality"] = check_contracts(df, date.today(), cal.is_trading_day)
    lake.root.joinpath("manifests", "data_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
