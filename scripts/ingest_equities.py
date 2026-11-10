"""Download daily (or intraday) candles for NSE equities into the data lake (dataset 'equity'). Idempotent,
resumable, shardable. Intraday-eligible (MIS) stocks are downloaded first (more liquid).

    python3 scripts/ingest_equities.py --interval 1day --start 2020-01-01 --end 2026-11-06 --shard 0/2
    python3 scripts/ingest_equities.py --interval 5minute --symbols-file data/universe_intraday.txt ...

Survivorship bias: Groww's instrument list contains only currently-listed stocks; delisted companies are absent.
"""
import _bootstrap  # noqa: F401

import argparse
import json
from datetime import date, datetime

from core.clock import IST
from core.config import load_config, resolve_path
from core.log import setup_logging
from ingestion.historical.datalake import DataLake
from ingestion.historical.downloader import ensure_candles
from ingestion.providers.groww import GrowwProvider


def universe(p: GrowwProvider) -> list[str]:
    df = p.get_instruments()
    eq = df[(df.exchange == "NSE") & (df.segment == "CASH") & (df.instrument_type == "EQ") & (df.series == "EQ")].copy()
    eq["mis"] = eq["is_intraday"].astype(str) == "1"
    eq = eq.sort_values(["mis", "groww_symbol"], ascending=[False, True])
    return eq["groww_symbol"].tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1day")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=datetime.now(IST).date().isoformat())
    ap.add_argument("--shard", default="0/1", help="i/n: this process handles every n-th symbol starting at i")
    ap.add_argument("--symbols-file")
    a = ap.parse_args()
    cfg = load_config()
    setup_logging(resolve_path(cfg, "logs"))
    p = GrowwProvider(cfg["settings"])
    lake = DataLake(resolve_path(cfg, "data_lake"))
    syms = open(a.symbols_file).read().split() if a.symbols_file else universe(p)
    i, n = map(int, a.shard.split("/"))
    mine = syms[i::n]
    out = resolve_path(cfg, "data_lake") / "manifests" / f"equity_progress_{a.interval}_{i}of{n}.json"
    s, e = date.fromisoformat(a.start), date.fromisoformat(a.end)
    done = 0
    for k, sym in enumerate(mine):
        try:
            df = ensure_candles(p, lake, "equity", sym, "CASH", a.interval, s, e)
            done += 1
        except Exception as ex:  # noqa: BLE001
            print(f"{sym}: {type(ex).__name__} {ex}", flush=True)
            if "Authentication" in type(ex).__name__ or "401" in str(ex):
                print("token expired - stopping; re-run later to resume", flush=True)
                break
        if k % 25 == 0:
            out.write_text(json.dumps({"shard": a.shard, "interval": a.interval, "done": k + 1, "of": len(mine),
                                       "last": sym, "at": datetime.now(IST).isoformat()}))
            print(f"[{a.shard}] {k + 1}/{len(mine)} {sym}: {len(df)} rows", flush=True)
    print(f"[{a.shard}] finished {done}/{len(mine)}; failures {len(p.failures)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
