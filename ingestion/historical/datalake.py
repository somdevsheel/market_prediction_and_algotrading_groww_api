"""Bronze / Silver / Gold storage on local Parquet (DuckDB-queryable).

Layout::

    bronze/<dataset>/<symbol>/<interval>/<start>_<end>.json   raw provider response + provenance
    silver/<dataset>/symbol=<symbol>/interval=<interval>/year=<Y>/month=<M>/part.parquet
    quarantine/<dataset>/<symbol>/<interval>/<ts>.parquet
    manifests/<dataset>.jsonl                                 one line per ingestion job

Silver writes are idempotent: rows are merged on ``timestamp`` so re-running a
download never creates duplicates.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from core.clock import IST

SCHEMA_VERSION = 1


def _safe(s: str) -> str:
    return s.replace("/", "_").replace(":", "")


@contextmanager
def _locked(path: Path):
    """Exclusive inter-process lock for read-merge-write of one partition (safe parallel ingestion)."""
    lp = path.with_suffix(path.suffix + ".lock")
    lp.parent.mkdir(parents=True, exist_ok=True)
    with open(lp, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


class DataLake:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        for layer in ("bronze", "silver", "gold", "quarantine", "manifests"):
            (self.root / layer).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ bronze
    def write_bronze(self, dataset: str, symbol: str, interval: str, start: datetime, end: datetime,
                     payload, source: str, params: dict) -> Path:
        body = json.dumps(payload, default=str, sort_keys=True)
        meta = {
            "source": source, "api_version": "growwapi-1.5.0", "dataset": dataset, "symbol": symbol,
            "interval": interval, "start": start.isoformat(), "end": end.isoformat(), "params": params,
            "downloaded_at": datetime.now(IST).isoformat(), "checksum_sha256": hashlib.sha256(body.encode()).hexdigest(),
        }
        p = self.root / "bronze" / dataset / _safe(symbol) / interval / f"{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"meta": meta, "payload": json.loads(body)}))
        return p

    # ------------------------------------------------------------------ silver
    def _silver_dir(self, dataset: str, symbol: str, interval: str) -> Path:
        return self.root / "silver" / dataset / f"symbol={_safe(symbol)}" / f"interval={interval}"

    def write_silver(self, dataset: str, symbol: str, interval: str, df: pd.DataFrame, source: str) -> int:
        if df.empty:
            return 0
        df = df.copy()
        df["source"] = source
        df["schema_version"] = SCHEMA_VERSION
        written = 0
        for (y, m), part in df.groupby([df["timestamp"].dt.year, df["timestamp"].dt.month]):
            d = self._silver_dir(dataset, symbol, interval) / f"year={y}" / f"month={m:02d}"
            d.mkdir(parents=True, exist_ok=True)
            f = d / "part.parquet"
            with _locked(f):
                if f.exists():
                    old = pd.read_parquet(f)
                    part = pd.concat([old, part]).drop_duplicates("timestamp", keep="last")
                part = part.sort_values("timestamp")
                tmp = f.with_suffix(".tmp")
                part.to_parquet(tmp, index=False, compression="zstd")
                tmp.replace(f)  # atomic swap: readers never see a half-written file
            written += len(part)
        return written

    def read_silver(self, dataset: str, symbol: str, interval: str,
                    start: Optional[datetime] = None, end: Optional[datetime] = None) -> pd.DataFrame:
        d = self._silver_dir(dataset, symbol, interval)
        files = sorted(d.glob("year=*/month=*/part.parquet"))
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(IST)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(IST)
        if start is not None:
            df = df[df.timestamp >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.timestamp <= pd.Timestamp(end)]
        return df.sort_values("timestamp").reset_index(drop=True)

    def quarantine(self, dataset: str, symbol: str, interval: str, df: pd.DataFrame) -> Optional[Path]:
        if df.empty:
            return None
        p = self.root / "quarantine" / dataset / _safe(symbol) / interval / f"{datetime.now(IST):%Y%m%dT%H%M%S%f}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
        return p

    # ---------------------------------------------------------------- gold
    def write_gold(self, dataset: str, name: str, df: pd.DataFrame) -> Path:
        p = self.root / "gold" / dataset / f"{_safe(name)}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False, compression="zstd")
        return p

    # ------------------------------------------------------------- manifests
    def append_manifest(self, dataset: str, entry: dict) -> None:
        entry = {"recorded_at": datetime.now(IST).isoformat(), **entry}
        mp = self.root / "manifests" / f"{dataset}.jsonl"
        with _locked(mp), open(mp, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def read_manifest(self, dataset: str) -> list[dict]:
        p = self.root / "manifests" / f"{dataset}.jsonl"
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    def completed_windows(self, dataset: str, symbol: str, interval: str) -> set[tuple[str, str]]:
        return {(e["start"], e["end"]) for e in self.read_manifest(dataset)
                if e.get("symbol") == symbol and e.get("interval") == interval and e.get("status") == "complete"}
