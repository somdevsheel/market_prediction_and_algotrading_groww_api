"""Backup + verification.  python scripts/backup.py [--verify-only <dir>] [--keep 14]

Creates backups/<ts>/ with a consistent SQLite snapshot (online backup API), governance/model files, config and
data-lake manifests + gold tables; writes sha256 checksums; verifies by integrity_check, row counts and
checksums. Optional S3 upload when boto3 is installed and BACKUP_S3_BUCKET is set (verified via object size).
"""
import _bootstrap  # noqa: F401

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from core.clock import IST
from core.config import PROJECT_ROOT, load_config, resolve_path

TABLES = ["signals", "risk_decisions", "orders", "order_events", "trades", "trade_events"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def counts(db: Path) -> dict:
    c = sqlite3.connect(db)
    out = {}
    for t in TABLES:
        try:
            out[t] = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except sqlite3.OperationalError:
            out[t] = None
    c.close()
    return out


def verify(d: Path) -> dict:
    man = json.loads((d / "MANIFEST.json").read_text())
    bad = [f for f, h in man["checksums"].items() if sha(d / f) != h]
    res = {"checksums_ok": not bad, "bad_files": bad}
    db = d / "ledger.sqlite3"
    if db.exists():
        c = sqlite3.connect(db)
        res["integrity"] = c.execute("PRAGMA integrity_check").fetchone()[0]
        c.close()
        res["counts_match"] = counts(db) == man["row_counts"]
    res["ok"] = res["checksums_ok"] and res.get("integrity", "ok") == "ok" and res.get("counts_match", True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only")
    ap.add_argument("--keep", type=int, default=14)
    a = ap.parse_args()
    if a.verify_only:
        print(json.dumps(verify(Path(a.verify_only)), indent=1))
        return 0
    cfg = load_config()
    root = PROJECT_ROOT / "backups"
    d = root / f"{datetime.now(IST):%Y%m%d-%H%M%S}"
    d.mkdir(parents=True)
    src = resolve_path(cfg, "ledger_db")
    if src.exists():
        s, t = sqlite3.connect(src), sqlite3.connect(d / "ledger.sqlite3")
        s.backup(t)
        s.close(), t.close()
    state = src.parent
    for extra in ("governance.json",):
        if (state / extra).exists():
            shutil.copy2(state / extra, d / extra)
    if (state / "models").exists():
        shutil.copytree(state / "models", d / "models")
    shutil.copytree(PROJECT_ROOT / "config", d / "config")
    lake = resolve_path(cfg, "data_lake")
    for sub in ("manifests", "gold"):
        if (lake / sub).exists():
            shutil.copytree(lake / sub, d / "data_lake" / sub)
    files = [p for p in d.rglob("*") if p.is_file()]
    man = {"created_at": datetime.now(IST).isoformat(), "row_counts": counts(d / "ledger.sqlite3") if src.exists() else {},
           "checksums": {str(p.relative_to(d)): sha(p) for p in files}}
    (d / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    res = verify(d)
    print("backup", d, "verified:", res)
    bucket = os.environ.get("BACKUP_S3_BUCKET")
    if bucket:
        try:
            import boto3

            s3 = boto3.client("s3")
            arch = shutil.make_archive(str(d), "gztar", d)
            key = f"nifty-quant/{Path(arch).name}"
            s3.upload_file(arch, bucket, key)
            size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
            print("s3 upload verified:", size == Path(arch).stat().st_size)
        except ImportError:
            print("BACKUP_S3_BUCKET set but boto3 not installed; skipped S3")
    for old in sorted([p for p in root.iterdir() if p.is_dir()])[:-a.keep]:
        shutil.rmtree(old)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
