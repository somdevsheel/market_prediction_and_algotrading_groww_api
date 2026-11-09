"""Aggregate ingestion manifests into a completeness / quality report per dataset."""
from __future__ import annotations

import pandas as pd


def completeness(lake) -> dict:
    out = {}
    for f in sorted((lake.root / "manifests").glob("*.jsonl")):
        rows = lake.read_manifest(f.stem)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        q = pd.json_normalize(df["quality"].dropna()) if "quality" in df else pd.DataFrame()
        out[f.stem] = {
            "windows": len(df),
            "symbols": int(df["symbol"].nunique()) if "symbol" in df else None,
            "status": df["status"].value_counts().to_dict() if "status" in df else {},
            "rows_valid": int(q["rows_valid"].sum()) if "rows_valid" in q else None,
            "duplicates": int(q["duplicates"].sum()) if "duplicates" in q else None,
            "invalid_ohlc": int(q["invalid_ohlc"].sum()) if "invalid_ohlc" in q else None,
            "non_positive_prices": int(q["non_positive_prices"].sum()) if "non_positive_prices" in q else None,
            "outside_session_quarantined": int(q["outside_session"].sum()) if "outside_session" in q else None,
            "missing_intervals": int(q["missing_intervals"].sum()) if "missing_intervals" in q else None,
            "failed_windows": df[df.get("status") == "failed"][["symbol", "start", "end"]].head(10).to_dict("records")
            if "status" in df else [],
        }
    return out
