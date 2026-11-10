"""Rank NSE EQ stocks by median daily turnover over the last ~60 sessions (1 request per stock) and write the
top-N list for intraday research (data/universe_intraday.txt). Current liquidity only -> mild look-ahead in
universe choice, documented in reports.

    python3 scripts/rank_liquidity.py --top 100
"""
import _bootstrap  # noqa: F401

import argparse
import json
from datetime import datetime, timedelta

import pandas as pd

from core.clock import IST
from core.config import PROJECT_ROOT, load_config
from ingestion.providers.groww import GrowwProvider


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=100)
    a = ap.parse_args()
    p = GrowwProvider(load_config()["settings"])
    df = p.get_instruments()
    eq = df[(df.exchange == "NSE") & (df.segment == "CASH") & (df.instrument_type == "EQ") & (df.series == "EQ")
            & (df.is_intraday.astype(str) == "1")]
    end = datetime.now(IST).date()
    start = end - timedelta(days=90)
    rows = []
    for i, sym in enumerate(eq.groww_symbol):
        try:
            r = p._call("historical", p.client.get_historical_candles, "NSE", "CASH", sym,
                        f"{start} 09:15:00", f"{end} 15:30:00", "1day")
            c = pd.DataFrame(r.get("candles") or [], columns=["t", "o", "h", "l", "c", "v", "oi"][:len((r.get("candles") or [[0] * 7])[0])])
            if len(c) >= 20:
                rows.append({"sym": sym, "median_turnover_cr": float((c["c"] * c["v"]).median() / 1e7), "last": float(c["c"].iloc[-1])})
        except Exception as e:  # noqa: BLE001
            print(sym, "failed:", e, flush=True)
        if i % 200 == 0:
            print(f"{i}/{len(eq)}", flush=True)
    r = pd.DataFrame(rows).sort_values("median_turnover_cr", ascending=False)
    r.to_csv(PROJECT_ROOT / "data" / "liquidity_ranking.csv", index=False)
    top = r.head(a.top)
    (PROJECT_ROOT / "data" / "universe_intraday.txt").write_text("\n".join(top.sym) + "\n")
    print(json.dumps({"ranked": len(r), "top_min_turnover_cr": round(top.median_turnover_cr.min(), 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
