"""Record LIVE Level-2 snapshots (5-level depth) of the near-month NIFTY future during market hours, so genuine
OFI (Cont-Kukanov-Stoikov) and microprice strategies can be backtested later. Read-only market data; no orders.

    python3 -m orderflow_scalping.data.ingestion.record_depth --seconds 1      (run 09:15-15:30 IST)

Snapshots -> data/storage/depth/<date>.parquet (bid/ask L1-L5 prices & quantities, LTP, volume, receive time).
REST polling (~1/s) is far coarser than true tick data; results from it are labelled 'snapshot-L2'.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

import pandas as pd

from orderflow_scalping._paths import STORE  # noqa: F401
from core.clock import IST
from core.config import load_config
from ingestion.providers.groww import GrowwProvider


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=1.0)
    a = ap.parse_args()
    cfg = load_config()
    p = GrowwProvider(cfg["settings"])
    fut = p.futures_contracts("NIFTY").iloc[0]
    sym = fut["trading_symbol"]
    rows, out = [], STORE / "depth" / f"{datetime.now(IST):%Y-%m-%d}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"recording depth for {sym} every {a.seconds}s -> {out}")
    while True:
        now = datetime.now(IST)
        if now.hour * 60 + now.minute >= 15 * 60 + 30:
            break
        try:
            q = p.get_quote(sym, "FNO")
            dep = q.raw.get("depth") or {}
            r = {"ts": now, "ltp": q.ltp, "volume": q.volume}
            for side in ("buy", "sell"):
                for i, lv in enumerate((dep.get(side) or [])[:5]):
                    r[f"{side}{i + 1}_px"], r[f"{side}{i + 1}_qty"] = lv.get("price"), lv.get("quantity")
            rows.append(r)
        except Exception as e:  # noqa: BLE001
            print("quote failed:", e)
        if len(rows) % 300 == 0 and rows:
            pd.DataFrame(rows).to_parquet(out, index=False)
        time.sleep(a.seconds)
    pd.DataFrame(rows).to_parquet(out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
