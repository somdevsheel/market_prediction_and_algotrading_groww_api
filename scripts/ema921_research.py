"""9/21 EMA crossover research: spot trades -> real-option simulation for 3 modes; saves trade-viewer data.

    python scripts/ema921_research.py --configs 30:10:00:13:30,15:10:30:13:30,15:09:30:14:30 --start 2020-01-01

Output per config: data/backtests/ema921/<tf>m_<start>_<cutoff>/{trades.json, summary.json}
(the dashboard's "EMA 9/21" tab reads these).
"""
import _bootstrap  # noqa: F401

import argparse
import json
from datetime import date, datetime, time

import pandas as pd

from backtesting.engine.ema_options import MODES, EMAOptionSim, metrics
from core.clock import IST
from core.config import load_config, resolve_path
from core.log import setup_logging
from ingestion.historical.datalake import DataLake
from ingestion.instruments.calendar import ExpiryCalendar
from ingestion.providers.groww import GrowwProvider
from strategies.ema_cross.core import EMAConfig, spot_backtest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="30:10:00:13:30,15:10:30:13:30,15:09:30:14:30")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-11-06")
    ap.add_argument("--slip-pts", type=float, default=0.5)
    a = ap.parse_args()
    cfg = load_config()
    setup_logging(resolve_path(cfg, "logs"))
    lake = DataLake(resolve_path(cfg, "data_lake"))
    p = GrowwProvider(cfg["settings"])
    cal = ExpiryCalendar(p, "NIFTY", lake.root / "manifests" / "expiries_cache.json")
    s, e = date.fromisoformat(a.start), date.fromisoformat(a.end)
    df = lake.read_silver("nifty", "NSE-NIFTY", "1minute", datetime.combine(s, time.min, IST),
                          datetime.combine(e, time.max, IST))
    df = df[(df.timestamp.dt.time >= time(9, 15)) & (df.timestamp.dt.time < time(15, 30))][
        ["timestamp", "open", "high", "low", "close"]]
    sim = EMAOptionSim(p, lake, cal, cfg["instruments"]["charges"], 0.5, a.slip_pts)
    root = resolve_path(cfg, "backtests") / "ema921"
    for c in a.configs.split(","):
        tf, sh, sm, ch, cm = c.split(":")
        ec = EMAConfig(int(tf), f"{sh}:{sm}", f"{ch}:{cm}")
        spot, nodays = spot_backtest(df, ec)
        rows = []
        for i, t in spot.iterrows():
            base = {"id": int(i), "date": t["date"], "direction": t["direction"], "side": int(t["side"]),
                    "entry_ts": str(t["entry_ts"]), "exit_ts": str(t["exit_ts"]), "nifty_entry": t["entry_spot"],
                    "nifty_exit": t["exit_spot"], "spot_points": t["spot_points"], "exit_reason": t["exit_reason"],
                    "cross_no": int(t["cross_no"]), "reversal": bool(t["reversal"]),
                    "whipsaw": bool((t["exit_ts"] - t["entry_ts"]).total_seconds() <= 2 * int(tf) * 60
                                    and t["exit_reason"] == "opposite_cross")}
            for m in MODES:
                r = sim.simulate({**t.to_dict()}, m)
                rows.append({**base, **({"mode": m, **r} if "skip" not in r else {"mode": m, "skip": r["skip"]})})
            if i % 50 == 0:
                print(f"{c}: {i}/{len(spot)} trades simulated", flush=True)
        tr = pd.DataFrame(rows)
        ok = tr[tr.get("skip").isna()] if "skip" in tr else tr
        summ = {"config": {"tf": ec.tf, "start": ec.start, "cutoff": ec.cutoff, "slippage": f"0.5% + {a.slip_pts:.2f} pts",
                           "lot": 65, "period": [a.start, a.end]},
                "no_trade_days": nodays, "spot": {"trades": len(spot), "points": round(float(spot.spot_points.sum()), 1)},
                "modes": {}}
        for m in MODES:
            g = ok[ok["mode"] == m].sort_values("entry_ts")
            y = pd.to_datetime(g["date"]).dt.year
            summ["modes"][m] = {**metrics(g), "skipped": int((tr["mode"].eq(m) & tr.get("skip", pd.Series(dtype=str)).notna()).sum()),
                                "by_year": {int(k): round(float(v), 0) for k, v in g.groupby(y)["net_pnl"].sum().items()},
                                "is_2020_23": round(float(g[y <= 2023].net_pnl.sum()), 0),
                                "oos_2024_26": round(float(g[y >= 2024].net_pnl.sum()), 0)}
        od = root / f"{ec.tf}m_{ec.start.replace(':', '')}_{ec.cutoff.replace(':', '')}"
        od.mkdir(parents=True, exist_ok=True)
        tr.to_json(od / "trades.json", orient="records", date_format="iso", default_handler=str)
        (od / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
        print("==", c, json.dumps({m: {k: v for k, v in summ["modes"][m].items() if k != "by_year"} for m in MODES}), flush=True)
        for m in MODES:
            print("   ", m, summ["modes"][m]["by_year"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
