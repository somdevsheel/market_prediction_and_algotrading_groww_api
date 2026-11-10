"""Does an ML volatility forecast improve the live ORB strategy's 'cheap IV' filter (IV <= forecast vol)?

Uses only OUT-OF-SAMPLE walk-forward predictions of today's NIFTY log range (made with information known at 09:15),
converted to annualised volatility with the Parkinson estimator: sigma_day = range / (2*sqrt(ln 2)).
    python3 -m ml_lab.vol_filter_test [results_dir]
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.engine.orb_runner import run_orb_backtest
from core.config import load_config, resolve_path
from ingestion.providers.groww import GrowwProvider

ROOT = Path(__file__).resolve().parent


def main() -> int:
    rd = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((ROOT / "results").glob("2*"))[-1]
    pr = pd.read_parquet(rd / "pred_volatility.parquet")
    cfg = load_config()
    gold = pd.read_parquet(resolve_path(cfg, "data_lake") / "gold" / "volatility" / "daily_features.parquet")
    p = GrowwProvider(cfg["settings"])
    # scale intraday-range vol to close-to-close vol with a constant fitted on 2008-2020 ONLY (no look-ahead)
    ny = pd.read_parquet(resolve_path(cfg, "data_lake") / "silver" / "yahoo" / "NIFTY.parquet")
    ny = ny[ny.date < "2021-01-01"]
    r_cc = np.log(ny.close).diff().dropna()
    park = np.log(ny.high / ny.low) ** 2 / (4 * np.log(2))
    k = float(np.sqrt((r_cc ** 2).mean()) / np.sqrt(park.mean()))
    print(f"close-to-close / intraday-range vol scale (2008-2020): {k:.3f}")
    rows = []
    for model in ("garch_fc (current)", "mlp", "random_forest", "transformer", "har_baseline"):
        f = gold.copy()
        if model.startswith("garch"):
            feat = "garch_fc"
        else:
            ann = np.exp(pr[model]) / (2 * np.sqrt(np.log(2))) * np.sqrt(252) * 100 * k
            m = pd.Series(ann.to_numpy(), index=pd.to_datetime(pr.date).dt.normalize())
            f["ml_fc"] = pd.to_datetime(f["date"]).dt.normalize().map(m)
            feat = "ml_fc"
        for scn in ("base", "conservative"):
            res = run_orb_backtest(cfg, p, date(2021, 1, 1), date(2026, 11, 6), scn, capital=20000, risk=1000,
                                   features=f, filter_overrides={"rv_forecast_feature": feat}, save=False)
            t = res.trades
            yrs = {k: round(v["net_pnl"]) for k, v in (res.metrics.get("by_year") or {}).items()}
            rows.append({"vol_forecast": model, "costs": scn, "trades": res.metrics.get("trades"),
                         "net": res.metrics.get("net_pnl"), "pf": res.metrics.get("profit_factor"),
                         "since_2024": sum(v for k, v in yrs.items() if k >= "2024"), "by_year": yrs})
            print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(rd / "vol_filter_test.csv", index=False)
    print(out.drop(columns="by_year").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
