"""Build gold research datasets.

    python scripts/build_gold.py features                         # daily vol/regime/event features (+ next session row)
    python scripts/build_gold.py iv --start 2021-01-01 --end 2026-11-09   # ATM IV history from real option candles
    python scripts/build_gold.py ml --start 2021-01-01 --end 2026-11-09 [--base breakout_vwap_v1]
"""
import _bootstrap  # noqa: F401

import argparse
import json
from datetime import date, datetime

import pandas as pd

from backtesting.engine.backtester import Backtester
from core.clock import IST, MarketCalendar
from core.config import load_config, resolve_path
from core.log import setup_logging
from ingestion.historical.datalake import DataLake
from ingestion.historical.iv_history import build_iv_history
from ingestion.instruments.calendar import ExpiryCalendar
from ingestion.providers.groww import GrowwProvider
from quant.features.daily_features import build_daily_features
from quant.machine_learning import pipeline as ml


def gold_path(cfg, *parts):
    p = resolve_path(cfg, "data_lake") / "gold"
    for x in parts:
        p = p / x
    return p


def load_features(cfg) -> pd.DataFrame | None:
    p = gold_path(cfg, "volatility", "daily_features.parquet")
    return pd.read_parquet(p) if p.exists() else None


def load_ml_probs(cfg, model: str = "logistic", base: str = "breakout_vwap_v1") -> dict:
    p = gold_path(cfg, "ml", f"oos_predictions_{base}.parquet")
    if not p.exists():
        return {}
    P = pd.read_parquet(p)
    P = P[P.model == model]
    return dict(zip(P.decision_ts, P.prob))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["features", "iv", "ml"])
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default=datetime.now(IST).date().isoformat())
    ap.add_argument("--base", default="breakout_vwap_v1")
    ap.add_argument("--final-model", default="logistic")
    a = ap.parse_args()
    cfg = load_config()
    setup_logging(resolve_path(cfg, "logs"))
    lake = DataLake(resolve_path(cfg, "data_lake"))
    cal = MarketCalendar(cfg["settings"], cfg["instruments"])
    nifty = lake.read_silver("nifty", "NSE-NIFTY", "5minute")
    if a.what == "features":
        vix = lake.read_silver("india_vix", "NSE-INDIAVIX", "5minute")
        ivp = gold_path(cfg, "implied_volatility", "atm_iv_daily.parquet")
        iv = pd.read_parquet(ivp) if ivp.exists() else None
        last = nifty["timestamp"].dt.date.max()
        f = build_daily_features(nifty, vix, iv, cal.is_trading_day, next_session=cal.next_trading_day(last))
        p = lake.write_gold("volatility", "daily_features", f)
        print(f"{len(f)} rows -> {p}; last row {f['date'].iloc[-1].date()} (next session)")
        # out-of-sample volatility-model evaluation (QLIKE; lower is better)
        from quant.volatility import ewma, garch, hv

        from quant.features.daily_features import daily_closes

        r = hv.log_returns(daily_closes(nifty).close).dropna()
        ev = garch.rolling_oos_evaluation(r, train=500, step=21)
        lam = ewma.compare_lambdas(r * 100).rename(columns={"oos_qlike": "oos_qlike_ewma"})
        ev = pd.concat([ev, lam.assign(model=lambda x: "ewma_" + x["lambda"].astype(str))[["model", "oos_qlike_ewma", "n"]]
                        .rename(columns={"oos_qlike_ewma": "oos_qlike"})], ignore_index=True)
        lake.write_gold("volatility", "model_eval", ev)
        print(ev.to_string(index=False))
        print(f.tail(3).T.to_string())
    elif a.what == "iv":
        p = GrowwProvider(cfg["settings"])
        ec = ExpiryCalendar(p, "NIFTY", lake.root / "manifests" / "expiries_cache.json")
        out = gold_path(cfg, "implied_volatility", "atm_iv_daily.parquet")
        existing = pd.read_parquet(out) if out.exists() else None
        df = build_iv_history(p, lake, ec, nifty, date.fromisoformat(a.start), date.fromisoformat(a.end),
                              cfg["settings"]["backtest"]["risk_free_rate"], cfg["instruments"]["strike_step"], existing)
        lake.write_gold("implied_volatility", "atm_iv_daily", df)
        print(f"{len(df)} days of ATM IV; failures {len(p.failures)}")
    else:
        p = GrowwProvider(cfg["settings"])
        feats = load_features(cfg)
        bt = Backtester(cfg, p, lake, "base", strategy_id=a.base, features=feats)
        ds = ml.build_dataset(bt, date.fromisoformat(a.start), date.fromisoformat(a.end))
        lake.write_gold("ml", f"dataset_{a.base}", ds)
        P, M = ml.walk_forward(ds)
        lake.write_gold("ml", f"oos_predictions_{a.base}", P)
        lake.write_gold("ml", f"metrics_{a.base}", M)
        print(f"samples {len(ds)}  label rate {ds['label'].mean():.3f}")
        print(M.to_string(index=False))
        mp = resolve_path(cfg, "ledger_db").parent / "models" / f"ml_{a.base}_{a.final_model}.pkl"
        ml.fit_final(ds, a.final_model, mp, {"base_strategy": a.base, "model": a.final_model, "samples": len(ds),
                                             "train_start": a.start, "train_end": a.end,
                                             "trained_at": datetime.now(IST).isoformat(),
                                             "oos_metrics": M.to_dict("records")})
        print("final model ->", mp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
