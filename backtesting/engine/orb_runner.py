"""Fast, correct backtest runner for ORB-class strategies (shared by scripts/backtest_strategy.py and the API).

Signals come from strategies/orb/orb_core.py (identical to live); the Backtester applies the contract-level IV
filter, the risk engine and real 1-minute option candles with the scenario's friction and full charges.
"""
from __future__ import annotations

import copy
from datetime import date, datetime, time

import pandas as pd

from backtesting.engine.backtester import Backtester, BacktestResult
from core.clock import IST
from core.config import resolve_path
from ingestion.historical.datalake import DataLake
from quant.features.daily_features import features_for
from strategies.base import MarketContext
from strategies.orb.orb_core import find_entry, passes_filters
from strategies.registry import build_strategy


def run_orb_backtest(cfg: dict, provider, start: date, end: date, scenario: str = "base", strategy_id: str | None = None,
                     capital: float | None = None, risk: float | None = None, square_off: str | None = None,
                     itm_steps: int | None = None, charges_multiplier: float = 1.0, save: bool = True,
                     reward_risk: float | None = None) -> BacktestResult:
    """``reward_risk``: option-premium target = entry + reward_risk x stop distance. None/0 = no fixed target
    (strategy default: winners run to the underlying stop or square-off)."""
    cfg = copy.deepcopy(cfg)
    sid = strategy_id or cfg["strategies"]["active_strategy"]
    scfg = cfg["strategies"]["strategies"][sid]
    if scfg.get("class") != "orb":
        raise ValueError(f"{sid}: only ORB-class strategies are supported")
    if capital:
        cfg["risk"]["capital"] = cfg["risk"]["max_premium_outlay"] = float(capital)
    if risk:
        cfg["risk"]["risk_per_trade_max"] = float(risk)
    if square_off:
        cfg["settings"]["session"]["square_off"] = square_off
    if itm_steps is not None:
        cfg["instruments"]["contract_selection"]["itm_steps"] = int(itm_steps)
    if reward_risk:  # strategy-level risk_overrides take precedence in the Backtester, so set it there
        scfg.setdefault("risk_overrides", {})["reward_risk"] = float(reward_risk)
    lake = DataLake(resolve_path(cfg, "data_lake"))
    fp = lake.root / "gold" / "volatility" / "daily_features.parquet"
    feats = pd.read_parquet(fp) if fp.exists() else None
    df = lake.read_silver("nifty", "NSE-NIFTY", "1minute", datetime.combine(start, time.min, IST),
                          datetime.combine(end, time.max, IST))
    if df.empty:
        raise ValueError("no 1-minute NIFTY data in the lake for that range (run scripts/bulk_ingest.py)")
    df = df[(df.timestamp.dt.time >= time(9, 15)) & (df.timestamp.dt.time < time(15, 30))]
    days = {d: g.reset_index(drop=True) for d, g in df.groupby(df.timestamp.dt.date) if len(g) >= 300}
    p = scfg["params"]
    strat = build_strategy(sid, scfg)
    sigs = []
    for d, day in days.items():
        ent = find_entry(day, p["mode"], int(p["or_minutes"]))
        if ent is None or not passes_filters(ent, p.get("orb_filters") or {}):
            continue
        f = features_for(feats, d)
        sig = strat.evaluate(MarketContext(now=ent.entry_ts.to_pydatetime(), index_5m=day.iloc[:ent.entry_idx + 1], features=f))
        if sig:
            sigs.append((d, ent.entry_ts, sig, f, day[["timestamp", "high", "low", "close"]]))
    exp = sid if not reward_risk else f"{sid}_rr{float(reward_risk):g}"
    bt = Backtester(cfg, provider, lake, scenario, strategy_id=sid, experiment=exp, interval=1, features=feats,
                    charges_multiplier=charges_multiplier)
    res = bt.run(start, end, precomputed=(sigs, sorted(days)))
    if save:
        res.save(resolve_path(cfg, "backtests"))
    return res
