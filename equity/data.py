"""Daily equity panel from the data lake (dataset 'equity'): wide DataFrames indexed by date, one column per stock."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.config import load_config, resolve_path


def load_panel(min_days: int = 120, cache: bool = True) -> dict[str, pd.DataFrame]:
    cfg = load_config()
    root = resolve_path(cfg, "data_lake") / "silver" / "equity"
    cp = resolve_path(cfg, "data_lake") / "gold" / "equity" / "daily_panel.parquet"
    files = sorted(root.glob("symbol=*/interval=1day/year=*/month=*/part.parquet"))
    if cache and cp.exists() and cp.stat().st_mtime > max((f.stat().st_mtime for f in files), default=0):
        long = pd.read_parquet(cp)
    else:
        frames = []
        for f in files:
            x = pd.read_parquet(f, columns=["timestamp", "open", "high", "low", "close", "volume"])
            x["sym"] = f.parts[-5].split("=", 1)[1].replace("NSE-", "")
            frames.append(x)
        long = pd.concat(frames, ignore_index=True)
        ts = pd.to_datetime(long["timestamp"])
        # trading date in IST (converting IST midnight to UTC would shift candles to the previous day)
        long["date"] = (ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None) if ts.dt.tz is not None else ts).dt.normalize()
        long = long.drop(columns=["timestamp"]).drop_duplicates(["sym", "date"])
        cp.parent.mkdir(parents=True, exist_ok=True)
        long.to_parquet(cp, index=False)
    panel = {k: long.pivot(index="date", columns="sym", values=k).sort_index() for k in ("open", "high", "low", "close", "volume")}
    keep = panel["close"].notna().sum() >= min_days
    panel = {k: v.loc[:, keep] for k, v in panel.items()}
    bad = (panel["close"] <= 0) | (panel["open"] <= 0)
    bad |= _spikes(panel["close"]) | _spikes(panel["open"])
    n_bad = int(bad.to_numpy().sum())
    for k in ("open", "high", "low", "close", "volume"):  # invalid prices -> NaN (never filled with invented values)
        panel[k] = panel[k].mask(bad)
    panel, ca_log = adjust_corporate_actions(panel)
    panel["turnover"] = panel["close"] * panel["volume"]
    ca_log["bad_prints_removed"] = n_bad
    panel["_corporate_actions"] = ca_log
    return panel


def _spikes(x: pd.DataFrame, k: float = 1.8) -> pd.DataFrame:
    """One-day bad prints: the value jumps by a factor > k versus BOTH neighbours, which agree with each other."""
    prev, nxt = x.shift(1), x.shift(-1)
    up = (x > k * prev) & (x > k * nxt)
    dn = (x < prev / k) & (x < nxt / k)
    neighbours_agree = (nxt / prev - 1).abs() < 0.3
    return (up | dn) & neighbours_agree


SPLIT_RATIOS = [1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0]


def adjust_corporate_actions(panel: dict, tol: float = 0.04, max_unexplained: float = 0.35) -> tuple[dict, dict]:
    """Groww daily candles are NOT adjusted for splits/bonuses. Detect overnight gaps whose price ratio matches a
    standard split/bonus ratio (within ``tol``) and back-adjust earlier prices (divide) and volumes (multiply).
    Stocks that still have an unexplained overnight move beyond ``max_unexplained`` are DROPPED (never guessed).
    Heuristic: a genuine crash that happens to equal e.g. exactly -50% would be mis-adjusted; logged for review."""
    o, c = panel["open"], panel["close"]
    ratio = c.shift(1) / o              # >1 for a split-like drop, <1 for a consolidation-like jump
    adj = pd.DataFrame(1.0, index=c.index, columns=c.columns)
    log = {"adjusted": [], "dropped": []}
    for sym in c.columns:
        r = ratio[sym].dropna()
        cand = r[(r > 1 + max_unexplained * 0.8) | (r < 1 / (1 + max_unexplained))]
        factor = pd.Series(1.0, index=c.index)
        for d, x in cand.items():
            k = x if x > 1 else 1 / x
            near = min(SPLIT_RATIOS, key=lambda s: abs(k / s - 1))
            if abs(k / near - 1) <= tol:
                f = near if x > 1 else 1 / near
                factor[factor.index < d] *= f
                log["adjusted"].append({"sym": sym, "date": str(d.date()), "ratio": round(float(x), 3), "applied": f})
        adj[sym] = factor
    out = {k: (v / adj if k in ("open", "high", "low", "close") else v * adj) for k, v in panel.items()}
    gap = out["open"] / out["close"].shift(1) - 1
    bad = ((gap < -max_unexplained) | (gap > max_unexplained * 1.7)).any()
    for sym in bad[bad].index:
        g = gap[sym]
        log["dropped"].append({"sym": sym, "events": {str(k.date()): round(float(v) * 100, 1)
                                                     for k, v in g[(g < -max_unexplained) | (g > max_unexplained * 1.7)].items()}})
    keep = ~bad
    out = {k: v.loc[:, keep] for k, v in out.items()}
    return out, log


def nifty_daily() -> pd.Series:
    cfg = load_config()
    root = resolve_path(cfg, "data_lake") / "silver" / "nifty" / "symbol=NSE-NIFTY" / "interval=1day"
    files = sorted(root.glob("year=*/month=*/part.parquet"))
    x = pd.concat([pd.read_parquet(f, columns=["timestamp", "close"]) for f in files])
    ts = pd.to_datetime(x["timestamp"])
    x.index = (ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None) if ts.dt.tz is not None else ts).dt.normalize()
    return x["close"].sort_index().groupby(level=0).last()
