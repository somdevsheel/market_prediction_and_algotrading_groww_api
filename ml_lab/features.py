"""Daily dataset for NIFTY with strict information timing (decision at 09:15 IST on day t).

Allowed at decision time:
* NIFTY / BANKNIFTY / INDIAVIX values up to day t-1's close, and today's opening gap (open_t / close_{t-1}).
* Every external market / macro series: only observations dated STRICTLY BEFORE day t (merge_asof, no exact
  match). US/Europe sessions dated t-1 close before 09:15 IST on t; Asian sessions dated t are excluded.
Targets (day t, after the decision):
* y_dir  = 1 if close_t > open_t else 0                     (intraday direction, tradable from the open)
* y_ret  = close_t / open_t - 1
* y_vol  = log(high_t / low_t)                                (Parkinson range: realized volatility proxy)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import load_config, resolve_path

EXTERNAL = ["SP500", "NASDAQ", "DOW", "USVIX", "FTSE", "DAX", "NIKKEI", "HANGSENG", "SHANGHAI",
            "USDINR", "CRUDE", "GOLD", "US10Y", "DXY"]


def _load(name: str) -> pd.DataFrame:
    p = resolve_path(load_config(), "data_lake") / "silver" / "yahoo" / f"{name}.parquet"
    return pd.read_parquet(p).sort_values("date").reset_index(drop=True)


def build_dataset() -> pd.DataFrame:
    n = _load("NIFTY")
    n = n[(n.high >= n.low) & (n.high > 0)].copy()
    d = pd.DataFrame({"date": n.date})
    o, h, l, c = n.open.to_numpy(), n.high.to_numpy(), n.low.to_numpy(), n.close.to_numpy()
    # Yahoo sometimes reports open == previous close for old NIFTY data -> gap 0; flagged, not invented
    d["gap"] = np.r_[np.nan, o[1:] / c[:-1] - 1]
    rcc = pd.Series(np.r_[np.nan, c[1:] / c[:-1] - 1])
    roc = pd.Series(c / o - 1)
    rng = pd.Series(np.log(h / l))
    for k in (1, 2, 3, 5, 10, 20):
        d[f"r_cc_{k}"] = (pd.Series(c).pct_change(k)).shift(1)                 # through t-1 close
    d["r_oc_1"] = roc.shift(1)
    d["rv_d"] = rng.shift(1)                                                       # HAR components (t-1 ...)
    d["rv_w"] = rng.shift(1).rolling(5).mean()
    d["rv_m"] = rng.shift(1).rolling(22).mean()
    d["vol_20"] = rcc.shift(1).rolling(20).std()
    d["dist_ma20"] = (pd.Series(c) / pd.Series(c).rolling(20).mean() - 1).shift(1)
    d["dist_ma200"] = (pd.Series(c) / pd.Series(c).rolling(200).mean() - 1).shift(1)
    d["dow"] = d.date.dt.dayofweek
    for name, col in (("BANKNIFTY", "bank"), ("INDIAVIX", "vix")):
        x = _load(name)
        x = x.set_index("date")["close"].reindex(d.date).to_numpy()
        s = pd.Series(x)
        d[f"{col}_r1"] = s.pct_change(fill_method=None).shift(1)
        if col == "vix":
            d["vix_lvl"] = s.shift(1)
            d["vix_ma_ratio"] = (s / s.rolling(20).mean()).shift(1)
    for name in EXTERNAL:  # strictly earlier sessions only
        x = _load(name)[["date", "close"]].copy()
        x[f"{name.lower()}_r1"] = x.close.pct_change()
        x[f"{name.lower()}_r5"] = x.close.pct_change(5)
        if name in ("USVIX", "US10Y"):
            x[f"{name.lower()}_lvl"] = x.close
        x = x.drop(columns="close")
        d = pd.merge_asof(d.sort_values("date"), x.sort_values("date"), on="date", allow_exact_matches=False)
    d["y_ret"] = roc.to_numpy()
    d["y_dir"] = (roc > 0).astype(int).to_numpy()
    d["y_vol"] = rng.to_numpy()
    d = d.replace([np.inf, -np.inf], np.nan)
    feats = [c for c in d.columns if c not in ("date", "y_ret", "y_dir", "y_vol")]
    d = d.dropna(subset=feats + ["y_vol"]).reset_index(drop=True)
    d.attrs["features"] = feats
    return d
