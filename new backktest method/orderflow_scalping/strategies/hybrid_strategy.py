"""Hybrid score: Score = w1 Z(OFI) + w2 Z(MPD) + w3 Z(VD) + w4 Z(VWAP signal).

* Z-scores use rolling past-only statistics; the microprice term is 0 in proxy mode (no L2 data).
* Weights = OLS of the forward ``horizon``-bar return on the standardised features, fitted on TRAIN rows only.
* Threshold = quantile of |score| on TRAIN (fixed), or per-regime train quantiles (regime-dependent).
The test period never influences weights or thresholds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from orderflow_scalping.indicators.ofi import normalized
from orderflow_scalping.indicators.volume_delta import cumulative
from orderflow_scalping.strategies.common import fresh, rz

LABEL = "Hybrid (proxy)"
FEATURES = ["z_ofi", "z_mpd", "z_vd", "z_vwap"]


def features(df) -> pd.DataFrame:
    return pd.DataFrame({"z_ofi": rz(normalized(df["ofi_proxy"], df["volume"], 5)),
                         "z_mpd": 0.0,  # not observable without L2
                         "z_vd": rz(cumulative(df["delta_bvc"], 10)),
                         "z_vwap": df["vwap_z"]}, index=df.index)


def fit(df, train_mask: np.ndarray, p: dict) -> dict:
    X = features(df)
    fwd = df["close"].shift(-int(p["horizon"])) / df["close"] - 1  # label only (train fit), never a feature
    same_day = df["date"].shift(-int(p["horizon"])) == df["date"]
    m = train_mask & X.notna().all(axis=1).to_numpy() & fwd.notna().to_numpy() & same_day.to_numpy()
    Xt = X[m].to_numpy()
    w = np.linalg.lstsq(np.c_[np.ones(len(Xt)), Xt], fwd[m].to_numpy(), rcond=None)[0][1:]
    score_tr = Xt @ w
    q = float(p["threshold_quantile"])
    thr = {"_all": float(np.quantile(np.abs(score_tr), q))}
    if p.get("regime_thresholds"):
        reg = df["regime"].to_numpy()[m]
        for r in np.unique(reg):
            s = np.abs(score_tr[reg == r])
            if len(s) > 1000:
                thr[str(r)] = float(np.quantile(s, q))
    return {"weights": dict(zip(FEATURES, map(float, w))), "thresholds": thr, "train_rows": int(m.sum())}


def signals(df, p: dict, model: dict):
    X = features(df).fillna(0).to_numpy()
    w = np.array([model["weights"][f] for f in FEATURES])
    score = pd.Series(X @ w, index=df.index)
    thr = df["regime"].map(lambda r: model["thresholds"].get(str(r), model["thresholds"]["_all"])) \
        if p.get("regime_thresholds") else model["thresholds"]["_all"]
    return fresh(score > thr, score < -thr), None, None
