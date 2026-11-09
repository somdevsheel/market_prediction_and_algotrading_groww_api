"""ML research pipeline (optional; never controls live orders unless a strategy using it is approved).

* Samples: candidate signals from the causal replay (``Backtester.iter_signals``) of a *base* strategy.
* Features: only information available at the decision time (signal diagnostics, time of day, daily gold
  features known before the open). No future candles.
* Label: the underlying reaches the signal's target before its invalidation level before square-off (1/0).
  Labels use future candles by definition; features never do.
* Validation: chronological, expanding-window walk-forward (refit every ``refit_months``). No shuffling.
* Output: out-of-sample probabilities per decision timestamp (gold/ml), metrics per model, final model file.
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

FEATURES = ["dir", "minutes_from_open", "extension", "volume_ratio", "or_width", "risk_pts", "fut_vwap_dist",
            "hv10", "hv20", "hv_ratio", "ewma_fc", "garch_fc", "vix_prev", "vix_chg", "ret_prev", "trend_prev",
            "range_prev", "hmm_p0", "hmm_p1", "hmm_p2", "event_flag", "rv_iv_ratio", "iv_pctile"]


def make_models(seed: int = 0) -> dict:
    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=10, random_state=seed),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                          colsample_bytree=0.8, random_state=seed, eval_metric="logloss")
    except Exception:  # noqa: BLE001
        pass
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, num_leaves=7,
                                            min_child_samples=15, random_state=seed, verbose=-1)
    except Exception:  # noqa: BLE001
        pass
    return models


def label_signal(sig, day_idx: pd.DataFrame, decision: pd.Timestamp, square_off: time) -> int:
    fut = day_idx[day_idx["timestamp"] >= decision]
    for _, b in fut.iterrows():
        if b["timestamp"].time() >= square_off:
            break
        if sig.direction == "BULLISH":
            if b["low"] <= sig.invalidation_level:
                return 0
            if b["high"] >= sig.target_level:
                return 1
        else:
            if b["high"] >= sig.invalidation_level:
                return 0
            if b["low"] <= sig.target_level:
                return 1
    return 0


def sample_row(d: date, decision: pd.Timestamp, sig, feats: dict) -> dict:
    g = sig.diagnostics
    row = {"date": d.isoformat(), "decision_ts": decision.isoformat(), "direction": sig.direction,
           "dir": 1 if sig.direction == "BULLISH" else -1,
           "minutes_from_open": (decision.hour * 60 + decision.minute) - (9 * 60 + 15),
           "extension": g.get("extension"), "volume_ratio": g.get("volume_ratio"),
           "or_width": (g.get("or_high") or np.nan) - (g.get("or_low") or np.nan),
           "risk_pts": abs(sig.reference_price - sig.invalidation_level),
           "fut_vwap_dist": (g.get("fut_close") - g.get("fut_vwap")) if g.get("fut_close") and g.get("fut_vwap") else np.nan,
           "event_flag": 1 if feats.get("events") else 0}
    for k in FEATURES:
        if k not in row:
            row[k] = feats.get(k, np.nan)
    return row


def build_dataset(bt, start: date, end: date) -> pd.DataFrame:
    idx, fut = bt.load_underlying(start, end)
    days = sorted(d for d in idx["timestamp"].dt.date.unique() if start <= d <= end and bt.cal.is_trading_day(d))
    sq = time(15, 15)
    rows = []
    for d, decision, sig, feats, day_idx in bt.iter_signals(idx, fut, days, first_only=False):
        r = sample_row(d, decision, sig, feats)
        r["label"] = label_signal(sig, day_idx, decision, sq)
        rows.append(r)
    return pd.DataFrame(rows)


def walk_forward(ds: pd.DataFrame, min_train: int = 150, refit_months: int = 3, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window walk-forward. Returns (oos predictions, metrics per model)."""
    ds = ds.sort_values("decision_ts").reset_index(drop=True)
    ds["month"] = pd.to_datetime(ds["date"]).dt.to_period("M")
    months = sorted(ds["month"].unique())
    X_all = ds[FEATURES].astype(float)
    preds = []
    i = 0
    while i < len(months):
        test_months = months[i:i + refit_months]
        train = ds[ds["month"] < test_months[0]]
        test = ds[ds["month"].isin(test_months)]
        i += refit_months
        if len(train) < min_train or test.empty or train["label"].nunique() < 2:
            continue
        Xtr = X_all.loc[train.index].fillna(X_all.loc[train.index].median())
        Xte = X_all.loc[test.index].fillna(X_all.loc[train.index].median())
        for name, m in make_models(seed).items():
            m.fit(Xtr, train["label"])
            p = m.predict_proba(Xte)[:, 1]
            preds.append(pd.DataFrame({"decision_ts": test["decision_ts"].values, "date": test["date"].values,
                                       "model": name, "prob": p, "label": test["label"].values,
                                       "train_size": len(train), "train_end": str(train["date"].max())}))
    P = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    rows = []
    for name, g in (P.groupby("model") if len(P) else []):
        base = g["label"].mean()
        top = g[g["prob"] >= g["prob"].quantile(0.7)]
        rows.append({"model": name, "oos_samples": len(g), "base_rate": round(base, 3),
                     "auc": round(roc_auc_score(g["label"], g["prob"]), 3) if g["label"].nunique() > 1 else None,
                     "brier": round(brier_score_loss(g["label"], g["prob"]), 4),
                     "brier_baseline": round(brier_score_loss(g["label"], np.full(len(g), base)), 4),
                     "hit_rate_top30pct": round(top["label"].mean(), 3) if len(top) else None})
    return P, pd.DataFrame(rows)


def fit_final(ds: pd.DataFrame, model_name: str, path: Path, meta: dict, seed: int = 0) -> Path:
    X = ds[FEATURES].astype(float)
    med = X.median()
    m = make_models(seed)[model_name]
    m.fit(X.fillna(med), ds["label"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": m, "medians": med.to_dict(), "features": FEATURES, "meta": meta}, f)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    return path


def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_one(bundle: dict, row: dict) -> float:
    x = pd.DataFrame([{k: row.get(k, np.nan) for k in bundle["features"]}]).astype(float).fillna(bundle["medians"])
    return float(bundle["model"].predict_proba(x)[:, 1][0])
