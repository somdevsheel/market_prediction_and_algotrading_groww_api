"""Expanding-window walk-forward: for each test year Y, train on all years < Y, predict Y. No shuffling."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from ml_lab.models import classic, deep
from ml_lab.models.classic import HAR

DEEP = ["mlp", "lstm", "gru", "transformer"]


def run(d: pd.DataFrame, task: str, first_test_year: int = 2013, seeds=(0, 1, 2), log=print) -> pd.DataFrame:
    feats = d.attrs["features"]
    X = d[feats].to_numpy(float)
    y = d["y_dir"].to_numpy(float) if task == "clf" else np.log(d["y_vol"].to_numpy(float))  # log-range
    years = d.date.dt.year.to_numpy()
    preds = []
    for Y in range(first_test_year, years.max() + 1):
        tr, te = np.where(years < Y)[0], np.where(years == Y)[0]
        if len(te) == 0:
            continue
        out = pd.DataFrame({"date": d.date.iloc[te].to_numpy(), "y": y[te], "y_ret": d.y_ret.iloc[te].to_numpy()})
        if task == "clf":
            out["base_rate"] = y[tr].mean()
            for name, m in classic.classifiers().items():
                m.fit(X[tr], y[tr])
                out[name] = m.predict_proba(X[te])[:, 1]
        else:
            har_i = [feats.index(c) for c in HAR]
            for name, (cols, m) in classic.regressors().items():
                Xi = np.log(X[:, har_i]) if cols == "har" else X
                m.fit(Xi[tr], y[tr])
                out[name] = m.predict(Xi[te])
        for name in DEEP:  # average of seeds = small ensemble (reduces luck of initialisation)
            out[name] = np.mean([deep.fit_predict(name, X, y, tr, te, task, seed=s) for s in seeds], axis=0)
        preds.append(out)
        log(f"  {task} {Y}: trained on {len(tr)} days, tested on {len(te)}")
    return pd.concat(preds, ignore_index=True)


def clf_report(p: pd.DataFrame, models: list[str], cost_bps: float = 6.0, band: float = 0.05) -> pd.DataFrame:
    rows = []
    for m in models + ["always_up"]:
        prob = p[m].to_numpy() if m != "always_up" else np.full(len(p), 0.51)
        side = np.where(prob > 0.5 + band, 1, np.where(prob < 0.5 - band, -1, 0)) if m != "always_up" else np.ones(len(p))
        pnl = side * p.y_ret.to_numpy() - (side != 0) * cost_bps / 1e4
        yrs = p.date.dt.year
        r = {"model": m, "accuracy": round(accuracy_score(p.y, prob > 0.5) * 100, 2),
             "auc": round(roc_auc_score(p.y, prob), 4) if m != "always_up" else 0.5,
             "logloss": round(log_loss(p.y, np.clip(prob, 1e-6, 1 - 1e-6)), 4) if m != "always_up" else None,
             "trade_days": int((side != 0).sum()), "strategy_return_pct": round(float(pnl.sum() * 100), 1),
             "years_positive": int((pd.Series(pnl).groupby(yrs.to_numpy()).sum() > 0).sum()), "years": int(yrs.nunique())}
        rows.append(r)
    return pd.DataFrame(rows)


def reg_report(p: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    y = p.y.to_numpy()
    rows = []
    for m in models:
        f = p[m].to_numpy()
        var_t, var_f = np.exp(2 * y), np.exp(2 * f)  # squared range ~ variance proxy
        rows.append({"model": m, "rmse_log_range": round(float(np.sqrt(np.mean((f - y) ** 2))), 4),
                     "r2": round(float(1 - np.sum((f - y) ** 2) / np.sum((y - y.mean()) ** 2)), 4),
                     "qlike": round(float(np.mean(var_t / var_f - np.log(var_t / var_f) - 1)), 4),
                     "corr": round(float(np.corrcoef(f, y)[0, 1]), 4)})
    return pd.DataFrame(rows).sort_values("qlike")
