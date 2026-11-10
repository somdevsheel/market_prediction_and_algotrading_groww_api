"""Price forecasts for every liquid stock: next trading day and ~1 month (21 trading days) ahead.

A single exact price is not predictable (no model beat "tomorrow = today" in our tests), so this predicts a
RANGE with LightGBM quantile regression: low (10th pct), middle (50th pct = median) and high (90th pct) of the
future close.  If the model is honest, the actual price lands inside low..high about 80 % of the time.

Data / cleaning / universe: same as ml_lab/stocks_bhs.py (Yahoo, from 2009, price >= Rs 20, >= Rs 2 cr/day).
Training rows: every 5th trading day (2010 ->).
Calibration: raw quantile ranges came out too narrow (73 % / 70 % coverage instead of 80 %), so they are widened by
conformalized quantile regression: models are fitted on data before a calibration year, and the widening needed
for 80 % coverage in that year is added to every forecast.
Evaluation: fit < 2023, calibrate on 2023, test 2024 -> today (unseen).
Final model: fit on everything up to a year ago, calibrate on the last year, predict from the latest close.

    python3 -m ml_lab.stock_price_forecast
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from ml_lab.stocks_bhs import MIN_PX, MIN_TV, ROOT, load_panels

HORIZONS = {"d1": 1, "m1": 21}
QUANTILES = {"lo": 0.10, "mid": 0.50, "hi": 0.90}
TEST_START = "2024-01-01"
CAL_DAYS = 365          # conformal calibration window (the year before the test / before today)
RAW = ["vol_21", "vol_63", "range_5", "mkt_vol_21", "mkt_ret_21", "mkt_dist_ma200", "ret_1", "ret_5"]   # kept as levels


def panel(P, jump, nifty):
    c, v, h, lo = P["close"], P["volume"], P["high"], P["low"]
    idx = c.index
    cf = c.ffill(limit=5)
    r = cf.pct_change(fill_method=None)
    dates = list(idx[252::5])
    if idx[-1] not in dates:
        dates.append(idx[-1])
    nc = nifty["close"].reindex(idx).ffill()
    mr = nc.pct_change()

    def at(df):
        return df.loc[dates]

    F = {"ret_1": at(r), "ret_5": at(cf.pct_change(5, fill_method=None)), "ret_21": at(cf.pct_change(21, fill_method=None)),
         "ret_63": at(cf.pct_change(63, fill_method=None)), "ret_252": at(cf.pct_change(252, fill_method=None)),
         "vol_21": at(r.rolling(21).std()), "vol_63": at(r.rolling(63).std()),
         "range_5": at(((h - lo) / cf).rolling(5).mean()),
         "dist_ma20": at(cf / cf.rolling(20).mean() - 1), "dist_ma200": at(cf / cf.rolling(200).mean() - 1),
         "dist_52wh": at(cf / cf.rolling(252).max() - 1),
         "vlm_ratio": at(v.rolling(5).mean() / v.rolling(63).mean())}
    tv = (cf * v).rolling(126, min_periods=100).median()
    elig = at((c.notna().rolling(252).sum() >= 240) & (cf >= MIN_PX) & (tv >= MIN_TV)) & ~at(jump.astype("float32").rolling(21).max() > 0)
    tgt = {k: at(cf.shift(-n) / cf - 1) for k, n in HORIZONS.items()}
    tgt_date = {k: pd.Series(idx, index=idx).shift(-n).reindex(dates) for k, n in HORIZONS.items()}
    close = at(cf)
    rows = []
    for dt in dates:
        u = elig.columns[elig.loc[dt].to_numpy()]
        if len(u) < 50:
            continue
        x = pd.DataFrame({k: f.loc[dt, u].to_numpy() for k, f in F.items()})
        x["sym"], x["date"], x["close"] = u, dt, close.loc[dt, u].to_numpy()
        for k in HORIZONS:
            x[f"y_{k}"] = tgt[k].loc[dt, u].to_numpy()
            x[f"t_{k}"] = tgt_date[k].loc[dt]
        rows.append(x)
    L = pd.concat(rows, ignore_index=True)
    feats = list(F)
    for f in feats:
        if f not in RAW:
            L[f] = L.groupby("date")[f].rank(pct=True) - 0.5
    mk = pd.DataFrame({"mkt_ret_21": nc.pct_change(21), "mkt_vol_21": mr.rolling(21).std() * np.sqrt(252),
                       "mkt_dist_ma200": nc / nc.rolling(200).mean() - 1})
    L = L.join(mk, on="date")
    feats += list(mk.columns)
    L[feats] = L[feats].astype("float32")
    return L, feats


def fit_q(X, y, q):
    return LGBMRegressor(objective="quantile", alpha=q, n_estimators=300, num_leaves=31, learning_rate=0.05,
                         min_child_samples=500, subsample=0.7, subsample_freq=1, colsample_bytree=0.8,
                         verbose=-1, random_state=0).fit(X, y)


def fit_calibrated(L, feats, k, cal_start: pd.Timestamp, cal_end: pd.Timestamp):
    """Quantile models fitted on outcomes known before cal_start; conformal widening from [cal_start, cal_end)."""
    y = f"y_{k}"
    tr = L[(L[f"t_{k}"] < cal_start) & L[y].notna()]
    ca = L[(L.date >= cal_start) & (L[f"t_{k}"] < cal_end) & L[y].notna()]
    clip = tr[y].quantile([0.001, 0.999]).to_numpy()
    models = {q: fit_q(tr[feats], tr[y].clip(*clip), a) for q, a in QUANTILES.items()}
    lo, hi = models["lo"].predict(ca[feats]), models["hi"].predict(ca[feats])
    score = np.maximum(lo - ca[y].to_numpy(), ca[y].to_numpy() - hi)          # CQR conformity score
    target = QUANTILES["hi"] - QUANTILES["lo"]
    widen = float(np.quantile(score, min(1.0, target * (1 + 1 / len(score)))))
    return models, widen


def predict_calibrated(models, widen, X):
    p = {q: m.predict(X) for q, m in models.items()}
    p["lo"], p["hi"] = p["lo"] - widen, p["hi"] + widen
    return p


def evaluate(L, feats) -> dict:
    out = {}
    test = pd.Timestamp(TEST_START)
    cal = test - pd.Timedelta(days=CAL_DAYS)
    for k, n in HORIZONS.items():
        y = f"y_{k}"
        models, widen = fit_calibrated(L, feats, k, cal, test)
        te = L[(L.date >= test) & L[y].notna()]
        pred = predict_calibrated(models, widen, te[feats])
        actual = te[y].to_numpy()
        mid_err = np.abs(pred["mid"] - actual)
        rw_err = np.abs(actual)                                    # "price stays the same" forecast
        inside = (actual >= pred["lo"]) & (actual <= pred["hi"])
        direction = np.sign(pred["mid"]) == np.sign(actual)
        out[k] = {"horizon_days": n, "test_rows": int(len(te)), "test_from": TEST_START,
                  "avg_error_pct_model": round(float(mid_err.mean() * 100), 3),
                  "avg_error_pct_same_price": round(float(rw_err.mean() * 100), 3),
                  "beats_same_price": bool(mid_err.mean() < rw_err.mean()),
                  "range_contains_actual_pct": round(float(inside.mean() * 100), 1),
                  "avg_range_width_pct": round(float((pred["hi"] - pred["lo"]).mean() * 100), 2),
                  "direction_correct_pct": round(float(direction[np.abs(pred["mid"]) > 1e-4].mean() * 100), 1),
                  "below_low_pct": round(float((actual < pred["lo"]).mean() * 100), 1),
                  "above_high_pct": round(float((actual > pred["hi"]).mean() * 100), 1),
                  "calibration_widening_pct": round(widen * 100, 2)}
        print(f"  {k}: {out[k]}", flush=True)
    return out


def forecast_latest(L, feats) -> pd.DataFrame:
    last = L.date.max()
    cur = L[L.date == last].copy()
    res = cur[["sym", "close"]].copy()
    res["as_of"] = str(last.date())
    end = L.date.max() + pd.Timedelta(days=1)
    for k, n in HORIZONS.items():
        models, widen = fit_calibrated(L, feats, k, end - pd.Timedelta(days=CAL_DAYS), end)
        p = predict_calibrated(models, widen, cur[feats])
        for q in QUANTILES:
            res[f"{k}_{q}_ret_pct"] = (p[q] * 100).round(2)
            res[f"{k}_{q}"] = (cur.close.to_numpy() * (1 + p[q])).round(2)
        res[f"{k}_days"] = n
    return res.sort_values("sym")


def main() -> int:
    t0 = datetime.now()
    P, jump, nifty = load_panels()
    L, feats = panel(P, jump, nifty)
    print(f"{len(L):,} stock-dates, {L.sym.nunique()} stocks, latest {L.date.max().date()} "
          f"({(L.date == L.date.max()).sum()} stocks)", flush=True)
    out = ROOT / "results" / f"forecast_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    print("Accuracy on unseen data (2024 -> today):", flush=True)
    metrics = evaluate(L, feats)
    fc = forecast_latest(L, feats)
    fc.to_csv(out / "forecast.csv", index=False)
    (out / "metrics.json").write_text(json.dumps({"as_of": str(L.date.max().date()), "quantiles": QUANTILES,
                                                  "accuracy": metrics, "features": feats}, indent=1))
    print(fc.head(8).to_string(index=False))
    print("results ->", out, f"({(datetime.now() - t0).seconds // 60} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
