"""ML / DL for NIFTY options: (A) next-day volatility forecasts, (B) option premium pricing vs Black-Scholes.

A1  India VIX next day (yfinance 2008->)        target log(VIX_t+1 / VIX_t), baseline = no change
A2  NIFTY ATM implied vol next day (2021->)      target IV_t+1 - IV_t,        baseline = no change
    Decision at day t close. US/European markets are joined only from sessions that closed before t's close.
B   Option premium from 5-min real option candles (2021->): given spot NOW, strike, time to expiry and
    YESTERDAY's ATM IV / VIX, what is the premium? Black-Scholes(prior-day ATM IV) vs LightGBM vs MLP.
    Train < 2025-01-01, test 2025-26 (out of sample).

    python3 -m ml_lab.options_ml
"""
from __future__ import annotations

import glob
import json
import re
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import norm
from sklearn.linear_model import Ridge

from core.config import load_config, resolve_path
from ml_lab.models.deep import fit_predict

ROOT = Path(__file__).resolve().parent
R = 0.065
LAKE = resolve_path(load_config(), "data_lake")


def yahoo(name):
    return pd.read_parquet(LAKE / "silver" / "yahoo" / f"{name}.parquet").set_index("date")["close"]


def models_forecast(X, y, dates, years, deep_L=10):
    """Yearly expanding walk-forward. Returns DataFrame with predictions of each model on test years."""
    out = []
    X = np.nan_to_num(X.astype(np.float32))
    for Y in years:
        tr = np.where(dates.year < Y)[0]
        te = np.where(dates.year == Y)[0]
        if len(te) == 0 or len(tr) < 200:
            continue
        p = pd.DataFrame(index=te)
        p["ridge"] = Ridge(alpha=1.0).fit(X[tr], y[tr]).predict(X[te])
        p["lightgbm"] = LGBMRegressor(n_estimators=300, num_leaves=7, learning_rate=0.02, min_child_samples=30,
                                      subsample=0.8, subsample_freq=1, verbose=-1, random_state=0).fit(X[tr], y[tr]).predict(X[te])
        for m in ("mlp", "lstm"):
            p[m] = np.mean([fit_predict(m, X, y, tr, te, "reg", L=deep_L if m == "lstm" else 1, seed=s) for s in (0, 1, 2)], axis=0)
        out.append(p)
    return pd.concat(out)


def score(level_now, level_next, pred_change, to_level, dates):
    """RMSE of forecast level vs no-change, all test years and 2024+."""
    res = {}
    rw = level_now
    for m in pred_change.columns:
        f = to_level(level_now, pred_change[m].to_numpy())
        e, e0 = (f - level_next) ** 2, (rw - level_next) ** 2
        late = dates >= pd.Timestamp("2024-01-01")
        dir_ok = np.sign(f - level_now) == np.sign(level_next - level_now)
        res[m] = {"rmse": round(float(np.sqrt(e.mean())), 4), "no_change_rmse": round(float(np.sqrt(e0.mean())), 4),
                  "improvement_pct": round(float((1 - np.sqrt(e.mean() / e0.mean())) * 100), 2),
                  "improvement_2024plus_pct": round(float((1 - np.sqrt(e[late].mean() / e0[late].mean())) * 100), 2),
                  "direction_acc_pct": round(float(dir_ok.mean() * 100), 1)}
    return res


def ext_prev(series: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Value from the latest session strictly before each India date (US/EU close after India)."""
    s = series.dropna().sort_index()
    left = pd.DataFrame({"date": idx})
    right = pd.DataFrame({"date": s.index, "v": s.to_numpy()})
    return pd.merge_asof(left, right, on="date", allow_exact_matches=False)["v"].set_axis(idx)


def part_a1():
    vix, nifty = yahoo("INDIAVIX"), yahoo("NIFTY")
    d = pd.DataFrame({"vix": vix, "nifty": nifty}).dropna()
    d = d[d.vix > 0]
    lv = np.log(d.vix)
    f = pd.DataFrame(index=d.index)
    for k in (1, 2, 5, 10, 21):
        f[f"dlv_{k}"] = lv.diff(k)
    f["lv"] = lv
    f["lv_ma20"], f["lv_ma60"] = lv - lv.rolling(20).mean(), lv - lv.rolling(60).mean()
    r = np.log(d.nifty).diff()
    f["r1"], f["r5"] = r, r.rolling(5).sum()
    f["rv5"], f["rv21"] = r.rolling(5).std() * np.sqrt(252) * 100, r.rolling(21).std() * np.sqrt(252) * 100
    f["vix_minus_rv21"] = d.vix - f.rv21
    usv = ext_prev(yahoo("USVIX"), d.index)
    f["us_vix_chg"] = np.log(usv).diff()
    f["sp_r1"] = ext_prev(np.log(yahoo("SP500")).diff(), d.index)
    f["dow"] = d.index.dayofweek
    y = lv.shift(-1) - lv
    ok = f.notna().all(axis=1) & y.notna()
    f, y, vixs = f[ok], y[ok], d.vix[ok]
    nxt = d.vix.shift(-1)[ok]
    p = models_forecast(f.to_numpy(), y.to_numpy(np.float32), f.index, range(2013, 2027), deep_L=20)
    dates = f.index[p.index]
    return score(vixs.iloc[p.index].to_numpy(), nxt.iloc[p.index].to_numpy(), p,
                 lambda now, ch: now * np.exp(ch), dates)


def part_a2():
    iv = pd.read_parquet(LAKE / "gold" / "implied_volatility" / "atm_iv_daily.parquet")
    iv["date"] = pd.to_datetime(iv.date)
    iv = iv.set_index("date").sort_index()
    v = iv.iv_atm_close
    f = pd.DataFrame(index=iv.index)
    f["iv"] = v
    for k in (1, 2, 5):
        f[f"div_{k}"] = v.diff(k)
    f["iv_ma10"] = v - v.rolling(10).mean()
    f["iv_1000_vs_close"] = iv.iv_atm_1000 - v
    f["skew_pe_ce"] = iv.iv_pe_1000 - iv.iv_ce_1000
    f["dte"] = iv.days_to_expiry
    f["roll_next"] = (iv.expiry.shift(-1) != iv.expiry).astype(float)   # weekly expiry calendar is known in advance
    f["dte_next"] = iv.days_to_expiry.shift(-1)                           # ditto (calendar), not a price
    r = np.log(iv.spot_eod).diff()
    f["r1"], f["rv5"] = r, r.rolling(5).std() * np.sqrt(252) * 100
    vix = yahoo("INDIAVIX").reindex(iv.index)
    f["vix"], f["iv_minus_vix"] = vix, v - vix
    f["us_vix_chg"] = np.log(ext_prev(yahoo("USVIX"), iv.index)).diff()
    y = v.shift(-1) - v
    ok = f.notna().all(axis=1) & y.notna()
    f, y = f[ok], y[ok]
    p = models_forecast(f.to_numpy(), y.to_numpy(np.float32), f.index, range(2023, 2027), deep_L=10)
    dates = f.index[p.index]
    return score(v[ok].iloc[p.index].to_numpy(), v.shift(-1)[ok].iloc[p.index].to_numpy(), p,
                 lambda now, ch: now + ch, dates)


SYM = re.compile(r"symbol=NSE-NIFTY-(\d{2}[A-Za-z]{3}\d{2})-(\d+)-(CE|PE)")


def option_rows() -> pd.DataFrame:
    spot = pd.concat(pd.read_parquet(p, columns=["timestamp", "close"]) for p in
                     glob.glob(str(LAKE / "silver" / "nifty" / "symbol=NSE-NIFTY" / "interval=5minute" / "**" / "*.parquet"), recursive=True))
    spot = spot.drop_duplicates("timestamp").set_index("timestamp")["close"].rename("S")
    sample = {time(9, 30), time(11, 0), time(13, 0), time(14, 30)}
    rows = []
    for p in glob.glob(str(LAKE / "silver" / "options" / "symbol=*" / "interval=5minute" / "**" / "*.parquet"), recursive=True):
        m = SYM.search(p)
        if not m:
            continue
        exp, k, typ = datetime.strptime(m.group(1), "%d%b%y").date(), float(m.group(2)), m.group(3)
        d = pd.read_parquet(p, columns=["timestamp", "close", "volume"])
        d = d[d.timestamp.dt.time.isin(sample) & (d.volume > 0) & (d.close >= 1.0)]
        if d.empty:
            continue
        d = d.assign(K=k, typ=typ, expiry=pd.Timestamp(exp))
        rows.append(d)
    o = pd.concat(rows).join(spot, on="timestamp").dropna(subset=["S"])
    o["date"] = o.timestamp.dt.tz_localize(None).dt.normalize()
    exp_dt = (o.expiry + pd.Timedelta(hours=15, minutes=30)).dt.tz_localize("Asia/Kolkata")
    o["T"] = (exp_dt - o.timestamp).dt.total_seconds() / (365 * 86400)
    o = o[o["T"] > 1 / (365 * 24)]
    # prior-day information only
    iv = pd.read_parquet(LAKE / "gold" / "implied_volatility" / "atm_iv_daily.parquet")[["date", "iv_atm_close"]]
    iv["date"] = pd.to_datetime(iv.date)
    vix = yahoo("INDIAVIX").rename("vix").reset_index()
    o = o.sort_values("date")
    o = pd.merge_asof(o, iv.sort_values("date").rename(columns={"iv_atm_close": "iv_prev"}), on="date", allow_exact_matches=False)
    o = pd.merge_asof(o, vix.sort_values("date").rename(columns={"vix": "vix_prev"}), on="date", allow_exact_matches=False)
    return o.dropna(subset=["iv_prev", "vix_prev"])


def bs_vec(S, K, T, sig, call):
    v = sig * np.sqrt(T)
    d1 = (np.log(S / K) + (R + 0.5 * sig ** 2) * T) / v
    d2 = d1 - v
    c = S * norm.cdf(d1) - K * np.exp(-R * T) * norm.cdf(d2)
    return np.where(call, c, c - S + K * np.exp(-R * T))


def part_b():
    o = option_rows()
    call = (o.typ == "CE").to_numpy()
    o["bs_iv"] = bs_vec(o.S.to_numpy(), o.K.to_numpy(), o["T"].to_numpy(), o.iv_prev.to_numpy() / 100, call)
    o["bs_vix"] = bs_vec(o.S.to_numpy(), o.K.to_numpy(), o["T"].to_numpy(), o.vix_prev.to_numpy() / 100, call)
    o["m"] = np.log(o.K / o.S)
    o["is_call"] = call.astype(float)
    o["sqrtT"] = np.sqrt(o["T"])
    o["z"] = o.m / (o.iv_prev / 100 * o.sqrtT)
    o["intr"] = np.where(call, np.maximum(o.S - o.K, 0), np.maximum(o.K - o.S, 0)) / o.S
    o["hour"] = o.timestamp.dt.hour + o.timestamp.dt.minute / 60
    o["bs_rel"] = o.bs_iv / o.S
    feats = ["m", "is_call", "T", "sqrtT", "z", "intr", "iv_prev", "vix_prev", "hour", "bs_rel"]
    o["y"] = o.close / o.S * 100
    tr = o[o.date < "2025-01-01"].reset_index(drop=True)
    te = o[o.date >= "2025-01-01"].reset_index(drop=True)
    X = pd.concat([tr, te])[feats].to_numpy(np.float32)
    y = pd.concat([tr, te]).y.to_numpy(np.float32)
    tri, tei = np.arange(len(tr)), np.arange(len(tr), len(tr) + len(te))
    te["lightgbm"] = LGBMRegressor(n_estimators=600, num_leaves=63, learning_rate=0.05, min_child_samples=50,
                                   verbose=-1, random_state=0).fit(X[tri], y[tri]).predict(X[tei]) / 100 * te.S
    te["mlp"] = np.mean([fit_predict("mlp", X, y, tri, tei, "reg", L=1, seed=s, epochs=40, patience=5) for s in (0, 1)], axis=0) / 100 * te.S
    te["bs_prev_atm_iv"], te["bs_prev_vix"] = te.bs_iv, te.bs_vix
    res = {"rows_train": len(tr), "rows_test": len(te), "contracts_test": int(te[["K", "typ", "expiry"]].drop_duplicates().shape[0])}
    near = (te.m.abs() < 0.01).to_numpy()
    for mdl in ("bs_prev_vix", "bs_prev_atm_iv", "lightgbm", "mlp"):
        err = (te[mdl].clip(lower=0.05) - te.close).abs()
        ape = err / te.close
        res[mdl] = {"MAE_rs": round(float(err.mean()), 2), "median_abs_pct_err": round(float(ape.median() * 100), 1),
                    "near_ATM_MAE_rs": round(float(err[near].mean()), 2),
                    "near_ATM_median_pct_err": round(float(ape[near].median() * 100), 1)}
    return res


def main() -> int:
    res = {"A1_india_vix_next_day": part_a1()}
    print("A1 India VIX next day:\n", pd.DataFrame(res["A1_india_vix_next_day"]).T.to_string(), flush=True)
    res["A2_nifty_atm_iv_next_day"] = part_a2()
    print("A2 NIFTY ATM IV next day:\n", pd.DataFrame(res["A2_nifty_atm_iv_next_day"]).T.to_string(), flush=True)
    res["B_option_premium_pricing"] = part_b()
    b = res["B_option_premium_pricing"]
    print("B option premium (test 2025-26):", {k: b[k] for k in ("rows_train", "rows_test", "contracts_test")})
    print(pd.DataFrame({k: v for k, v in b.items() if isinstance(v, dict)}).T.to_string())
    out = ROOT / "results" / f"options_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(res, indent=1))
    print("results ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
