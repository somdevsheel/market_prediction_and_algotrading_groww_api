"""Do Indian company fundamentals make the BUY signal stronger?  Same XGBoost, same months, with vs without.

Fundamentals: data_lake/gold/india_fundamentals/fund_panel.parquet (python3 -m ml_lab.data.india_fundamentals),
point-in-time: a quarter's numbers are used only from the day AFTER its filing date, and only for 200 days.
The filings end in Mar 2025, so the test covers decision months 2019-2024 only (training still starts in 2009;
months before 2018 simply have no fundamentals = neutral 0).

Features added (cross-sectional percentile ranks per month, like the price features):
  growth   rev_yoy, np_yoy, sue (earnings surprise), margin, margin_chg
  value    ep (TTM profit / market cap), sp (TTM sales / mcap), bp (book equity / mcap)
  quality  roe, de (debt / equity), cash_mcap, loss (TTM loss), days_since_result, has_fund
Market cap = Yahoo (split-adjusted) close x split factor after that date x shares in the filing.

    python3 -m ml_lab.fundamentals_test        (~15-25 min on CPU)
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ml_lab.stocks_bhs import CLASSIC, LAKE, ROOT, backtest, build, load_panels, perf

PANEL = LAKE / "gold" / "india_fundamentals" / "fund_panel.parquet"
TEST_YEARS = range(2019, int(os.environ.get("FUND_TEST_END", 2024)) + 1)   # FUND_TEST_END=2026 adds the new filings
MAX_AGE_DAYS = 200
SEEDS = (0, 1)
FUND = ["f_rev_yoy", "f_np_yoy", "f_sue", "f_margin", "f_margin_chg", "f_ep", "f_sp", "f_bp", "f_roe", "f_de",
        "f_cash_mcap", "f_loss", "f_days_since_result"]


def add_fundamentals(L: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_parquet(PANEL).rename(columns={"symbol": "sym"}).dropna(subset=["avail"]).sort_values("avail")
    x = L[["sym", "date"]].reset_index().sort_values("date")
    m = pd.merge_asof(x, p, left_on="date", right_on="avail", by="sym", direction="backward")
    m = m.set_index("index").sort_index()
    age = (m.date - m.avail).dt.days
    fresh = age <= MAX_AGE_DAYS
    cf = close.ffill(limit=5)
    px = cf.stack().rename("px")
    px.index.names = ["date", "sym"]
    adj_px = pd.MultiIndex.from_frame(m[["date", "sym"]])
    m["px"] = px.reindex(adj_px).to_numpy()
    mcap = m.px * m.split_after * m.shares
    mcap = mcap.where(mcap > 1e7)                                   # < Rs 1 crore = broken share count
    eq = m.equity.where(m.equity > 0)
    raw = pd.DataFrame({
        "f_rev_yoy": m.rev_yoy.clip(-1, 5), "f_np_yoy": m.np_yoy.clip(-5, 5), "f_sue": m.sue.clip(-10, 10),
        "f_margin": m.margin.clip(-2, 1), "f_margin_chg": m.margin_chg.clip(-1, 1),
        "f_ep": (m.np_ttm / mcap).clip(-1, 1), "f_sp": (m.rev_ttm / mcap).clip(0, 50), "f_bp": (m.equity / mcap).clip(-5, 20),
        "f_roe": (m.np_ttm / eq).clip(-2, 2), "f_de": (m.debt / eq).clip(0, 20), "f_cash_mcap": (m.cash / mcap).clip(0, 5),
        "f_loss": (m.np_ttm < 0).astype(float).where(m.np_ttm.notna()), "f_days_since_result": age.astype(float),
    }, index=m.index)
    raw = raw.where(fresh, axis=0)
    out = L.copy()
    for c in FUND:
        out[f"raw_{c}"] = raw[c]
        out[c] = (raw[c].groupby(L.date).rank(pct=True) - 0.5).fillna(0.0)
    out["f_has_fund"] = fresh.astype(float).to_numpy()
    return out


def fit_xgb(L, feats, tr, te):
    Xtr, ytr, Xte = L.loc[tr, feats].to_numpy(np.float32), L.loc[tr, "y"].to_numpy(), L.loc[te, feats].to_numpy(np.float32)
    ps, imp = [], np.zeros(len(feats))
    for s in SEEDS:
        m = CLASSIC["xgboost"]().set_params(random_state=s)
        m.fit(Xtr, ytr)
        ps.append(m.predict_proba(Xte))
        imp += m.feature_importances_
    return np.mean(ps, 0), imp / len(SEEDS)


def metrics(T: pd.DataFrame, opens: pd.DataFrame) -> dict:
    k = T.fwd.notna()
    ics = T[k].groupby("date").apply(lambda g: spearmanr(g.score, g.fwd).statistic)
    dec = T[k].groupby("date").apply(lambda g: g.fwd[g.score >= g.score.quantile(0.9)].mean() - g.fwd[g.score <= g.score.quantile(0.1)].mean())
    top10 = T[k].groupby("date").apply(lambda g: g.nlargest(10, "score").fwd.mean())
    ew = T[k].groupby("date").fwd.mean()
    eq, closed, ch = backtest(T, opens)
    s = perf(eq.equity)
    buy = T[k & (T.signal == 2)]
    return {"IC_mean": round(ics.mean(), 4), "IC_t": round(ics.mean() / ics.std() * np.sqrt(len(ics)), 2),
            "IC_positive_months_pct": round((ics > 0).mean() * 100), "top_minus_bottom_decile_pct_month": round(dec.mean() * 100, 2),
            "top10_avg_month_pct": round(top10.mean() * 100, 2), "top10_minus_all_pct_month": round((top10 - ew).mean() * 100, 2),
            "BUY_count": int(len(buy)), "BUY_avg_month_pct": round(buy.fwd.mean() * 100, 2), "BUY_precision_pct": round((buy.y == 2).mean() * 100, 1),
            "portfolio_cagr_net_pct": s["cagr_pct"], "portfolio_max_dd_pct": s["max_dd_pct"], "portfolio_sharpe": s["sharpe"],
            "charges_rs": round(ch)}, ics, eq.equity


def main() -> int:
    t0 = datetime.now()
    P, jump, nifty = load_panels()
    L, feats, seq = build(P, jump, nifty)
    del seq
    L = add_fundamentals(L, P["close"])
    feats_f = feats + FUND + ["f_has_fund"]
    tm = L.date.dt.year.isin(TEST_YEARS)
    print(f"{len(L):,} stock-months; test months 2019-2024: {tm.sum():,} rows, fundamentals available for "
          f"{L.loc[tm, 'f_has_fund'].mean() * 100:.0f}% ({(datetime.now() - t0).seconds}s)", flush=True)
    out = ROOT / "results" / f"fund_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    # 1. each fundamental on its own: monthly rank IC vs next-month return (test years, raw values)
    k = tm & L.fwd.notna()
    single = {}
    for c in FUND:
        ic = L[k].groupby("date").apply(lambda g: spearmanr(g[f"raw_{c}"], g.fwd, nan_policy="omit").statistic
                                        if g[f"raw_{c}"].notna().sum() > 30 else np.nan).dropna()
        single[c] = {"IC_mean": round(ic.mean(), 4), "IC_t": round(ic.mean() / ic.std() * np.sqrt(len(ic)), 2), "months": len(ic)}
    print("\nSingle fundamental factors, 2019-2024 (rank IC with next month's return; |t| > 2 = real):")
    print(pd.DataFrame(single).T.sort_values("IC_t").to_string(), flush=True)

    # 2. XGBoost with vs without, yearly walk-forward
    res, ics, curves, imps, preds = {}, {}, {}, {}, []
    opens = P["open"].ffill(limit=10)
    variants = (("price_only", feats),) if os.environ.get("FUND_PRICE_ONLY") else (("price_only", feats), ("price+fundamentals", feats_f))
    for name, fs in variants:
        parts, imp = [], np.zeros(len(fs))
        for Y in TEST_YEARS:
            tr = np.where((L.y >= 0) & L.exit.notna() & (L.exit < pd.Timestamp(f"{Y}-01-01")))[0]
            te = np.where(L.date.dt.year == Y)[0]
            p, im = fit_xgb(L, fs, tr, te)
            imp += im
            df = L.loc[te, ["date", "sym", "entry", "exit", "tv", "fwd", "y"]].copy()
            df["p0"], df["p1"], df["p2"] = p[:, 0], p[:, 1], p[:, 2]
            df["signal"], df["score"] = p.argmax(1), p[:, 2] - p[:, 0]
            parts.append(df)
            print(f"  {name} {Y}: trained {len(tr):,} ({(datetime.now() - t0).seconds // 60} min)", flush=True)
        T = pd.concat(parts, ignore_index=True)
        res[name], ics[name], curves[name] = metrics(T, opens)
        imps[name] = pd.Series(imp / len(TEST_YEARS), index=fs)
        preds.append(T.assign(variant=name))
    yr = {n: ic.groupby(ic.index.year).mean().round(3) for n, ic in ics.items()}
    eqy = {n: pd.concat([pd.Series([1e5], index=[2018]), c.groupby(c.index.year).last()]).pct_change().iloc[1:].mul(100).round(1)
           for n, c in curves.items()}
    tab = pd.DataFrame(res).T
    pd.set_option("display.width", 250)
    print("\nXGBoost, out-of-sample 2019-2024 (same months, same stocks):")
    print(tab.T.to_string())
    print("\nIC by year:"); print(pd.DataFrame(yr).to_string())
    print("\nTop-10 portfolio return by year % (net of charges):"); print(pd.DataFrame(eqy).to_string())
    fi = imps[list(imps)[-1]].sort_values(ascending=False)
    print("\nFeature importance (price+fundamentals model), top 20:"); print(fi.head(20).round(4).to_string())
    tab.to_csv(out / "summary.csv")
    pd.DataFrame(single).T.to_csv(out / "single_factor_ic.csv")
    pd.DataFrame(yr).to_csv(out / "ic_by_year.csv")
    pd.DataFrame(eqy).to_csv(out / "yearly_return.csv")
    fi.to_csv(out / "importance.csv")
    pd.concat(preds).to_parquet(out / "predictions.parquet")
    (out / "report.json").write_text(json.dumps({"summary": res, "single": single, "test_years": list(TEST_YEARS)}, indent=1, default=str))
    print(f"\n-> {out}  ({(datetime.now() - t0).seconds // 60} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
