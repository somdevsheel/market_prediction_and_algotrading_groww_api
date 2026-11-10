"""ML / DL for NIFTY-100 stock prices (yfinance, adjusted, 2010->). Walk-forward by year (train < Y, test Y).

Targets (decision at day t's close, information <= close t):
  * next-day:   r1  = close_{t+1}/close_t - 1   -> price forecast close_{t+1}; benchmark = random walk (close_t)
  * next-month: r21 = open_{t+22}/open_{t+1} - 1 (enter next open, hold ~1 month) -> monthly top-10 portfolio
Features are cross-sectionally ranked per date (relative strength). Survivorship: current constituents only.

    python3 -m ml_lab.stocks_ml
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from torch import nn

from core.config import load_config, resolve_path
from ml_lab.models.deep import DEV

ROOT = Path(__file__).resolve().parent
COST_RT = 0.0030      # delivery round trip: STT 0.1% x2 + stamp/exchange/GST + ~5 bps slippage per side (approx)


def panel() -> tuple[pd.DataFrame, list[str]]:
    lake = resolve_path(load_config(), "data_lake")
    d = pd.read_parquet(lake / "silver" / "yahoo_stocks" / "nifty100_daily.parquet")
    P = {k: d.pivot(index="date", columns="sym", values=k) for k in ("open", "high", "low", "close", "volume")}
    c, o, v = P["close"], P["open"], P["volume"]
    nifty = pd.read_parquet(lake / "silver" / "yahoo" / "NIFTY.parquet").set_index("date")["close"].reindex(c.index).ffill()
    F = {}
    for k in (1, 5, 21, 63, 126, 252):
        F[f"ret_{k}"] = c.pct_change(k, fill_method=None)
    F["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    r = c.pct_change(fill_method=None)
    F["vol_21"], F["vol_63"] = r.rolling(21).std(), r.rolling(63).std()
    F["vlm_ratio"] = v.rolling(5).mean() / v.rolling(63).mean()
    dlt = c.diff()
    up, dn = dlt.clip(lower=0).ewm(alpha=1 / 14).mean(), (-dlt.clip(upper=0)).ewm(alpha=1 / 14).mean()
    F["rsi14"] = 100 - 100 / (1 + up / dn)
    for m in (20, 50, 200):
        F[f"dist_ma{m}"] = c / c.rolling(m).mean() - 1
    F["dist_52wh"] = c / c.rolling(252).max() - 1
    F["range_5"] = ((P["high"] - P["low"]) / c).rolling(5).mean()
    feats = list(F)
    ranked = {k: f.rank(axis=1, pct=True) - 0.5 for k, f in F.items()}   # cross-sectional ranks per date
    long = pd.concat({k: x.stack(future_stack=True) for k, x in ranked.items()}, axis=1)
    long.index.names = ["date", "sym"]
    mk = pd.DataFrame({"mkt_ret_5": nifty.pct_change(5), "mkt_ret_21": nifty.pct_change(21),
                       "mkt_vol_21": nifty.pct_change().rolling(21).std(), "mkt_dist_ma200": nifty / nifty.rolling(200).mean() - 1})
    long = long.join(mk, on="date")
    feats += list(mk.columns)
    long["close"] = c.stack(future_stack=True)
    long["r1"] = (c.shift(-1) / c - 1).stack(future_stack=True)
    long["r21"] = (o.shift(-22) / o.shift(-1) - 1).stack(future_stack=True)
    long["close_next"] = c.shift(-1).stack(future_stack=True)
    long = long.reset_index().dropna(subset=feats + ["close"])
    return long, feats


class MLP(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.n = nn.Sequential(nn.Linear(f, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.n(x).squeeze(-1)


def fit_mlp(Xtr, ytr, Xte, seed=0, epochs=30):
    torch.manual_seed(seed)
    ysd = ytr.std() + 1e-9
    cut = int(len(Xtr) * 0.85)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)  # noqa: E731
    Xf, yf, Xv, yv = t(Xtr[:cut]), t(ytr[:cut] / ysd), t(Xtr[cut:]), t(ytr[cut:] / ysd)
    m = MLP(Xtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    best, state, bad = 9e9, None, 0
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(Xf), device=DEV)
        for b in range(0, len(Xf), 1024):
            j = perm[b:b + 1024]
            opt.zero_grad()
            nn.functional.mse_loss(m(Xf[j]), yf[j]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(m(Xv), yv).item()
        if vl < best - 1e-6:
            best, state, bad = vl, {k: x.clone() for k, x in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 4:
                break
    m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        return m(t(Xte)).cpu().numpy() * ysd


def walk_forward(L: pd.DataFrame, feats: list[str], target: str, years=range(2015, 2027)) -> pd.DataFrame:
    out = []
    yr = L.date.dt.year
    for Y in years:
        # train only on rows whose target window ended before the test year starts (no overlap leakage)
        horizon_days = 1 if target == "r1" else 32
        tr = L[(L.date < pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=horizon_days * 1.5)) & L[target].notna()]
        te = L[(yr == Y) & L[target].notna()].copy()
        if te.empty or len(tr) < 5000:
            continue
        Xtr, ytr, Xte = tr[feats].to_numpy(np.float32), tr[target].to_numpy(np.float32), te[feats].to_numpy(np.float32)
        ytr_c = np.clip(ytr, *np.quantile(ytr, [0.005, 0.995]))
        te["ridge"] = Ridge(alpha=10.0).fit(Xtr, ytr_c).predict(Xte)
        te["lightgbm"] = LGBMRegressor(n_estimators=300, num_leaves=15, learning_rate=0.03, min_child_samples=200,
                                       subsample=0.7, subsample_freq=1, colsample_bytree=0.7, verbose=-1,
                                       random_state=0).fit(Xtr, ytr_c).predict(Xte)
        te["mlp"] = np.mean([fit_mlp(Xtr, ytr_c, Xte, s) for s in (0, 1, 2)], axis=0)
        out.append(te)
        print(f"  {target} {Y}: train {len(tr):,} rows, test {len(te):,}", flush=True)
    return pd.concat(out, ignore_index=True)


def ic(p: pd.DataFrame, m: str, target: str) -> pd.Series:
    return p.groupby("date").apply(lambda g: g[m].rank().corr(g[target].rank()), include_groups=False)


def main() -> int:
    L, feats = panel()
    print(f"panel {len(L):,} rows, {L.sym.nunique()} stocks, {L.date.min().date()} -> {L.date.max().date()}, device {DEV}")
    models = ["ridge", "lightgbm", "mlp"]
    res = {"daily": {}, "monthly": {}}
    # ---------- next-day
    pd_ = walk_forward(L, feats, "r1")
    rw_err = np.abs(pd_.close_next - pd_.close) / pd_.close
    for m in models:
        price_hat = pd_.close * (1 + pd_[m])
        err = np.abs(pd_.close_next - price_hat) / pd_.close
        s = ic(pd_, m, "r1")
        top = pd_[pd_.groupby("date")[m].rank(ascending=False) <= 10].groupby("date").r1.mean()
        allm = pd_.groupby("date").r1.mean()
        res["daily"][m] = {"price_mape_pct": round(float(err.mean() * 100), 4), "random_walk_mape_pct": round(float(rw_err.mean() * 100), 4),
                           "beats_random_walk": bool(err.mean() < rw_err.mean()),
                           "direction_acc_pct": round(float(((pd_[m] > 0) == (pd_.r1 > 0)).mean() * 100), 2),
                           "mean_daily_IC": round(float(s.mean()), 4), "IC_2024plus": round(float(s[s.index >= "2024-01-01"].mean()), 4),
                           "top10_daily_gross_ann_pct": round(float((top - allm).mean() * 252 * 100), 1),
                           "top10_daily_net_ann_pct_vs_all": round(float((top - allm - COST_RT).mean() * 252 * 100), 1)}
    # ---------- next-month (month-end rebalance)
    pm = walk_forward(L, feats, "r21")

    me = pm.groupby(pm.date.dt.to_period("M")).date.transform("max") == pm.date
    pm = pm[me]
    for m in models:
        s = ic(pm, m, "r21")
        top = pm[pm.groupby("date")[m].rank(ascending=False) <= 10].groupby("date").r21.mean()
        allm = pm.groupby("date").r21.mean()
        net = top - COST_RT * 0.6   # ~60% of top-10 names change each month (assumption, documented)
        def cagr(x):
            eq = (1 + x).cumprod()
            return round(float((eq.iloc[-1] ** (12 / len(x)) - 1) * 100), 1) if len(x) else None
        res["monthly"][m] = {"mean_monthly_IC": round(float(s.mean()), 4), "IC_2024plus": round(float(s[s.index >= "2024-01-01"].mean()), 4),
                             "top10_cagr_net_pct": cagr(net), "all_stocks_cagr_pct": cagr(allm),
                             "top10_cagr_net_2024plus": cagr(net[net.index >= "2024-01-01"]),
                             "all_cagr_2024plus": cagr(allm[allm.index >= "2024-01-01"]),
                             "months_top10_beats_all_pct": round(float((net > allm).mean() * 100), 1)}
    out = ROOT / "results" / f"stocks_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(res, indent=1))
    pd.set_option("display.width", 220)
    print("\nNEXT-DAY (price forecast vs random walk, direction, ranking):")
    print(pd.DataFrame(res["daily"]).T.to_string())
    print("\nNEXT-MONTH (top-10 monthly portfolio vs holding all 98):")
    print(pd.DataFrame(res["monthly"]).T.to_string())
    print("results ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
