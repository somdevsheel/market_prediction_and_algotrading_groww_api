"""Stock trading with ML / DL: monthly top-10 portfolio from the NIFTY-500 list (yfinance, adjusted, 2010->).

Decision at each month-end close; trade at the next day's open; hold to the next month-end's next open.
Universe (point in time): the 100 most liquid stocks by 6-month median traded value on that date, with >= 1 year
of history and price > Rs 20. Stocks that later fell out of the top 100 are still in the sample. Residual
survivorship bias: companies that are no longer in today's NIFTY 500 are missing.

Models (yearly walk-forward, trained only on months whose holding period ended before the test year):
  baseline : momentum (12-1 month rule, no ML)
  classic  : ridge, lasso, elastic net, logistic regression, KNN, linear SVR, random forest, extra trees,
             hist-gradient-boosting, XGBoost, LightGBM, CatBoost
  deep     : MLP, LSTM, GRU, 1-D CNN, Transformer (sequence nets read the last 60 days + snapshot features)
  ensemble : average rank of every ML/DL model
Model choice: rank models on 2014-2021 only; 2022-2026 is a hold-out that is reported, never used to choose.
Portfolio: Rs 1,00,000, 10 stocks, equal rupees, whole shares, keep a holding while it ranks in the top 20,
Groww delivery charges + 0.1 % slippage per side.

    python3 -m ml_lab.stocks_ml
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
from torch import nn

from core.config import load_config, resolve_path
from ml_lab.models.deep import DEV

ROOT = Path(__file__).resolve().parent
LAKE = resolve_path(load_config(), "data_lake")
UNIVERSE, TOP_N, KEEP_RANK, SEQ_L = 100, 10, 20, 60
CAPITAL = 100_000.0
SLIPPAGE = 0.001


def order_cost(value: float, side: str) -> float:
    """Approximate Groww equity-delivery charges for one order (verify against the current Groww schedule)."""
    brokerage = max(5.0, min(20.0, 0.001 * value))
    stt = 0.001 * value
    exch, sebi = 0.0000297 * value, 0.000001 * value
    stamp = 0.00015 * value if side == "buy" else 0.0
    dp = 15.93 if side == "sell" else 0.0          # depository charge per scrip per sell day, incl. GST
    gst = 0.18 * (brokerage + exch + sebi)
    return brokerage + stt + exch + sebi + stamp + dp + gst + SLIPPAGE * value


# ---------------------------------------------------------------- data
def load():
    d = pd.read_parquet(LAKE / "silver" / "yahoo_stocks" / "nifty500_daily.parquet")
    P = {k: d.pivot(index="date", columns="sym", values=k).sort_index() for k in ("open", "high", "low", "close", "volume")}
    nifty = pd.read_parquet(LAKE / "silver" / "yahoo" / "NIFTY.parquet").set_index("date")
    return P, nifty


def build(P, nifty):
    c, o, v, h, lo = P["close"], P["open"], P["volume"], P["high"], P["low"]
    idx = c.index
    nc = nifty["close"].reindex(idx).ffill()
    r = c.pct_change(fill_method=None)
    F = {}
    for k in (5, 21, 63, 126, 252):
        F[f"ret_{k}"] = c.pct_change(k, fill_method=None)
    F["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    F["vol_21"], F["vol_63"] = r.rolling(21).std(), r.rolling(63).std()
    F["max_ret_21"] = r.rolling(21).max()
    F["vlm_ratio"] = v.rolling(5).mean() / v.rolling(63).mean()
    dlt = c.diff()
    up, dn = dlt.clip(lower=0).ewm(alpha=1 / 14).mean(), (-dlt.clip(upper=0)).ewm(alpha=1 / 14).mean()
    F["rsi14"] = 100 - 100 / (1 + up / dn)
    for m in (20, 50, 200):
        F[f"dist_ma{m}"] = c / c.rolling(m).mean() - 1
    F["dist_52wh"] = c / c.rolling(252).max() - 1
    F["range_5"] = ((h - lo) / c).rolling(5).mean()
    mr = nc.pct_change()
    F["beta_252"] = r.rolling(252, min_periods=200).cov(mr) .div(mr.rolling(252, min_periods=200).var(), axis=0)
    feats = list(F)

    tv = (c * v).rolling(126, min_periods=100).median()
    elig = (c.notna().rolling(252).sum() >= 250) & (c > 20) & tv.notna()
    # decision dates: every month-end + the latest available day (for current picks)
    me = pd.Series(idx, index=idx).groupby(idx.to_period("M")).max().tolist()
    if me[-1] != idx[-1]:
        me.append(idx[-1])
    pos = {dt: i for i, dt in enumerate(idx)}
    rows = []
    for k, dt in enumerate(me):
        p = pos[dt]
        u = tv.loc[dt].where(elig.loc[dt]).dropna().nlargest(UNIVERSE).index
        if len(u) < UNIVERSE * 0.8:
            continue
        entry = idx[p + 1] if p + 1 < len(idx) else None
        nxt = me[k + 1] if k + 1 < len(me) else None
        exit_ = idx[pos[nxt] + 1] if nxt is not None and pos[nxt] + 1 < len(idx) else None
        x = pd.DataFrame({f: F[f].loc[dt, u] for f in feats})
        x["sym"], x["date"], x["entry"], x["exit"] = u, dt, entry, exit_
        x["p"] = p
        x["fwd"] = (o.loc[exit_, u] / o.loc[entry, u] - 1).to_numpy() if exit_ is not None else np.nan
        rows.append(x)
    L = pd.concat(rows, ignore_index=True)
    L["mom_raw"] = L["mom_12_1"]
    # cross-sectional ranks within the month's universe (relative strength), market context as levels
    for f in feats:
        L[f] = L.groupby("date")[f].rank(pct=True) - 0.5
    L[feats] = L[feats].fillna(0.0)
    mk = pd.DataFrame({"mkt_ret_21": nc.pct_change(21), "mkt_ret_126": nc.pct_change(126),
                       "mkt_vol_21": mr.rolling(21).std() * np.sqrt(252), "mkt_dist_ma200": nc / nc.rolling(200).mean() - 1})
    L = L.join(mk, on="date")
    feats += list(mk.columns)
    L["y"] = L.groupby("date")["fwd"].rank(pct=True) - 0.5          # train on cross-sectional rank of return
    # 60-day sequences for the GRU: vol-scaled own return, return vs market, log volume vs 63-day mean
    vol63 = F["vol_63"].to_numpy()
    R = np.nan_to_num(np.clip(r.to_numpy() / np.where(vol63 > 0, vol63, np.nan), -5, 5))
    RX = np.nan_to_num(np.clip((r.sub(mr, axis=0)).to_numpy() / np.where(vol63 > 0, vol63, np.nan), -5, 5))
    LV = np.nan_to_num(np.clip(np.log(v / v.rolling(63).mean()).to_numpy(), -3, 3))
    col = {s: j for j, s in enumerate(c.columns)}
    j = L.sym.map(col).to_numpy()
    p = L.p.to_numpy()
    seq = np.zeros((len(L), SEQ_L, 3), np.float32)
    for t in range(SEQ_L):
        rr = p - (SEQ_L - 1 - t)
        seq[:, t, 0], seq[:, t, 1], seq[:, t, 2] = R[rr, j], RX[rr, j], LV[rr, j]
    return L, feats, seq


# ---------------------------------------------------------------- deep models
class MLP(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.n = nn.Sequential(nn.Linear(f, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, s=None):
        return self.n(x).squeeze(-1)


class SeqNet(nn.Module):
    """Encodes the 60-day sequence (LSTM / GRU / 1-D CNN / Transformer), then joins snapshot features."""

    def __init__(self, f, kind, ch=3, hid=32):
        super().__init__()
        self.kind = kind
        if kind in ("lstm", "gru"):
            self.enc = (nn.LSTM if kind == "lstm" else nn.GRU)(ch, hid, batch_first=True)
        elif kind == "cnn":
            self.enc = nn.Sequential(nn.Conv1d(ch, hid, 5, padding=2), nn.ReLU(), nn.Conv1d(hid, hid, 5, padding=2, dilation=1),
                                     nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        else:
            self.inp = nn.Linear(ch, hid)
            self.pos = nn.Parameter(torch.zeros(1, SEQ_L, hid))
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(hid, 4, 64, 0.1, batch_first=True), 2)
        self.head = nn.Sequential(nn.Linear(hid + f, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x, s):
        if self.kind in ("lstm", "gru"):
            out, _ = self.enc(s)
            z = out[:, -1]
        elif self.kind == "cnn":
            z = self.enc(s.transpose(1, 2)).squeeze(-1)
        else:
            z = self.enc(self.inp(s) + self.pos).mean(1)
        return self.head(torch.cat([z, x], 1)).squeeze(-1)


def fit_deep(kind, Xtr, Str, ytr, Xte, Ste, seed, epochs=60, patience=8):
    torch.manual_seed(seed)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)  # noqa: E731
    cut = int(len(Xtr) * 0.85)                                       # time-ordered: last 15 % of train = validation
    Xf, Sf, yf = t(Xtr[:cut]), t(Str[:cut]), t(ytr[:cut])
    Xv, Sv, yv = t(Xtr[cut:]), t(Str[cut:]), t(ytr[cut:])
    m = (MLP(Xtr.shape[1]) if kind == "mlp" else SeqNet(Xtr.shape[1], kind)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    best, state, bad = 9e9, None, 0
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(Xf), device=DEV)
        for b in range(0, len(Xf), 512):
            i = perm[b:b + 512]
            opt.zero_grad()
            nn.functional.mse_loss(m(Xf[i], Sf[i]), yf[i]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(m(Xv, Sv), yv).item()
        if vl < best - 1e-6:
            best, state, bad = vl, {k: x.clone() for k, x in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        return m(t(Xte), t(Ste)).cpu().numpy()


def _logit(Xtr, ytr, Xte):
    m = LogisticRegression(C=0.1, max_iter=2000).fit(Xtr, (ytr > 0).astype(int))   # P(beats the month's median)
    return m.predict_proba(Xte)[:, 1]


CLASSIC = {
    "ridge": lambda: Ridge(alpha=10.0),
    "lasso": lambda: Lasso(alpha=1e-4, max_iter=5000),
    "elastic_net": lambda: ElasticNet(alpha=2e-4, l1_ratio=0.5, max_iter=5000),
    "knn": lambda: KNeighborsRegressor(n_neighbors=200, n_jobs=-1),
    "linear_svr": lambda: LinearSVR(C=0.01, epsilon=0.0, max_iter=5000, dual="auto", random_state=0),
    "random_forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=100, max_features=0.5, n_jobs=-1, random_state=0),
    "extra_trees": lambda: ExtraTreesRegressor(n_estimators=300, min_samples_leaf=100, max_features=0.5, n_jobs=-1, random_state=0),
    "hist_gb": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=100, random_state=0),
    "xgboost": lambda: XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.02, subsample=0.7, colsample_bytree=0.7,
                                    min_child_weight=100, n_jobs=8, random_state=0, verbosity=0),
    "lightgbm": lambda: LGBMRegressor(n_estimators=400, num_leaves=15, learning_rate=0.02, min_child_samples=100,
                                      subsample=0.7, subsample_freq=1, colsample_bytree=0.7, verbose=-1, random_state=0),
    "catboost": lambda: CatBoostRegressor(iterations=500, depth=5, learning_rate=0.03, verbose=0, random_seed=0, thread_count=8),
}
DEEP = ["mlp", "lstm", "gru", "cnn", "transformer"]
ML_MODELS = list(CLASSIC) + ["logistic"] + DEEP
MODELS = ["momentum"] + ML_MODELS + ["ensemble"]
FAMILY = {**{m: "classic ML" for m in list(CLASSIC) + ["logistic"]}, **{m: "deep learning" for m in DEEP},
          "momentum": "rule (no ML)", "ensemble": "ensemble"}


def predict(L, feats, seq, tr_mask, te_mask):
    tr, te = np.where(tr_mask)[0], np.where(te_mask)[0]
    Xtr, ytr, Xte = L.loc[tr, feats].to_numpy(np.float32), L.loc[tr, "y"].to_numpy(np.float32), L.loc[te, feats].to_numpy(np.float32)
    out = pd.DataFrame(index=te)
    out["momentum"] = L.loc[te, "mom_raw"].to_numpy()
    for name, make in CLASSIC.items():
        out[name] = make().fit(Xtr, ytr).predict(Xte)
    out["logistic"] = _logit(Xtr, ytr, Xte)
    for k in DEEP:
        out[k] = np.mean([fit_deep(k, Xtr, seq[tr], ytr, Xte, seq[te], s) for s in (0, 1, 2)], axis=0)
    dates = L.loc[te, "date"].to_numpy()
    rk = [pd.Series(out[m].to_numpy(), index=te).groupby(dates).rank(pct=True) for m in ML_MODELS]
    out["ensemble"] = sum(rk) / len(rk)
    return out


# ---------------------------------------------------------------- portfolio
def simulate(L, score, opens):
    """Monthly rebalance at entry-day open, whole shares, real per-order costs. Returns per-period equity."""
    cash, hold, rec, charges, trades = CAPITAL, {}, [], 0.0, 0
    basis, closed = {}, []                           # rupees paid incl. buy charges; net P&L of each closed trade
    dates = sorted(L.loc[score.index].dropna(subset=["exit"]).date.unique())
    for dt in dates:
        g = L.loc[score.index][L.loc[score.index].date == dt].assign(s=score)
        g = g.dropna(subset=["s"]).sort_values("s", ascending=False)
        entry, exit_ = g.entry.iloc[0], g.exit.iloc[0]
        px = opens.loc[entry]
        rank = {s: i + 1 for i, s in enumerate(g.sym)}
        keep = {s: q for s, q in hold.items() if rank.get(s, 999) <= KEEP_RANK}
        for s, q in hold.items():
            if s not in keep:
                val = q * px[s]
                cost = order_cost(val, "sell")
                cash += val - cost
                charges += cost
                trades += 1
                closed.append({"sym": s, "exit": entry, "pnl": val - cost - basis.pop(s)})
        hold = keep
        equity = cash + sum(q * px[s] for s, q in hold.items())
        new = [s for s in g.sym if s not in hold][: TOP_N - len(hold)]
        slot = equity / TOP_N
        for s in new:
            budget = min(slot, cash / max(1, len(new) - new.index(s)))
            q = int(budget / (px[s] * (1 + 0.003)))
            if q <= 0 or not np.isfinite(px[s]):
                continue
            val = q * px[s]
            cost = order_cost(val, "buy")
            if val + cost > cash:
                continue
            cash -= val + cost
            charges += cost
            trades += 1
            hold[s] = q
            basis[s] = val + cost
        end_px = opens.loc[exit_]
        rec.append({"date": dt, "exit": exit_, "equity_end": cash + sum(q * end_px[s] for s, q in hold.items()),
                    "n_hold": len(hold)})
    eq = pd.DataFrame(rec).set_index("date")
    return eq, charges, trades, pd.DataFrame(closed)


def stats(eq: pd.Series, start_value=CAPITAL):
    rets = eq.pct_change().fillna(eq.iloc[0] / start_value - 1)
    yrs = len(eq) / 12
    cagr = (eq.iloc[-1] / start_value) ** (1 / yrs) - 1
    dd = (eq / eq.cummax().clip(lower=start_value) - 1).min()
    return {"cagr_pct": round(cagr * 100, 1), "max_dd_pct": round(min(dd, 0) * 100, 1),
            "sharpe": round(float(rets.mean() / rets.std() * np.sqrt(12)), 2), "final_rs": round(float(eq.iloc[-1]))}


def main() -> int:
    P, nifty = load()
    L, feats, seq = build(P, nifty)
    print(f"{len(L):,} stock-months, {L.date.nunique()} decision dates, {L.sym.nunique()} distinct stocks, device {DEV}", flush=True)
    preds = []
    for Y in range(2014, 2027):
        tr = (L.exit.notna() & (L.exit < pd.Timestamp(f"{Y}-01-01"))).to_numpy()
        te = (L.date.dt.year == Y).to_numpy()
        if te.sum() == 0:
            continue
        preds.append(predict(L, feats, seq, tr, te))
        print(f"  {Y}: train {tr.sum():,} stock-months, test {te.sum():,}", flush=True)
    pr = pd.concat(preds)
    opens = P["open"].ffill()
    out = ROOT / "results" / f"stocks_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    L.loc[pr.index, ["date", "sym", "fwd"]].join(pr).to_parquet(out / "predictions.parquet")
    T = L.loc[pr.index]
    res, curves, yearly = {}, {}, {}
    no = nifty["open"]
    base_win = float((T.fwd.dropna() > 0).mean() * 100)
    for m in MODELS:
        eq, ch, n, closed = simulate(L, pr[m], opens)
        curves[m] = eq.equity_end
        mret = eq.equity_end.pct_change().fillna(eq.equity_end.iloc[0] / CAPITAL - 1)
        nret = (no.reindex(eq.exit).to_numpy() / no.reindex(T.groupby("date").entry.first().reindex(eq.index)).to_numpy() - 1)
        top = T.assign(s=pr[m]).dropna(subset=["fwd"])
        top = top[top.groupby("date").s.rank(ascending=False) <= TOP_N]
        def _sh(x):
            return round(float(x.mean() / x.std() * np.sqrt(12)), 2)
        sel, hold_ = mret[mret.index < "2022-01-01"], mret[mret.index >= "2022-01-01"]
        ic = T.assign(s=pr[m]).dropna(subset=["fwd"]).groupby("date").apply(
            lambda g: g.s.rank().corr(g.fwd.rank()), include_groups=False)
        s = stats(eq.equity_end)
        late = eq.equity_end[eq.index >= "2024-01-01"]
        base = eq.equity_end[eq.index < "2024-01-01"].iloc[-1]
        s.update({"cagr_2024plus_pct": stats(late, base)["cagr_pct"], "mean_IC": round(float(ic.mean()), 4),
                  "IC_2024plus": round(float(ic[ic.index >= "2024-01-01"].mean()), 4),
                  "charges_rs": round(ch), "orders": n,
                  "trade_win_rate_pct": round(float((closed.pnl > 0).mean() * 100), 1) if len(closed) else None,
                  "avg_win_rs": round(float(closed.pnl[closed.pnl > 0].mean())) if len(closed) else None,
                  "avg_loss_rs": round(float(closed.pnl[closed.pnl <= 0].mean())) if len(closed) else None,
                  "pick_up_pct": round(float((top.fwd > 0).mean() * 100), 1),
                  "months_up_pct": round(float((mret > 0).mean() * 100), 1),
                  "months_beat_nifty_pct": round(float((mret.to_numpy() > nret).mean() * 100), 1),
                  "sharpe_2014_21_select": _sh(sel), "sharpe_2022_26_holdout": _sh(hold_),
                  "family": FAMILY[m]})
        res[m] = s
        yearly[m] = eq.equity_end.groupby(eq.index.year).last().pct_change().mul(100).round(1)
    # benchmarks on the same periods (NIFTY 50 price index excludes ~1.3 %/yr dividends; EW universe is gross)
    per = T.dropna(subset=["fwd"]).groupby("date").agg(entry=("entry", "first"), exit=("exit", "first"), ew=("fwd", "mean"))
    per["nifty"] = no.reindex(per.exit).to_numpy() / no.reindex(per.entry).to_numpy() - 1
    for b in ("nifty", "ew"):
        eq = CAPITAL * (1 + per[b]).cumprod()
        curves[b] = eq
        s = stats(eq)
        late = eq[eq.index >= "2024-01-01"]
        s["cagr_2024plus_pct"] = stats(late, eq[eq.index < "2024-01-01"].iloc[-1])["cagr_pct"]
        res["bench_" + {"nifty": "nifty50_price", "ew": "equal_weight_top100_gross"}[b]] = s
        yearly["bench_" + b] = eq.groupby(eq.index.year).last().pct_change().mul(100).round(1)
    tab = pd.DataFrame(res).T
    ml = tab.loc[MODELS]
    chosen = ml.sharpe_2014_21_select.astype(float).idxmax()
    # current picks: models trained on every month whose holding period has finished
    last = L.date.max()
    tr = L.exit.notna().to_numpy()
    te = (L.date == last).to_numpy()
    cur = predict(L, feats, seq, tr, te)
    picks = L.loc[cur.index, ["sym"]].assign(**{m: cur[m].rank(ascending=False).astype(int) for m in MODELS})
    picks = picks.sort_values(chosen).head(20)[["sym", chosen, "ensemble", "momentum"]]
    picks.to_csv(out / "current_picks.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "equity.csv")
    (out / "report.json").write_text(json.dumps({"summary": res, "yearly_pct": {k: v.dropna().to_dict() for k, v in yearly.items()},
                                                  "picks_as_of": str(last.date()), "chosen_on_2014_2021": chosen,
                                                  "base_rate_up_pct": base_win}, indent=1, default=str))
    pd.set_option("display.width", 250)
    (out / "summary.csv").write_text(tab.to_csv())
    print(f"\nOUT-OF-SAMPLE 2014 -> 2026, Rs 1,00,000, top-10 monthly, after Groww charges + slippage."
          f" Base rate: {base_win:.1f}% of all stock-months went up.")
    cols = ["family", "cagr_pct", "max_dd_pct", "sharpe", "cagr_2024plus_pct", "trade_win_rate_pct", "avg_win_rs",
            "avg_loss_rs", "pick_up_pct", "months_beat_nifty_pct", "mean_IC", "sharpe_2014_21_select", "sharpe_2022_26_holdout"]
    print(ml.sort_values("sharpe_2014_21_select", ascending=False)[cols].to_string())
    print(tab.loc[[i for i in tab.index if i.startswith("bench")], ["cagr_pct", "max_dd_pct", "sharpe", "cagr_2024plus_pct"]].to_string())
    print(f"\nModel chosen on 2014-2021 only: {chosen} -> hold-out 2022-2026 Sharpe {ml.loc[chosen, 'sharpe_2022_26_holdout']}"
          f" (NIFTY-hold-out comparison in yearly table)")
    print("\nYearly return %:")
    print(pd.DataFrame(yearly)[[chosen, "ensemble", "momentum", "bench_nifty", "bench_ew"]].dropna(how="all").to_string())
    print(f"\nCurrent ranking as of {last.date()} (top 20 by the chosen model {chosen}):")
    print(picks.to_string(index=False))
    print("results ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
