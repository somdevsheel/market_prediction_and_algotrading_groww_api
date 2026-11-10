"""BUY / HOLD / SELL classifiers for the whole Indian stock market (Groww list, Yahoo prices) + backtest.

Data       data_lake/silver/yahoo_stocks/all_daily.parquet (5,201 NSE+BSE stocks). Used from 2009-01-01: earlier
           Yahoo history for Indian stocks is full of bad prints (2007: 9.5k >50 % jumps; 2009+: ~40 a year).
Cleaning   one-day spikes that reverse the next day are removed; a stock is skipped for a month if an
           unexplained > 50 % one-day jump happened in the 21 trading days before the decision.
Universe   each month-end: price >= Rs 20, >= 1 year of history, 6-month median traded value >= Rs 2 crore/day.
Label      return from next day's open to the open after the next month-end:
           BUY  > +4 %   |   SELL < -4 %   |   HOLD otherwise
Models     classic : logistic, KNN, random forest, extra trees, hist-gradient-boosting, XGBoost, LightGBM, CatBoost
           deep    : MLP, LSTM, GRU, 1-D CNN, Transformer (sequence nets read 60 days of price/volume)
           ensemble: average probabilities of all of the above
           Yearly walk-forward: train only on months whose holding period ended before the test year.
Backtest   Rs 1,00,000, max 10 stocks, equal rupees, whole shares.
           BUY signal -> buy (best score first, if a slot is free) | HOLD -> keep | SELL or left universe -> sell.
           Groww delivery charges + slippage by liquidity (0.1 % / 0.2 % / 0.4 %).
Selection  models are ranked on 2013-2019 only; 2020-2026 is a hold-out.

    python3 -m ml_lab.stocks_bhs
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from xgboost import XGBClassifier

from core.config import load_config, resolve_path
from ml_lab.models.deep import DEV

ROOT = Path(__file__).resolve().parent
LAKE = resolve_path(load_config(), "data_lake")
START = "2009-01-01"
MIN_TV, MIN_PX = 2e7, 20.0                  # Rs 2 crore median daily traded value, Rs 20 price
UP, DN = 0.04, -0.04
SEQ_L, TOP_N, CAPITAL = 60, 10, 100_000.0
TEST_YEARS = range(2013, 2027)
SELECT_END = 2019
CLASSES = ["SELL", "HOLD", "BUY"]


def slippage(tv: float) -> float:
    return 0.001 if tv >= 5e8 else 0.002 if tv >= 1e8 else 0.004


def order_cost(value: float, side: str, slip: float) -> float:
    """Groww equity delivery (approx.): brokerage min(Rs 20, 0.1 %) floor Rs 5, STT 0.1 %, exchange, SEBI,
    stamp 0.015 % (buy), GST 18 %, DP ~Rs 15.93 per scrip per sell day, plus slippage."""
    brokerage = max(5.0, min(20.0, 0.001 * value))
    exch, sebi = 0.0000297 * value, 0.000001 * value
    return (brokerage + 0.001 * value + exch + sebi + (0.00015 * value if side == "buy" else 0.0)
            + (15.93 if side == "sell" else 0.0) + 0.18 * (brokerage + exch + sebi) + slip * value)


# ------------------------------------------------------------------ data
def load_panels():
    d = pd.read_parquet(LAKE / "silver" / "yahoo_stocks" / "all_daily.parquet",
                        columns=["date", "sym", "open", "high", "low", "close", "volume"])
    d = d[d.date >= START]
    P = {k: d.pivot(index="date", columns="sym", values=k).sort_index().astype("float32")
         for k in ("open", "high", "low", "close", "volume")}
    c = P["close"]
    r = c / c.ffill().shift(1) - 1
    rn = r.shift(-1)
    spike = (r.abs() > 0.4) & (rn.abs() > 0.25) & (np.sign(r) != np.sign(rn)) & \
            ((c.shift(-1) / c.ffill().shift(1) - 1).abs() < 0.15)
    for k in ("open", "high", "low", "close"):
        P[k] = P[k].mask(spike)
    c = P["close"]
    r = c / c.ffill().shift(1) - 1
    jump = (r.abs() > 0.5)
    print(f"cleaning: removed {int(spike.sum().sum())} one-day spikes; {int(jump.sum().sum())} big jumps left (flagged)", flush=True)
    nifty = pd.read_parquet(LAKE / "silver" / "yahoo" / "NIFTY.parquet").set_index("date")
    return P, jump, nifty


def build(P, jump, nifty):
    c, o, v, h, lo = P["close"], P["open"], P["volume"], P["high"], P["low"]
    idx = c.index
    cf = c.ffill(limit=5)
    r = cf.pct_change(fill_method=None)
    me = pd.Series(idx, index=idx).groupby(idx.to_period("M")).max().tolist()
    if me[-1] != idx[-1]:
        me.append(idx[-1])
    pos = {dt: i for i, dt in enumerate(idx)}
    me = [dt for dt in me if pos[dt] >= 252]
    nc = nifty["close"].reindex(idx).ffill()
    mr = nc.pct_change()

    def at(df):                                    # only month-end rows (saves memory)
        return df.loc[me]

    F = {}
    for k in (5, 21, 63, 126, 252):
        F[f"ret_{k}"] = at(cf.pct_change(k, fill_method=None))
    F["mom_12_1"] = at(cf.shift(21) / cf.shift(252) - 1)
    F["vol_21"], F["vol_63"] = at(r.rolling(21).std()), at(r.rolling(63).std())
    F["max_ret_21"], F["min_ret_21"] = at(r.rolling(21).max()), at(r.rolling(21).min())
    F["vlm_ratio"] = at(v.rolling(5).mean() / v.rolling(63).mean())
    dlt = cf.diff()
    up, dn = dlt.clip(lower=0).ewm(alpha=1 / 14).mean(), (-dlt.clip(upper=0)).ewm(alpha=1 / 14).mean()
    F["rsi14"] = at(100 - 100 / (1 + up / dn))
    for m in (20, 50, 200):
        F[f"dist_ma{m}"] = at(cf / cf.rolling(m).mean() - 1)
    F["dist_52wh"] = at(cf / cf.rolling(252).max() - 1)
    F["dist_52wl"] = at(cf / cf.rolling(252).min() - 1)
    F["range_5"] = at(((h - lo) / cf).rolling(5).mean())
    F["beta_252"] = at(r.rolling(252, min_periods=200).cov(mr).div(mr.rolling(252, min_periods=200).var(), axis=0))
    tv = (cf * v).rolling(126, min_periods=100).median()
    F["log_tv"] = np.log(at(tv) + 1)
    F["log_px"] = np.log(at(cf))
    feats = list(F)
    elig = at((c.notna().rolling(252).sum() >= 240) & (cf >= MIN_PX) & (tv >= MIN_TV)) & ~at(jump.astype("float32").rolling(21).max() > 0)
    tv_me = at(tv)
    rows = []
    for k, dt in enumerate(me):
        p = pos[dt]
        u = elig.columns[elig.loc[dt].to_numpy()]
        if len(u) < 50:
            continue
        entry = idx[p + 1] if p + 1 < len(idx) else pd.NaT
        nxt = me[k + 1] if k + 1 < len(me) else None
        exit_ = idx[pos[nxt] + 1] if nxt is not None and pos[nxt] + 1 < len(idx) else pd.NaT
        x = pd.DataFrame({f: F[f].loc[dt, u].to_numpy() for f in feats}, index=u)
        x["sym"], x["date"], x["entry"], x["exit"], x["p"] = u, dt, entry, exit_, p
        x["tv"] = tv_me.loc[dt, u].to_numpy()
        if pd.notna(exit_):
            x["fwd"] = (o.loc[exit_, u] / o.loc[entry, u] - 1).to_numpy()
        else:
            x["fwd"] = np.nan
        rows.append(x.reset_index(drop=True))
    L = pd.concat(rows, ignore_index=True)
    L["mom_raw"] = L["mom_12_1"]
    L["breadth"] = L.groupby("date")["dist_ma200"].transform(lambda s: (s > 0).mean())   # before ranking
    raw_keep = ["log_tv", "log_px"]
    for f in feats:                                 # cross-sectional percentile ranks per month (except size/price)
        if f not in raw_keep:
            L[f] = L.groupby("date")[f].rank(pct=True) - 0.5
    L[feats] = L[feats].fillna(0.0)
    mk = pd.DataFrame({"mkt_ret_21": nc.pct_change(21), "mkt_ret_126": nc.pct_change(126),
                       "mkt_vol_21": mr.rolling(21).std() * np.sqrt(252), "mkt_dist_ma200": nc / nc.rolling(200).mean() - 1})
    L = L.join(mk, on="date")
    feats += list(mk.columns) + ["breadth"]
    L["y"] = np.where(L.fwd > UP, 2, np.where(L.fwd < DN, 0, 1))
    L.loc[L.fwd.isna(), "y"] = -1
    # 60-day sequences (vol-scaled return, return vs market, log volume vs 63-day mean)
    vol63 = r.rolling(63).std().to_numpy()
    R = np.nan_to_num(np.clip(r.to_numpy() / np.where(vol63 > 0, vol63, np.nan), -5, 5)).astype(np.float32)
    RX = np.nan_to_num(np.clip(r.sub(mr, axis=0).to_numpy() / np.where(vol63 > 0, vol63, np.nan), -5, 5)).astype(np.float32)
    LV = np.nan_to_num(np.clip(np.log(v / v.rolling(63).mean()).to_numpy(), -3, 3)).astype(np.float32)
    col = {s: j for j, s in enumerate(c.columns)}
    j, p = L.sym.map(col).to_numpy(), L.p.to_numpy()
    seq = np.zeros((len(L), SEQ_L, 3), np.float32)
    for t in range(SEQ_L):
        rr = p - (SEQ_L - 1 - t)
        seq[:, t, 0], seq[:, t, 1], seq[:, t, 2] = R[rr, j], RX[rr, j], LV[rr, j]
    return L, feats, seq


# ------------------------------------------------------------------ models
class MLP(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.n = nn.Sequential(nn.Linear(f, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x, s):
        return self.n(x)


class SeqNet(nn.Module):
    def __init__(self, f, kind, ch=3, hid=32):
        super().__init__()
        self.kind = kind
        if kind in ("lstm", "gru"):
            self.enc = (nn.LSTM if kind == "lstm" else nn.GRU)(ch, hid, batch_first=True)
        elif kind == "cnn":
            self.enc = nn.Sequential(nn.Conv1d(ch, hid, 5, padding=2), nn.ReLU(), nn.Conv1d(hid, hid, 5, padding=2),
                                     nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        else:
            self.inp = nn.Linear(ch, hid)
            self.pos = nn.Parameter(torch.zeros(1, SEQ_L, hid))
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(hid, 4, 64, 0.1, batch_first=True), 2)
        self.head = nn.Sequential(nn.Linear(hid + f, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 3))

    def forward(self, x, s):
        if self.kind in ("lstm", "gru"):
            z = self.enc(s)[0][:, -1]
        elif self.kind == "cnn":
            z = self.enc(s.transpose(1, 2)).squeeze(-1)
        else:
            z = self.enc(self.inp(s) + self.pos).mean(1)
        return self.head(torch.cat([z, x], 1))


def fit_deep(kind, Xtr, Str, ytr, Xte, Ste, seed, epochs=40, patience=6, bs=1024):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=DEV)  # noqa: E731
    cut = int(len(Xtr) * 0.85)                                              # time-ordered validation tail
    Xf, Sf, yf = t(Xtr[:cut]), t(Str[:cut]), t(ytr[:cut], torch.long)
    Xv, Sv, yv = t(Xtr[cut:]), t(Str[cut:]), t(ytr[cut:], torch.long)
    m = (MLP(Xtr.shape[1]) if kind == "mlp" else SeqNet(Xtr.shape[1], kind)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    best, state, bad = 9e9, None, 0
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(Xf), device=DEV)
        for b in range(0, len(Xf), bs):
            i = perm[b:b + bs]
            opt.zero_grad()
            lossf(m(Xf[i], Sf[i]), yf[i]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = sum(lossf(m(Xv[b:b + 8192], Sv[b:b + 8192]), yv[b:b + 8192]).item() * len(yv[b:b + 8192])
                     for b in range(0, len(Xv), 8192)) / len(Xv)
        if vl < best - 1e-5:
            best, state, bad = vl, {k: x.clone() for k, x in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state)
    m.eval()
    out = []
    with torch.no_grad():
        Xt, St = t(Xte), t(Ste)
        for b in range(0, len(Xt), 8192):
            out.append(torch.softmax(m(Xt[b:b + 8192], St[b:b + 8192]), 1).cpu().numpy())
    return np.concatenate(out)


CLASSIC = {
    "logistic": lambda: LogisticRegression(C=0.1, max_iter=3000),
    "knn": lambda: KNeighborsClassifier(n_neighbors=300, n_jobs=-1),
    "random_forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=200, max_features=0.4, n_jobs=-1, random_state=0),
    "extra_trees": lambda: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=200, max_features=0.5, n_jobs=-1, random_state=0),
    "hist_gb": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300, random_state=0),
    "xgboost": lambda: XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.03, subsample=0.7, colsample_bytree=0.7,
                                     min_child_weight=200, n_jobs=12, random_state=0, verbosity=0, tree_method="hist"),
    "lightgbm": lambda: LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.03, min_child_samples=300,
                                       subsample=0.7, subsample_freq=1, colsample_bytree=0.7, verbose=-1, random_state=0),
    "catboost": lambda: CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, loss_function="MultiClass",
                                           verbose=0, random_seed=0, thread_count=12),
}
DEEP = ["mlp", "lstm", "gru", "cnn", "transformer"]
MODELS = list(CLASSIC) + DEEP + ["ensemble"]
FAMILY = {**{m: "ML" for m in CLASSIC}, **{m: "DL" for m in DEEP}, "ensemble": "ML+DL"}


def predict(L, feats, seq, tr, te, log=None):
    Xtr, ytr = L.loc[tr, feats].to_numpy(np.float32), L.loc[tr, "y"].to_numpy()
    Xte = L.loc[te, feats].to_numpy(np.float32)
    probs = {}
    for name, make in CLASSIC.items():
        X1, X2 = (Xtr, Xte)
        if name in ("logistic", "knn"):                       # scale-sensitive models
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            X1, X2 = (Xtr - mu) / sd, (Xte - mu) / sd
        Xfit, yfit = (X1[::3], ytr[::3]) if name == "knn" else (X1, ytr)   # KNN: thin the training set for speed
        probs[name] = make().fit(Xfit, yfit).predict_proba(X2)
        if log:
            log(name)
    for k in DEEP:
        probs[k] = np.mean([fit_deep(k, Xtr, seq[tr], ytr, Xte, seq[te], s) for s in (0, 1)], axis=0)
        if log:
            log(k)
    probs["ensemble"] = np.mean([probs[m] for m in MODELS if m != "ensemble"], axis=0)
    return probs


# ------------------------------------------------------------------ backtest
def backtest(T: pd.DataFrame, opens: pd.DataFrame):
    """T: test rows with columns sym, date, entry, exit, tv, signal (0/1/2), score. Per-month marks."""
    cash, hold, basis, closed, rec, charges = CAPITAL, {}, {}, [], [], 0.0
    for dt, g in T.dropna(subset=["exit"]).groupby("date", sort=True):
        entry, exit_ = g.entry.iloc[0], g.exit.iloc[0]
        px = opens.loc[entry]
        sig = dict(zip(g.sym, g.signal))
        tvs = dict(zip(g.sym, g.tv))
        for s in list(hold):
            if sig.get(s, 0) == 0:                               # SELL signal, or no longer in the universe
                p = px[s]
                if not np.isfinite(p):
                    continue
                val = hold[s] * p
                cost = order_cost(val, "sell", slippage(tvs.get(s, 0)))
                cash += val - cost
                charges += cost
                closed.append({"sym": s, "date": dt, "pnl": val - cost - basis.pop(s)})
                hold.pop(s)
        equity = cash + sum(q * px[s] for s, q in hold.items() if np.isfinite(px[s]))
        buys = g[(g.signal == 2) & ~g.sym.isin(list(hold))].sort_values("score", ascending=False)
        free = TOP_N - len(hold)
        for _, b in buys.head(free).iterrows():
            p = px[b.sym]
            budget = min(equity / TOP_N, cash)
            q = int(budget / (p * 1.01)) if np.isfinite(p) and p > 0 else 0
            if q <= 0:
                continue
            val = q * p
            cost = order_cost(val, "buy", slippage(b.tv))
            if val + cost > cash:
                continue
            cash -= val + cost
            charges += cost
            hold[b.sym], basis[b.sym] = q, val + cost
        end = opens.loc[exit_]
        rec.append({"date": dt, "equity": cash + sum(q * end[s] for s, q in hold.items() if np.isfinite(end[s])),
                    "n_hold": len(hold), "cash": cash})
    return pd.DataFrame(rec).set_index("date"), pd.DataFrame(closed), charges


def perf(eq: pd.Series, start=CAPITAL) -> dict:
    r = eq.pct_change().fillna(eq.iloc[0] / start - 1)
    yrs = len(eq) / 12
    return {"cagr_pct": round(((eq.iloc[-1] / start) ** (1 / yrs) - 1) * 100, 1),
            "max_dd_pct": round(float((eq / eq.cummax().clip(lower=start) - 1).min() * 100), 1),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(12)), 2) if r.std() > 0 else 0.0}


def main() -> int:
    t0 = datetime.now()
    P, jump, nifty = load_panels()
    L, feats, seq = build(P, jump, nifty)
    lab = L[L.y >= 0]
    print(f"{len(L):,} stock-months | {L.date.nunique()} months | {L.sym.nunique()} distinct stocks | "
          f"universe now {int((L.date == L.date.max()).sum())} stocks | classes SELL/HOLD/BUY = "
          f"{', '.join(f'{(lab.y == k).mean() * 100:.0f}%' for k in (0, 1, 2))} | device {DEV}", flush=True)
    out = ROOT / "results" / f"bhs_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    allp = []
    for Y in TEST_YEARS:
        tr = np.where((L.y >= 0) & L.exit.notna() & (L.exit < pd.Timestamp(f"{Y}-01-01")))[0]
        te = np.where(L.date.dt.year == Y)[0]
        if len(te) == 0:
            continue
        pr = predict(L, feats, seq, tr, te)
        df = L.loc[te, ["date", "sym", "entry", "exit", "tv", "fwd", "y", "mom_raw"]].copy()
        for m, p in pr.items():
            df[f"{m}_p0"], df[f"{m}_p1"], df[f"{m}_p2"] = p[:, 0], p[:, 1], p[:, 2]
        allp.append(df)
        df.to_parquet(out / f"pred_{Y}.parquet")
        print(f"  {Y}: trained on {len(tr):,}, predicted {len(te):,}  ({(datetime.now() - t0).seconds // 60} min)", flush=True)
    T = pd.concat(allp, ignore_index=True)
    opens = P["open"].ffill(limit=10)
    no = nifty["open"]
    lab = T[T.y >= 0]
    res, curves, yearly = {}, {}, {}
    for m in MODELS:
        p = T[[f"{m}_p0", f"{m}_p1", f"{m}_p2"]].to_numpy()
        T["signal"], T["score"] = p.argmax(1), p[:, 2] - p[:, 0]
        k = T.y >= 0
        sig, y, fwd = T.signal[k], T.y[k], T.fwd[k]
        eq, closed, ch = backtest(T, opens)
        curves[m] = eq.equity
        s = perf(eq.equity)
        sel, hold_ = eq.equity[eq.index.year <= SELECT_END], eq.equity[eq.index.year > SELECT_END]
        s.update({
            "family": FAMILY[m],
            "accuracy_pct": round(float((sig == y).mean() * 100), 1),
            "buy_signals_pct": round(float((sig == 2).mean() * 100), 1),
            "BUY_precision_pct": round(float((y[sig == 2] == 2).mean() * 100), 1) if (sig == 2).any() else None,
            "BUY_went_up_pct": round(float((fwd[sig == 2] > 0).mean() * 100), 1) if (sig == 2).any() else None,
            "BUY_avg_month_ret_pct": round(float(fwd[sig == 2].mean() * 100), 2) if (sig == 2).any() else None,
            "SELL_precision_pct": round(float((y[sig == 0] == 0).mean() * 100), 1) if (sig == 0).any() else None,
            "SELL_avg_month_ret_pct": round(float(fwd[sig == 0].mean() * 100), 2) if (sig == 0).any() else None,
            "trade_win_rate_pct": round(float((closed.pnl > 0).mean() * 100), 1) if len(closed) else None,
            "trades": len(closed), "charges_rs": round(ch), "avg_stocks_held": round(float(eq.n_hold.mean()), 1),
            "sharpe_2013_19_select": perf(sel)["sharpe"], "sharpe_2020_26_holdout": perf(hold_, sel.iloc[-1])["sharpe"],
            "cagr_2020_26_holdout": perf(hold_, sel.iloc[-1])["cagr_pct"],
        })
        res[m] = s
        yearly[m] = pd.concat([pd.Series([CAPITAL], index=[eq.index[0].year - 1]),
                               eq.equity.groupby(eq.index.year).last()]).pct_change().iloc[1:].mul(100).round(1)
    # benchmarks (same months): NIFTY 50 price index; equal weight of the whole universe (gross)
    per = T.dropna(subset=["fwd"]).groupby("date").agg(entry=("entry", "first"), exit=("exit", "first"), ew=("fwd", "mean"),
                                                       base_up=("fwd", lambda s: (s > 0).mean()))
    per["nifty"] = no.reindex(per.exit).to_numpy() / no.reindex(per.entry).to_numpy() - 1
    for b in ("nifty", "ew"):
        eq = CAPITAL * (1 + per[b]).cumprod()
        s = perf(eq)
        sel, hold_ = eq[eq.index.year <= SELECT_END], eq[eq.index.year > SELECT_END]
        s.update({"family": "benchmark", "sharpe_2013_19_select": perf(sel)["sharpe"],
                  "sharpe_2020_26_holdout": perf(hold_, sel.iloc[-1])["sharpe"], "cagr_2020_26_holdout": perf(hold_, sel.iloc[-1])["cagr_pct"]})
        res[{"nifty": "NIFTY50 (price)", "ew": "all universe equal-wt (gross)"}[b]] = s
        yearly[b] = pd.concat([pd.Series([CAPITAL], index=[eq.index[0].year - 1]), eq.groupby(eq.index.year).last()]).pct_change().iloc[1:].mul(100).round(1)
    tab = pd.DataFrame(res).T
    chosen = tab.loc[MODELS].sharpe_2013_19_select.astype(float).idxmax()
    tab.to_csv(out / "summary.csv")
    pd.DataFrame(yearly).to_csv(out / "yearly.csv")
    pd.DataFrame(curves).to_csv(out / "equity.csv")
    # current signals: train on everything with a finished holding period, predict the latest date
    last = L.date.max()
    tr = np.where((L.y >= 0) & L.exit.notna())[0]
    te = np.where(L.date == last)[0]
    pr = predict(L, feats, seq, tr, te)
    cur = L.loc[te, ["sym", "tv"]].copy()
    for m in (chosen, "ensemble"):
        cur[f"{m}_signal"] = [CLASSES[i] for i in pr[m].argmax(1)]
        cur[f"{m}_p_buy"], cur[f"{m}_p_sell"] = pr[m][:, 2].round(3), pr[m][:, 0].round(3)
    cur = cur.sort_values(f"{chosen}_p_buy", ascending=False)
    cur.to_csv(out / "current_signals.csv", index=False)
    base_up = float((lab.fwd > 0).mean() * 100)
    (out / "report.json").write_text(json.dumps({"summary": res, "chosen_on_2013_2019": chosen, "base_rate_up_pct": base_up,
                                                 "class_share": {c: float((lab.y == i).mean()) for i, c in enumerate(CLASSES)},
                                                 "signals_as_of": str(last.date())}, indent=1, default=str))
    pd.set_option("display.width", 260)
    print(f"\nOUT-OF-SAMPLE 2013-2026. Base rate: {base_up:.1f}% of all stock-months went up; "
          f"labels SELL/HOLD/BUY = {', '.join(f'{(lab.y == i).mean() * 100:.0f}%' for i in range(3))}")
    cols = ["family", "accuracy_pct", "BUY_precision_pct", "BUY_went_up_pct", "BUY_avg_month_ret_pct", "SELL_precision_pct",
            "SELL_avg_month_ret_pct", "cagr_pct", "max_dd_pct", "sharpe", "trade_win_rate_pct", "avg_stocks_held",
            "sharpe_2013_19_select", "sharpe_2020_26_holdout", "cagr_2020_26_holdout"]
    print(tab.reindex(columns=cols).sort_values("sharpe_2013_19_select", ascending=False).to_string())
    print(f"\nModel chosen on 2013-2019 only: {chosen}")
    print("\nYearly return %:")
    print(pd.DataFrame(yearly).to_string())
    print(f"\nSignals as of {last.date()} ({chosen}): BUY {int((cur[f'{chosen}_signal'] == 'BUY').sum())}, "
          f"HOLD {int((cur[f'{chosen}_signal'] == 'HOLD').sum())}, SELL {int((cur[f'{chosen}_signal'] == 'SELL').sum())}")
    print(cur.head(15).to_string(index=False))
    print("results ->", out, f"| total {(datetime.now() - t0).seconds // 60} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
