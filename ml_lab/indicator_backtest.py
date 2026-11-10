"""Technical-indicator strategies for Indian stocks that beat holding all liquid stocks (daily bars, long-only).

140 indicators were backtested one by one with textbook rules (commit 643a8c6).  Only three beat the
"hold all liquid stocks" benchmark after costs, in both 2010-2019 and 2020 -> today, so only those are kept:
  50  Coppock Curve          15.5 %/yr, Sharpe 0.79   (hold-all 12.2 %/yr, Sharpe 0.65)
  139 Rolling Sharpe Ratio   14.6 %/yr, Sharpe 0.76
  140 Rolling Sortino Ratio  14.2 %/yr, Sharpe 0.74
The other 137 rules (and the S/R + candlestick notebook port) were removed; they remain in git history.

Rule evaluated at the close of day t; trade at the open of day t+1.  BUY state -> hold, otherwise no position.

Universe (point-in-time, same as ml_lab/stocks_bhs.py): Yahoo daily data from 2009, price >= Rs 20, median traded value
>= Rs 2 crore/day over 126 days, >= ~1 year of history, no >50 % one-day data jump in the last 21 days.

Portfolio per indicator: equal weight across every stock currently in a BUY state (cash when none).
Costs (Groww delivery, Rs ~10,000 per position): buy 0.24 %, sell 0.38 % (incl. Rs 15.93 DP charge), plus slippage
0.1 / 0.2 / 0.4 % per side by liquidity.  Daily re-balancing friction between held stocks is ignored.

Benchmarks: "Hold all liquid stocks" (same universe, always invested, same costs) and NIFTY 50.

    python3 -m ml_lab.indicator_backtest
"""
from __future__ import annotations

import json
from datetime import datetime

import numba as nb
import numpy as np
import pandas as pd

from ml_lab.stocks_bhs import MIN_PX, MIN_TV, ROOT, load_panels

START_EVAL = "2010-01-01"
SPLIT = 2020
BUY_COST, SELL_COST = 0.0024, 0.0038


# ============================================================ numba kernels (inputs are float64 (T, S))
@nb.njit(cache=True, parallel=True)
def nb_hold(entry, exit_, close):
    """Position state machine: enter on `entry`, leave on `exit_`."""
    T, S = entry.shape
    out = np.zeros((T, S), np.bool_)
    for s in nb.prange(S):
        on = False
        for t in range(T):
            if on:
                if exit_[t, s]:
                    on = False
            elif entry[t, s] and not np.isnan(close[t, s]):
                on = True
            out[t, s] = on
    return out


@nb.njit(cache=True, parallel=True)
def nb_wdot(x, w):
    T, S = x.shape
    n = len(w)
    ws = w.sum()
    out = np.full((T, S), np.nan)
    for s in nb.prange(S):
        for t in range(n - 1, T):
            acc = 0.0
            ok = True
            for k in range(n):
                v = x[t - n + 1 + k, s]
                if np.isnan(v):
                    ok = False
                    break
                acc += w[k] * v
            if ok:
                out[t, s] = acc / ws
    return out


def F(x) -> np.ndarray:
    return np.ascontiguousarray(x.to_numpy(np.float64) if isinstance(x, pd.DataFrame) else x, dtype=np.float64)


def wma(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return pd.DataFrame(nb_wdot(F(x), np.arange(1, n + 1, dtype=np.float64)), index=x.index, columns=x.columns)


# ============================================================ data
class Data:
    def __init__(self):
        P, jump, nifty = load_panels()
        c = P["close"].astype("float64")
        cf = c.ffill(limit=5)
        tv = (cf * P["volume"]).rolling(126, min_periods=100).median()
        elig = (c.notna().rolling(252).sum() >= 240) & (cf >= MIN_PX) & (tv >= MIN_TV) & \
               ~(jump.astype("float32").rolling(21).max() > 0)
        keep = elig[elig.index >= START_EVAL].any()
        cols = keep[keep].index
        self.idx = c.index
        self.C = cf[cols]
        self.O = P["open"][cols].astype("float64").ffill(limit=5).fillna(self.C).where(self.C.notna())
        self.elig = elig[cols].to_numpy() & (self.idx >= START_EVAL)[:, None]
        self.nifty = nifty["close"].reindex(self.idx).ffill()
        self.ret = self.C.pct_change(fill_method=None)
        # forward open-to-open return earned by a position decided at the close of t
        R = self.O.shift(-2) / self.O.shift(-1) - 1
        J = jump[cols].reindex(self.idx).fillna(False)
        R = R.mask(J.shift(-1, fill_value=False) | J.shift(-2, fill_value=False))
        self.R = R.fillna(0.0).to_numpy()
        tvn = tv[cols].to_numpy()
        self.slip = np.where(tvn >= 5e8, 0.001, np.where(tvn >= 1e8, 0.002, 0.004))
        self.T, self.S = self.C.shape
        print(f"{self.S} stocks ever liquid since {START_EVAL}; {self.T} days "
              f"{self.idx[0].date()} -> {self.idx[-1].date()}", flush=True)

    def hold(self, entry, exit_):
        b = lambda x: np.asarray(x.fillna(False), dtype=np.bool_)
        return nb_hold(b(entry), b(exit_), F(self.C))


# ============================================================ the three strategies that beat holding all stocks
def coppock(d: Data):
    """Coppock Curve (daily version of 14/11/10 months): long while it is rising vs a month ago."""
    c = wma(d.C.pct_change(294, fill_method=None) + d.C.pct_change(231, fill_method=None), 210)
    return c > c.shift(21)


def rolling_sharpe(d: Data):
    """BUY when the 126-day Sharpe ratio > 1; SELL when it drops below 0."""
    s = d.ret.rolling(126).mean() / d.ret.rolling(126).std() * np.sqrt(252)
    return d.hold(s > 1, s < 0)


def rolling_sortino(d: Data):
    """BUY when the 126-day Sortino ratio > 1.5; SELL when it drops below 0."""
    dn = np.sqrt((d.ret.clip(upper=0) ** 2).rolling(126).mean())
    s = d.ret.rolling(126).mean() / dn * np.sqrt(252)
    return d.hold(s > 1.5, s < 0)


STRATEGIES = [
    {"no": 50, "name": "Coppock Curve", "category": "Momentum",
     "rule": "Coppock (daily version of 14/11/10 months) rising vs a month ago", "f": coppock},
    {"no": 139, "name": "Rolling Sharpe Ratio", "category": "Statistical",
     "rule": "BUY: 126-day Sharpe > 1; SELL: Sharpe < 0", "f": rolling_sharpe},
    {"no": 140, "name": "Rolling Sortino Ratio", "category": "Statistical",
     "rule": "BUY: 126-day Sortino > 1.5; SELL: Sortino < 0", "f": rolling_sortino},
]


# ============================================================ backtest engine
def run(d: Data, sig) -> tuple[pd.Series, dict]:
    if isinstance(sig, pd.Series):                                     # market-wide rule (benchmark)
        s = sig.reindex(d.idx).fillna(False).to_numpy(bool)[:, None] & np.ones((1, d.S), bool)
    else:
        s = np.asarray(sig.fillna(False) if hasattr(sig, "fillna") else sig, dtype=bool)
    pos = s & d.elig
    pos[-2:] = False                                                    # no forward return available
    prev = np.vstack([np.zeros((1, d.S), bool), pos[:-1]])
    ent, ext = pos & ~prev, prev & ~pos
    n = pos.sum(1).astype(float)
    nprev = np.concatenate([[0.0], n[:-1]])
    gross = np.where(n > 0, (d.R * pos).sum(1) / np.maximum(n, 1), 0.0)
    cin = np.where(n > 0, ((BUY_COST + d.slip) * ent).sum(1) / np.maximum(n, 1), 0.0)
    cout = np.where(nprev > 0, ((SELL_COST + d.slip) * ext).sum(1) / np.maximum(nprev, 1), 0.0)
    dates = d.idx[1:]                                                   # return earned from the next open
    r = pd.Series((gross - cin - cout)[:-1], index=dates)
    r = r[r.index >= START_EVAL]
    rg = pd.Series(gross[:-1], index=dates)
    rg = rg[rg.index >= START_EVAL]

    # trades (open trades marked at the last price)
    P, E = pos.T.ravel(), ent.T.ravel()                                 # stock-major
    tid = np.cumsum(E) * P
    lr = np.log1p(np.clip(d.R.T.ravel(), -0.95, None)) * P
    k = tid.max() + 1
    sum_lr = np.bincount(tid, weights=lr, minlength=k)[1:]
    length = np.bincount(tid, minlength=k)[1:]
    net = np.exp(sum_lr) * (1 - BUY_COST - SELL_COST - 2 * d.slip.T.ravel()[E]) - 1
    yrs = (d.idx[-1] - pd.Timestamp(START_EVAL)).days / 365.25
    live = d.idx >= START_EVAL
    stats = {"trades": int(len(net)),
             "trades_per_year": round(float(len(net) / yrs), 1),
             "win_rate_pct": round(float((net > 0).mean() * 100), 1) if len(net) else np.nan,
             "avg_trade_pct": round(float(net.mean() * 100), 2) if len(net) else np.nan,
             "median_hold_days": float(np.median(length)) if len(net) else np.nan,
             "avg_stocks_held": round(float(n[live].mean()), 1),
             "pct_time_invested": round(float((n[live] > 0).mean() * 100), 1),
             "gross_cagr_pct": perf(rg).get("cagr_pct"), "gross_sharpe": perf(rg).get("sharpe"),
             "cost_drag_pct_per_year": round(float((cin + cout)[live].sum() / yrs * 100), 2)}
    return r, stats


def perf(r: pd.Series) -> dict:
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    sd = r.std()
    return {"cagr_pct": round(float((eq.iloc[-1] ** (1 / yrs) - 1) * 100), 1),
            "sharpe": round(float(r.mean() / sd * np.sqrt(252)), 2) if sd > 0 else 0.0,
            "max_dd_pct": round(float((eq / eq.cummax().clip(lower=1) - 1).min() * 100), 1),
            "vol_pct": round(float(sd * np.sqrt(252) * 100), 1)}


def periods(r: pd.Series) -> dict:
    out = {}
    for tag, rr in (("", r), ("is_", r[r.index.year < SPLIT]), ("oos_", r[r.index.year >= SPLIT])):
        out.update({tag + k: v for k, v in perf(rr).items()})
    return out


def main() -> int:
    t0 = datetime.now()
    d = Data()
    out = ROOT / "results" / f"indicators_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    rows, yearly, monthly = [], {}, {}
    r_ew, _st = run(d, pd.Series(True, index=d.idx))
    nr = d.nifty.pct_change().reindex(r_ew.index).fillna(0)
    for name, r in (("Hold all liquid stocks", r_ew), ("NIFTY 50", nr)):
        rows.append({"no": 0, "name": name, "category": "Benchmark", "rule": "", **periods(r)})
        yearly[name] = (1 + r).groupby(r.index.year).prod() - 1
        monthly[name] = (1 + r).cumprod().resample("ME").last()
    ew_y = yearly["Hold all liquid stocks"]

    for spec in STRATEGIES:
        r, st = run(d, spec["f"](d))
        y = (1 + r).groupby(r.index.year).prod() - 1
        p = periods(r)
        row = {k: spec[k] for k in ("no", "name", "category", "rule")} | p | st
        row["years_beat_hold_all"] = f"{int((y > ew_y.reindex(y.index)).sum())}/{len(y)}"
        rows.append(row)
        key = f"{spec['no']}. {spec['name']}"
        yearly[key] = y
        monthly[key] = (1 + r).cumprod().resample("ME").last()
        print(f"#{spec['no']:>3} {spec['name']:24s} CAGR {p['cagr_pct']:>5} Sharpe {p['sharpe']:>5} "
              f"DD {p['max_dd_pct']:>6} | win {st['win_rate_pct']}% | 2010-19 {p['is_cagr_pct']} "
              f"2020+ {p['oos_cagr_pct']} | beat hold-all {row['years_beat_hold_all']} yrs", flush=True)

    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    Y = pd.DataFrame(yearly).T.mul(100).round(1)
    Y.columns = [str(c) for c in Y.columns]
    Y.to_csv(out / "yearly.csv")
    pd.DataFrame(monthly).round(4).to_csv(out / "equity_monthly.csv")
    (out / "report.json").write_text(json.dumps({
        "as_of": str(d.idx[-1].date()), "start": START_EVAL, "split_year": SPLIT, "stocks": int(d.S),
        "costs": {"buy": BUY_COST, "sell": SELL_COST, "slippage": "0.1/0.2/0.4 % by liquidity"},
        "method": __doc__}, indent=1))
    print("results ->", out, f"({(datetime.now() - t0).seconds}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
