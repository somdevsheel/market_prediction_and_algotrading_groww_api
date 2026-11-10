"""Buy / Hold / Sell signals on top of the stock ML predictions (ml_lab/stocks_ml.py output).

Signals (all decided on information available at the time; trades at the next day's open):
  BUY   : model rank <= 10 at the month-end and a free slot
  HOLD  : held and model rank <= KEEP (KEEP = 10 / 20 / 30 / 50)
  SELL  : (a) model rank > KEEP or the stock left the top-100 universe       -> sold at the next open
          (b) trailing stop: close <= (1 - STOP) x highest close since buying -> sold next open (checked daily)
          (c) market sell: everything to cash when the market signal is OFF at a month-end
              trend = NIFTY close < 200-day average; ml = ridge+LightGBM forecast of NIFTY's next-month return < 0
Cash earns nothing (conservative). Same Groww charges + slippage as stocks_ml.py.

    python3 -m ml_lab.stock_signals [results_dir]
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from ml_lab.stocks_ml import CAPITAL, LAKE, ROOT, TOP_N, order_cost

MODELS = ["hist_gb", "xgboost", "extra_trees", "lightgbm", "gru", "ensemble", "momentum"]
KEEPS, STOPS, MARKETS = (10, 20, 30, 50), (None, 0.10, 0.15, 0.20), ("none", "trend", "ml")


def market_signals(nifty: pd.DataFrame, dates) -> dict[str, pd.Series]:
    c = nifty["close"]
    me = c.groupby(c.index.to_period("M")).tail(1)
    f = pd.DataFrame({"r21": c.pct_change(21), "r63": c.pct_change(63), "r126": c.pct_change(126), "r252": c.pct_change(252),
                      "vol21": c.pct_change().rolling(21).std() * np.sqrt(252),
                      "ma200": c / c.rolling(200).mean() - 1, "ma50": c / c.rolling(50).mean() - 1}).loc[me.index]
    y = me.shift(-1) / me - 1
    f, y = f.dropna(), y.reindex(f.dropna().index)
    pred = pd.Series(np.nan, index=f.index)
    for Y in range(2014, 2027):
        tr = (f.index < pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=35)) & y.notna()
        te = f.index.year == Y
        if te.sum() == 0:
            continue
        a = Ridge(alpha=1.0).fit(f[tr], y[tr]).predict(f[te])
        b = LGBMRegressor(n_estimators=200, num_leaves=4, learning_rate=0.03, min_child_samples=15, verbose=-1,
                          random_state=0).fit(f[tr], y[tr]).predict(f[te])
        pred[te] = (a + b) / 2
    dates = pd.DatetimeIndex(dates)
    trend = (c / c.rolling(200).mean()).reindex(dates, method="ffill") >= 1
    ml = pred.reindex(dates, method="ffill") >= 0
    return {"none": pd.Series(True, index=dates), "trend": trend, "ml": ml.fillna(True)}


def simulate(pr: pd.DataFrame, model: str, keep: int, stop, on: pd.Series, O: pd.DataFrame, C: pd.DataFrame):
    days = O.index
    dec = sorted(pr.date.unique())
    entry_of = {d: days[days.searchsorted(d, side="right")] for d in dec if days.searchsorted(d, side="right") < len(days)}
    rank_at = {d: g.set_index("sym")[model].rank(ascending=False) for d, g in pr.groupby("date")}
    rebal = {entry_of[d]: d for d in entry_of}
    start = min(rebal)
    cash, hold, peak, basis = CAPITAL, {}, {}, {}
    pending, closed, curve, charges = set(), [], [], 0.0

    def sell(s, px, why):
        nonlocal cash, charges
        val = hold[s] * px
        cost = order_cost(val, "sell")
        cash += val - cost
        charges += cost
        closed.append({"sym": s, "pnl": val - cost - basis.pop(s), "why": why})
        hold.pop(s)
        peak.pop(s, None)

    for t in days[days.searchsorted(start):]:
        o = O.loc[t]
        for s in list(pending):                                      # stop-loss exits decided yesterday
            if s in hold:
                sell(s, o[s], "stop")
        pending.clear()
        if t in rebal:
            d = rebal[t]
            rk = rank_at[d]
            if not on.get(d, True):
                for s in list(hold):
                    sell(s, o[s], "market")
            else:
                for s in list(hold):
                    if rk.get(s, 999) > keep:
                        sell(s, o[s], "model")
                equity = cash + sum(q * o[s] for s, q in hold.items())
                new = [s for s in rk.sort_values().index[:TOP_N] if s not in hold][: TOP_N - len(hold)]
                for i, s in enumerate(new):
                    px = o[s]
                    budget = min(equity / TOP_N, cash / (len(new) - i))
                    q = int(budget / (px * 1.003)) if np.isfinite(px) and px > 0 else 0
                    if q <= 0:
                        continue
                    val = q * px
                    cost = order_cost(val, "buy")
                    if val + cost > cash:
                        continue
                    cash -= val + cost
                    charges += cost
                    hold[s], basis[s], peak[s] = q, val + cost, px
        c = C.loc[t]
        for s in hold:
            peak[s] = max(peak[s], c[s])
            if stop is not None and c[s] <= peak[s] * (1 - stop):
                pending.add(s)
        curve.append((t, cash + sum(q * c[s] for s, q in hold.items())))
    eq = pd.Series(dict(curve))
    return eq, pd.DataFrame(closed), charges


def stats(eq: pd.Series, closed: pd.DataFrame, charges: float) -> dict:
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    r = eq.pct_change().dropna()
    h = eq[eq.index >= "2022-01-01"]
    hy = (h.index[-1] - h.index[0]).days / 365.25
    y = eq.groupby(eq.index.year).last()
    yr = pd.concat([pd.Series([CAPITAL], index=[eq.index[0].year - 1]), y]).pct_change().iloc[1:] * 100
    return {"cagr_pct": round(((eq.iloc[-1] / CAPITAL) ** (1 / yrs) - 1) * 100, 1),
            "max_dd_pct": round(float((eq / eq.cummax() - 1).min() * 100), 1),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2),
            "cagr_2022plus_pct": round(((h.iloc[-1] / h.iloc[0]) ** (1 / hy) - 1) * 100, 1),
            "losing_years": int((yr < 0).sum()),
            "win_rate_pct": round(float((closed.pnl > 0).mean() * 100), 1) if len(closed) else None,
            "trades": len(closed), "charges_rs": round(charges),
            "sells_model/stop/market": "/".join(str(int((closed.why == w).sum())) if len(closed) else "0"
                                                 for w in ("model", "stop", "market")),
            **{str(k): round(float(v), 1) for k, v in yr.items()}}


def main(rd: str | None = None) -> int:
    rd = Path(rd) if rd else sorted((ROOT / "results").glob("stocks_*/predictions.parquet"))[-1].parent
    pr = pd.read_parquet(rd / "predictions.parquet")
    d = pd.read_parquet(LAKE / "silver" / "yahoo_stocks" / "nifty500_daily.parquet")
    O = d.pivot(index="date", columns="sym", values="open").sort_index().ffill()
    C = d.pivot(index="date", columns="sym", values="close").sort_index().ffill()
    nifty = pd.read_parquet(LAKE / "silver" / "yahoo" / "NIFTY.parquet").set_index("date")
    mk = market_signals(nifty, sorted(pr.date.unique()))
    print({k: f"{(~v).mean() * 100:.0f}% of months in cash" for k, v in mk.items()}, flush=True)
    rows = []
    for m, keep, stop, mkt in itertools.product(MODELS, KEEPS, STOPS, MARKETS):
        eq, closed, ch = simulate(pr, m, keep, stop, mk[mkt], O, C)
        rows.append({"model": m, "sell_rank_gt": keep, "trailing_stop": stop or "-", "market_sell": mkt, **stats(eq, closed, ch)})
    res = pd.DataFrame(rows)
    res.to_csv(rd / "signals_grid.csv", index=False)
    # benchmark: buy NIFTY once and hold (price index)
    nc = nifty["close"].loc[C.index[C.index.searchsorted(pr.date.min(), side="right")]:]
    pd.set_option("display.width", 250)
    base = res[(res.sell_rank_gt == 20) & (res.trailing_stop == "-") & (res.market_sell == "none")]
    cols = ["model", "sell_rank_gt", "trailing_stop", "market_sell", "cagr_pct", "max_dd_pct", "sharpe", "cagr_2022plus_pct",
            "losing_years", "win_rate_pct", "trades", "sells_model/stop/market"]
    print("\nCURRENT RULE (sell when rank > 20, no stop, no market filter):")
    print(base[cols].to_string(index=False))
    print("\nEFFECT OF EACH SELL RULE (average over the 7 models):")
    for k in ("sell_rank_gt", "trailing_stop", "market_sell"):
        print(res.groupby(k)[["cagr_pct", "max_dd_pct", "sharpe", "cagr_2022plus_pct", "win_rate_pct"]].mean().round(2).to_string(), "\n")
    print("TOP 15 COMBINATIONS by Sharpe (of", len(res), "tested):")
    print(res.sort_values("sharpe", ascending=False)[cols].head(15).to_string(index=False))
    print(f"\nNIFTY 50 buy & hold: CAGR {((nc.iloc[-1] / nc.iloc[0]) ** (365.25 / (nc.index[-1] - nc.index[0]).days) - 1) * 100:.1f}%,"
          f" max DD {(nc / nc.cummax() - 1).min() * 100:.1f}%")
    print("results ->", rd / "signals_grid.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
