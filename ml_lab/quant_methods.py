"""Classic quant methods on Indian stocks (long-only, delivery), same universe/costs as the other stock backtests.

1. Mean reversion (short-term reversal): every H trading days buy the N liquid stocks that fell most over the last 5
   days, optionally only when NIFTY is above its 200-day average and/or only stocks above their own 200-day average.
2. Long-only pairs trading (statistical arbitrage without shorting): every 6 months pick the most correlated stock
   pairs from the previous year's daily returns (banks with banks, IT with IT ... found from the data, no look-ahead);
   when one stock of a pair falls > Z standard deviations behind its partner (60-day log price ratio), buy the laggard;
   sell when the gap closes (|z| < 0.5), after 20 days, or if the gap widens past 4 (stop).
3. Risk sizing on the ML top-10 portfolio (XGBoost BUY/HOLD/SELL, monthly, rebuilt day by day from the saved
   out-of-sample predictions): (a) risk parity = weights proportional to 1 / volatility, (b) volatility targeting =
   invest less when the portfolio's recent volatility is above 15 %/yr (no leverage), (c) both.
4. Monte Carlo: 10,000 block-bootstrap simulations (21-day blocks) of each final strategy's daily returns ->
   range of 1-year and 3-year outcomes, chance of loss, drawdowns.

Variants: each method has a small grid; the variant is CHOSEN on 2010-2019 (2013-2019 for ML) only, 2020 -> today is
the unseen test.  Weights are decided at the close and traded at the next open; costs on every weight change (Groww
delivery buy 0.24 % / sell 0.38 % + slippage 0.1-0.4 % by liquidity).  Survivorship bias as in the other backtests.

    python3 -m ml_lab.quant_methods
"""
from __future__ import annotations

import glob
import json
from datetime import datetime

import numpy as np
import pandas as pd

from ml_lab.indicator_backtest import BUY_COST, SELL_COST, START_EVAL, Data, perf
from ml_lab.stocks_bhs import ROOT

IS_END = 2019
CAPITAL_EXAMPLE = 1_000_000           # Rs 10 lakh for the Monte Carlo money examples


# ------------------------------------------------------------------ engine
def run_weights(d: Data, W: np.ndarray) -> tuple[pd.Series, dict]:
    """W[t, s] = weight decided at the close of day t (held from open t+1 to open t+2). Cash = 1 - sum(W)."""
    W = np.nan_to_num(W.astype(float))
    W[-2:] = 0.0
    prev = np.vstack([np.zeros((1, d.S)), W[:-1]])
    dw = W - prev
    cost = (np.clip(dw, 0, None) * (BUY_COST + d.slip)).sum(1) + (np.clip(-dw, 0, None) * (SELL_COST + d.slip)).sum(1)
    gross = (W * d.R).sum(1)
    r = pd.Series((gross - cost)[:-1], index=d.idx[1:])
    g = pd.Series(gross[:-1], index=d.idx[1:])
    live = r.index >= START_EVAL
    yrs = max((r.index[-1] - pd.Timestamp(START_EVAL)).days / 365.25, 1e-9)
    st = {"avg_invested_pct": round(float(W[:-1].sum(1)[live].mean() * 100), 1),
          "avg_stocks_held": round(float((W[:-1] > 0).sum(1)[live].mean()), 1),
          "turnover_per_year_x": round(float(np.abs(dw[:-1]).sum(1)[live].sum() / yrs / 2), 1),
          "cost_drag_pct_per_year": round(float(cost[:-1][live].sum() / yrs * 100), 2),
          "gross_cagr_pct": perf(g[live]).get("cagr_pct")}
    return r[live], st


def windows(r: pd.Series) -> dict:
    out = {}
    for tag, m in (("", slice(None)), ("is_", r.index.year <= IS_END), ("oos_", r.index.year > IS_END),
                   ("c13_", r.index.year >= 2013)):
        rr = r[m] if not isinstance(m, slice) else r
        out.update({tag + k: v for k, v in perf(rr).items()})
    return out


# ------------------------------------------------------------------ 1. mean reversion
def mean_reversion(d: Data, n: int, hold: int, mkt_filter: bool, stock_trend: bool) -> np.ndarray:
    C = d.C.to_numpy()
    r5 = C / np.roll(C, 5, axis=0) - 1
    r5[:5] = np.nan
    ma200 = d.C.rolling(200, min_periods=200).mean().to_numpy()
    nma = d.nifty.rolling(200).mean().to_numpy()
    mkt_up = d.nifty.to_numpy() > nma
    W = np.zeros((d.T, d.S))
    start = int(np.searchsorted(d.idx, pd.Timestamp(START_EVAL)))
    for t in range(start, d.T - 2, hold):
        if mkt_filter and not mkt_up[t]:
            continue
        ok = d.elig[t] & np.isfinite(r5[t])
        if stock_trend:
            ok &= C[t] > ma200[t]
        cand = np.where(ok)[0]
        if len(cand) < n:
            continue
        pick = cand[np.argsort(r5[t, cand])[:n]]
        W[t:t + hold, pick] = 1.0 / n
    return W


# ------------------------------------------------------------------ 2. long-only pairs
def pair_sets(d: Data, every: int = 126, lookback: int = 252, min_corr: float = 0.7, max_pairs: int = 150):
    """Point-in-time pair lists: [(t_from, t_to, [(a, b), ...]), ...] chosen from the previous `lookback` days."""
    ret = d.ret.to_numpy()
    start = int(np.searchsorted(d.idx, pd.Timestamp(START_EVAL)))
    out = []
    for t in range(start, d.T, every):
        win = ret[t - lookback + 1:t + 1]
        ok = d.elig[t] & np.isfinite(win).all(0)
        cols = np.where(ok)[0]
        if len(cols) < 20:
            continue
        x = win[:, cols]
        x = (x - x.mean(0)) / x.std(0)
        cm = (x.T @ x) / len(x)
        iu = np.triu_indices(len(cols), 1)
        cv = cm[iu]
        order = np.argsort(-cv)
        used: dict[int, int] = {}
        pairs = []
        for k in order:
            if cv[k] < min_corr or len(pairs) >= max_pairs:
                break
            a, b = cols[iu[0][k]], cols[iu[1][k]]
            if used.get(a, 0) >= 2 or used.get(b, 0) >= 2:
                continue
            used[a] = used.get(a, 0) + 1
            used[b] = used.get(b, 0) + 1
            pairs.append((a, b))
        out.append((t, min(t + every, d.T), pairs))
    return out


def pairs_long_only(d: Data, sets, z_in: float, slots: int, z_out: float = 0.5, z_stop: float = 4.0,
                    max_hold: int = 20) -> tuple[np.ndarray, list]:
    lc = np.log(d.C.to_numpy())
    W = np.zeros((d.T, d.S))
    trades = []
    held: dict[int, tuple] = {}                         # stock -> (pair index in period, sign, entry t)
    for t0, t1, pairs in sets:
        if not pairs:
            continue
        a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
        lo = max(0, t0 - 80)
        x = lc[lo:t1, a] - lc[lo:t1, b]
        xs = pd.DataFrame(x)
        z = ((xs - xs.rolling(60, min_periods=60).mean()) / xs.rolling(60, min_periods=60).std()).to_numpy()
        for s in list(held):                           # positions from the previous period: close (pair list changed)
            trades.append((s, held[s][2], t0))
            held.pop(s)
        for t in range(t0, min(t1, d.T - 2)):
            zt = z[t - lo]
            for s in list(held):
                k, sign, te = held[s]
                zo = sign * zt[k]                       # oriented: negative = laggard still behind
                if not np.isfinite(zo) or zo > -z_out or zo < -z_stop or t - te >= max_hold:
                    trades.append((s, te, t))
                    held.pop(s)
            free = slots - len(held)
            if free > 0:
                cand = []
                for k in np.where(np.isfinite(zt) & (np.abs(zt) > z_in))[0]:
                    s, sign = (a[k], 1.0) if zt[k] < 0 else (b[k], -1.0)   # z < 0: a is behind b -> buy a
                    if s not in held and d.elig[t, s] and abs(zt[k]) < z_stop:
                        cand.append((-abs(zt[k]), s, k, sign))
                for _, s, k, sign in sorted(cand)[:free]:
                    if s not in held:
                        held[s] = (k, sign, t)
            for s in held:
                W[t, s] = 1.0 / slots
    return W, trades


# ------------------------------------------------------------------ 3. ML portfolio + risk sizing
def ml_portfolio(d: Data, model: str = "xgboost", slots: int = 10, sizing: str = "equal") -> np.ndarray:
    run = sorted(p for p in (ROOT / "results").glob("bhs_*") if (p / "summary.csv").exists())[-1]
    P = pd.concat([pd.read_parquet(f, columns=["date", "sym", "entry", "exit", f"{model}_p0", f"{model}_p1", f"{model}_p2"])
                   for f in sorted(glob.glob(str(run / "pred_*.parquet")))])
    P["sig"] = P[[f"{model}_p0", f"{model}_p1", f"{model}_p2"]].to_numpy().argmax(1)
    P["score"] = P[f"{model}_p2"] - P[f"{model}_p0"]
    col = {s: i for i, s in enumerate(d.C.columns)}
    pos = {t: i for i, t in enumerate(d.idx)}
    vol60 = d.ret.rolling(60, min_periods=40).std().to_numpy()
    W = np.zeros((d.T, d.S))
    hold: set[str] = set()
    for dt, g in P.groupby("date", sort=True):
        e = pos.get(g.entry.iloc[0])
        x = pos.get(g.exit.iloc[0], d.T) if pd.notna(g.exit.iloc[0]) else d.T
        if e is None:
            continue
        sig = dict(zip(g.sym, g.sig))
        hold = {s for s in hold if sig.get(s, 0) != 0}
        for s in g[(g.sig == 2) & ~g.sym.isin(list(hold))].sort_values("score", ascending=False).sym:
            if len(hold) >= slots:
                break
            hold.add(s)
        idx = [col[s] for s in hold if s in col]
        if not idx:
            continue
        t_sig = max(e - 1, 0)
        if sizing == "risk_parity":
            iv = 1.0 / np.where(np.isfinite(vol60[t_sig, idx]) & (vol60[t_sig, idx] > 0), vol60[t_sig, idx], np.nanmedian(vol60[t_sig, idx]))
            w = iv / iv.sum() * len(idx) / slots
        else:
            w = np.full(len(idx), 1.0 / slots)
        W[max(e - 1, 0):max(x - 1, e), idx] = w
    return W


def vol_target(d: Data, W: np.ndarray, target: float = 0.15, every: int = 5, floor: float = 0.25) -> np.ndarray:
    gross = (np.nan_to_num(W) * d.R).sum(1)
    known = pd.Series(gross).shift(2)                   # return of the position held 2 days ago is known at today's close
    vol = known.rolling(63, min_periods=40).std().to_numpy() * np.sqrt(252)
    k = np.ones(d.T)
    cur = 1.0
    for t in range(d.T):
        if t % every == 0 and np.isfinite(vol[t]) and vol[t] > 0:
            new = float(np.clip(target / vol[t], floor, 1.0))
            if abs(new - cur) > 0.1 or new in (floor, 1.0):
                cur = new
        k[t] = cur
    return W * k[:, None]


# ------------------------------------------------------------------ 4. Monte Carlo
def monte_carlo(r: pd.Series, years: int, n: int = 10_000, block: int = 21, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    x = r.to_numpy()
    L = 252 * years
    nb = int(np.ceil(L / block))
    starts = rng.integers(0, len(x) - block, size=(n, nb))
    paths = x[(starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :L]]
    eq = np.cumprod(1 + paths, axis=1)
    tot = eq[:, -1] - 1
    dd = (eq / np.maximum.accumulate(np.maximum(eq, 1.0), axis=1) - 1).min(1)
    ann = (1 + tot) ** (1 / years) - 1
    q = lambda v, p: round(float(np.percentile(v, p) * 100), 1)  # noqa: E731
    return {"years": years, "median_total_pct": q(tot, 50), "p5_total_pct": q(tot, 5), "p95_total_pct": q(tot, 95),
            "median_per_year_pct": q(ann, 50), "p5_per_year_pct": q(ann, 5), "p95_per_year_pct": q(ann, 95),
            "chance_of_loss_pct": round(float((tot < 0).mean() * 100), 1),
            "chance_loss_gt_20pct": round(float((tot < -0.2).mean() * 100), 1),
            "median_worst_drop_pct": q(dd, 50), "bad_case_worst_drop_pct": q(dd, 5),
            "rs10L_median": round(CAPITAL_EXAMPLE * (1 + float(np.percentile(tot, 50)))),
            "rs10L_bad_5pct": round(CAPITAL_EXAMPLE * (1 + float(np.percentile(tot, 5)))),
            "rs10L_good_95pct": round(CAPITAL_EXAMPLE * (1 + float(np.percentile(tot, 95))))}


# ------------------------------------------------------------------ main
def main() -> int:
    t0 = datetime.now()
    d = Data()
    out = ROOT / "results" / f"quant_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    rows, series = [], {}

    def add(method, variant, W, extra=None):
        r, st = run_weights(d, W)
        series[f"{method} | {variant}"] = r
        rows.append({"method": method, "variant": variant, **windows(r), **st, **(extra or {})})
        w = rows[-1]
        print(f"{method:24s} {variant:44s} CAGR {w.get('cagr_pct')!s:>6} Sharpe {w.get('sharpe')!s:>5} DD {w.get('max_dd_pct')!s:>6} "
              f"| IS {w.get('is_sharpe')} OOS {w.get('oos_sharpe')} ({w.get('oos_cagr_pct')}%/yr) cost {st['cost_drag_pct_per_year']}%/yr", flush=True)

    # benchmarks
    add("Benchmark", "Hold all liquid stocks", d.elig / np.maximum(d.elig.sum(1, keepdims=True), 1))
    nr = d.nifty.pct_change().reindex(d.idx[1:]).fillna(0)
    nr = nr[nr.index >= START_EVAL]
    series["Benchmark | NIFTY 50"] = nr
    rows.append({"method": "Benchmark", "variant": "NIFTY 50", **windows(nr)})

    # 1. mean reversion grid
    for n in (10, 20):
        for hold in (5, 10, 21):
            for mf in (False, True):
                for stf in (False, True):
                    v = f"N={n} hold={hold}d market-filter={'on' if mf else 'off'} stock>200DMA={'on' if stf else 'off'}"
                    add("1. Mean reversion", v, mean_reversion(d, n, hold, mf, stf))

    # 2. long-only pairs grid
    sets = pair_sets(d)
    ex = [(d.C.columns[a], d.C.columns[b]) for a, b in sets[-1][2][:12]]
    print("example pairs (latest period):", ex, flush=True)
    for z_in in (2.0, 2.5):
        for slots in (10, 20):
            W, tr = pairs_long_only(d, sets, z_in, slots)
            add("2. Pairs (long-only)", f"entry z={z_in} slots={slots}", W, {"trades": len(tr)})

    # 3. ML portfolio + risk sizing
    base = ml_portfolio(d, sizing="equal")
    rp = ml_portfolio(d, sizing="risk_parity")
    add("3. ML portfolio sizing", "equal weight (as before)", base)
    add("3. ML portfolio sizing", "risk parity (1/volatility)", rp)
    add("3. ML portfolio sizing", "vol target 15%/yr", vol_target(d, base))
    add("3. ML portfolio sizing", "risk parity + vol target 15%/yr", vol_target(d, rp))

    S = pd.DataFrame(rows)
    # choose each method's variant on the in-sample years only
    chosen = {}
    for m, g in S[S.method.str[0].isin(list("123"))].groupby("method"):
        col = "is_sharpe" if not m.startswith("3") else "c13_sharpe"
        if m.startswith("3"):                          # ML preds start 2013: choose on 2013-2019
            col = "_sel"
            g = g.assign(_sel=[perf(series[f"{m} | {v}"][(series[f'{m} | {v}'].index.year >= 2013) &
                                                       (series[f'{m} | {v}'].index.year <= IS_END)]).get("sharpe") for v in g.variant])
        chosen[m] = g.sort_values(col, ascending=False).variant.iloc[0]
    S["chosen_on_in_sample"] = [chosen.get(m) == v for m, v in zip(S.method, S.variant)]
    S.to_csv(out / "summary.csv", index=False)
    print("\nchosen on in-sample years:", json.dumps(chosen, indent=1))

    # yearly + Monte Carlo for the chosen variants and benchmarks
    final = ["Benchmark | Hold all liquid stocks", "Benchmark | NIFTY 50"] + [f"{m} | {v}" for m, v in chosen.items()]
    Y = pd.DataFrame({k: (1 + series[k]).groupby(series[k].index.year).prod() - 1 for k in final}).mul(100).round(1)
    Y.to_csv(out / "yearly.csv")
    print(Y.to_string())
    mc = []
    for k in final:
        r = series[k]
        r = r[r.index.year >= 2013]                    # common period for a fair comparison
        for yrs in (1, 3):
            mc.append({"strategy": k, **monte_carlo(r, yrs)})
    MC = pd.DataFrame(mc)
    MC.to_csv(out / "montecarlo.csv", index=False)
    print(MC.to_string(index=False))
    pd.DataFrame({k: (1 + v).cumprod() for k, v in series.items() if k in final}).resample("ME").last().round(4).to_csv(out / "equity_monthly.csv")
    (out / "report.json").write_text(json.dumps({"as_of": str(d.idx[-1].date()), "chosen": chosen, "method": __doc__}, indent=1))
    print("results ->", out, f"({(datetime.now() - t0).seconds // 60} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
