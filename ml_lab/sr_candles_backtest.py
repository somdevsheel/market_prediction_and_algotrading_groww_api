"""Port of EURUSD_SR_WITH_CANDLES_Backtesting.ipynb to every liquid Indian stock (daily bars, long-only).

Notebook strategy
  * support = swing low (lows falling for n1=2 bars into it, rising for n2=2 bars after), resistance = swing high,
    collected over the last backCandles=30 bars
  * BUY  signal: bullish engulfing or hammer ("star" with long lower wick) whose low/body touches the nearest support
  * SELL signal: bearish engulfing or shooting star at the nearest resistance (not traded: no overnight shorting of
    stocks in the cash market; it is reported only)
  * order at the next bar's open, stop-loss 0.075 below / take-profit 0.060 above the signal close (EURUSD prices)

Variants
  A  as written:  EURUSD thresholds converted to % of price (body >= 0.17 %, touch <= 0.125 %, SL 6.25 %, TP 5 %),
                  INCLUDING the notebook's look-ahead (swing points need the 2 bars AFTER them, and the notebook also
                  uses swing points of the 2 bars after the signal bar -> future data)
  B  fixed:       same thresholds, only swing points already confirmed at the signal close
  C  fixed + volatility-scaled: thresholds in ATR(14) units, matching their size on EURUSD
                  (body >= 0.25 ATR, touch <= 0.2 ATR, SL 9 ATR, TP 7.5 ATR)

Execution: entry at next open; SL/TP checked on each later bar's low/high (if both inside one bar -> stop first,
conservative); gap through a level -> filled at the open.  One position per stock.  Costs as in
ml_lab/indicator_backtest.py (Groww delivery + slippage).  Portfolio = equal weight across open positions.

    python3 -m ml_lab.sr_candles_backtest
"""
from __future__ import annotations

import json
from datetime import datetime

import numba as nb
import numpy as np
import pandas as pd

from ml_lab.indicator_backtest import BUY_COST, SELL_COST, START_EVAL, SPLIT, Data, F, perf, periods

N1, N2, BACK = 2, 2, 30


@nb.njit(cache=True, parallel=True)
def nb_signals(o, h, l, c, body_min, lim, lookahead):
    """Returns 2 = BUY (pattern at support), 1 = SELL (pattern at resistance), 0 = none.  body_min/lim are (T,S)."""
    T, S = c.shape
    sig = np.zeros((T, S), np.int8)
    for s in nb.prange(S):
        sup = np.zeros(T, np.bool_)
        res = np.zeros(T, np.bool_)
        for k in range(N1, T - N2):
            okS = okR = True
            for i in range(k - N1 + 1, k + 1):
                if l[i, s] > l[i - 1, s]:
                    okS = False
                if h[i, s] < h[i - 1, s]:
                    okR = False
            for i in range(k + 1, k + N2 + 1):
                if l[i, s] < l[i - 1, s]:
                    okS = False
                if h[i, s] > h[i - 1, s]:
                    okR = False
            sup[k], res[k] = okS, okR
        for t in range(BACK, T - N2):
            if np.isnan(c[t, s]) or np.isnan(c[t - 1, s]) or np.isnan(body_min[t, s]):
                continue
            bm = body_min[t, s]
            O, H, L, C, O1, C1 = o[t, s], h[t, s], l[t, s], c[t, s], o[t - 1, s], c[t - 1, s]
            body = max(abs(O - C), 1e-9)
            body1 = abs(O1 - C1)
            up_w, lo_w = H - max(O, C), min(O, C) - L
            eng_bull = body > bm and body1 > bm and O1 > C1 and O < C and O <= C1 and C > O1
            eng_bear = body > bm and body1 > bm and O1 < C1 and O > C and O >= C1 and C < O1
            hammer = lo_w / body > 1 and up_w < 0.2 * lo_w and body > bm
            shoot = up_w / body > 1 and lo_w < 0.2 * up_w and body > bm
            last = t if lookahead else t - N2            # notebook: swing points up to t (needs t+1, t+2)
            first = t - BACK + N1
            if eng_bull or hammer:
                best, bd = np.nan, 1e18
                for k in range(first, last + 1):
                    if sup[k] and abs(l[k, s] - L) < bd:
                        bd, best = abs(l[k, s] - L), l[k, s]
                if not np.isnan(best):
                    if (abs(L - best) <= lim[t, s] or abs(min(O, C) - best) <= lim[t, s]) and max(O, C) > best and H > best:
                        sig[t, s] = 2
                        continue
            if eng_bear or shoot:
                best, bd = np.nan, 1e18
                for k in range(first, last + 1):
                    if res[k] and abs(h[k, s] - H) < bd:
                        bd, best = abs(h[k, s] - H), h[k, s]
                if not np.isnan(best):
                    if (abs(H - best) <= lim[t, s] or abs(max(O, C) - best) <= lim[t, s]) and min(O, C) < best and L < best:
                        sig[t, s] = 1
    return sig


@nb.njit(cache=True, parallel=True)
def nb_trades(sig, elig, o, h, l, c, sl_d, tp_d):
    """Long trades from BUY signals. Returns daily position return matrix, held mask, entry/exit flags,
    and per-trade arrays (stock, entry idx, exit idx, gross return, reason 0=TP 1=SL 2=open at end)."""
    T, S = c.shape
    R = np.zeros((T, S))
    held = np.zeros((T, S), np.bool_)
    ent = np.zeros((T, S), np.bool_)
    ext = np.zeros((T, S), np.bool_)
    maxn = T // 2 + 1
    tr = np.full((S, maxn, 5), np.nan)
    cnt = np.zeros(S, np.int64)
    for s in nb.prange(S):
        t = 0
        while t < T - 2:
            if sig[t, s] == 2 and elig[t, s] and not np.isnan(o[t + 1, s]):
                e = o[t + 1, s]
                stop, tgt = c[t, s] - sl_d[t, s], c[t, s] + tp_d[t, s]
                prev = e
                j = t + 1
                reason, px = 2.0, np.nan
                while j < T:
                    if np.isnan(c[j, s]):
                        j += 1
                        continue
                    held[j, s] = True
                    if j == t + 1:
                        ent[j, s] = True
                    op = o[j, s] if j > t + 1 else e
                    if op <= stop:
                        px, reason = op, 1.0
                    elif op >= tgt:
                        px, reason = op, 0.0
                    elif l[j, s] <= stop:
                        px, reason = stop, 1.0
                    elif h[j, s] >= tgt:
                        px, reason = tgt, 0.0
                    if not np.isnan(px):
                        R[j, s] = px / prev - 1
                        ext[j, s] = True
                        break
                    R[j, s] = c[j, s] / prev - 1
                    prev = c[j, s]
                    j += 1
                if j >= T:
                    j = T - 1
                    px = prev
                k = cnt[s]
                tr[s, k, 0], tr[s, k, 1], tr[s, k, 2] = s, t + 1, j
                tr[s, k, 3], tr[s, k, 4] = px / e - 1, reason
                cnt[s] += 1
                t = j + 1                                  # next signal after the exit bar
            else:
                t += 1
    return R, held, ent, ext, tr, cnt


def run_variant(d: Data, name: str, lookahead: bool, scale: str) -> tuple[pd.Series, dict, pd.DataFrame]:
    C = F(d.C)
    if scale == "pct":
        bm, lim, sl, tp = 0.0017 * C, 0.00125 * C, 0.0625 * C, 0.05 * C
    else:
        a = F(d.atr(14))
        bm, lim, sl, tp = 0.25 * a, 0.2 * a, 9.0 * a, 7.5 * a
    sig = nb_signals(F(d.O), F(d.H), F(d.L), C, bm, lim, lookahead)
    elig = d.elig.copy()
    R, held, ent, ext, tr, cnt = nb_trades(sig, elig, F(d.O), F(d.H), F(d.L), C, sl, tp)
    n = held.sum(1).astype(float)
    nprev = np.concatenate([[0.0], n[:-1]])
    slip = d.slip
    gross = np.where(n > 0, (R * held).sum(1) / np.maximum(n, 1), 0.0)
    cin = np.where(n > 0, ((BUY_COST + slip) * ent).sum(1) / np.maximum(n, 1), 0.0)
    # exit cost charged on the exit day (the position is still in that day's denominator)
    cout = np.where(n > 0, ((SELL_COST + slip) * ext).sum(1) / np.maximum(n, 1), 0.0)
    r = pd.Series(gross - cin - cout, index=d.idx)
    r = r[r.index >= START_EVAL]
    rg = pd.Series(gross, index=d.idx)
    rg = rg[rg.index >= START_EVAL]

    rows = np.concatenate([tr[s, :cnt[s]] for s in range(d.S) if cnt[s]])
    T = pd.DataFrame(rows, columns=["s", "entry_i", "exit_i", "gross", "reason"])
    T = T[d.idx[T.entry_i.astype(int)] >= pd.Timestamp(START_EVAL)]
    T["sym"] = d.C.columns[T.s.astype(int)]
    T["entry"] = d.idx[T.entry_i.astype(int)]
    T["exit"] = d.idx[T.exit_i.astype(int)]
    sl_in = slip[T.entry_i.astype(int), T.s.astype(int)]
    T["net"] = (1 + T.gross) * (1 - BUY_COST - SELL_COST - 2 * sl_in) - 1
    T["days"] = T.exit_i - T.entry_i + 1
    T["result"] = T.reason.map({0.0: "take-profit", 1.0: "stop-loss", 2.0: "still open"})
    yrs = (d.idx[-1] - pd.Timestamp(START_EVAL)).days / 365.25
    buys = int(((sig == 2) & d.elig).sum())
    sells = int(((sig == 1) & d.elig).sum())
    # signal quality: average forward returns after BUY signals vs all eligible stock-days
    fw = {}
    for k in (5, 10, 20):
        f = (d.C.shift(-k) / d.O.shift(-1) - 1).to_numpy()
        m_all = d.elig & np.isfinite(f)
        m_b = m_all & (sig == 2)
        m_s = m_all & (sig == 1)
        fw[f"fwd{k}_after_buy_pct"] = round(float(np.nanmean(f[m_b]) * 100), 2)
        fw[f"fwd{k}_after_sell_pct"] = round(float(np.nanmean(f[m_s]) * 100), 2)
        fw[f"fwd{k}_any_day_pct"] = round(float(np.nanmean(f[m_all]) * 100), 2)
    st = {"variant": name, "buy_signals": buys, "sell_signals": sells, "trades": len(T),
          "trades_per_year": round(len(T) / yrs, 1),
          "win_rate_pct": round(float((T.net > 0).mean() * 100), 1),
          "avg_trade_pct": round(float(T.net.mean() * 100), 2),
          "take_profit_pct_of_trades": round(float((T.reason == 0).mean() * 100), 1),
          "stop_loss_pct_of_trades": round(float((T.reason == 1).mean() * 100), 1),
          "median_hold_days": float(T.days.median()),
          "avg_stocks_held": round(float(n[d.idx >= START_EVAL].mean()), 1),
          "pct_time_invested": round(float((n[d.idx >= START_EVAL] > 0).mean() * 100), 1),
          "gross_cagr_pct": perf(rg).get("cagr_pct"), **periods(r), **fw}
    return r, st, T


def main() -> int:
    t0 = datetime.now()
    d = Data()
    from ml_lab.indicator_backtest import run
    from ml_lab.stocks_bhs import ROOT
    out = ROOT / "results" / f"sr_candles_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    r_ew, _ = run(d, pd.Series(True, index=d.idx))
    nr = d.nifty.pct_change().reindex(r_ew.index).fillna(0)
    rows = [{"variant": "Hold all liquid stocks", **periods(r_ew)}, {"variant": "NIFTY 50", **periods(nr)}]
    yearly = {"Hold all liquid stocks": (1 + r_ew).groupby(r_ew.index.year).prod() - 1,
              "NIFTY 50": (1 + nr).groupby(nr.index.year).prod() - 1}
    for name, la, sc in (("A. as written (with look-ahead)", True, "pct"),
                         ("B. fixed (no look-ahead)", False, "pct"),
                         ("C. fixed + volatility-scaled (ATR)", False, "atr")):
        r, st, T = run_variant(d, name, la, sc)
        rows.append(st)
        yearly[name] = (1 + r).groupby(r.index.year).prod() - 1
        T.drop(columns=["s", "entry_i", "exit_i", "reason"]).to_csv(out / f"trades_{name[0]}.csv", index=False)
        print(json.dumps(st), flush=True)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    Y = (pd.DataFrame(yearly) * 100).round(1)
    Y.to_csv(out / "yearly.csv")
    print(Y.to_string())
    print("results ->", out, f"({(datetime.now() - t0).seconds}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
