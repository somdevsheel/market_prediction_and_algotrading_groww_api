"""Backtest 140 technical indicators one by one on every liquid Indian stock (daily bars, long-only).

Each indicator gets ONE textbook rule with its standard settings (no parameter tuning -> no curve fitting):
  BUY state  -> hold the stock,   SELL/flat state -> no position.
The rule is evaluated at the close of day t; the trade happens at the open of day t+1.

Universe (point-in-time, same as ml_lab/stocks_bhs.py): Yahoo daily data from 2009, price >= Rs 20, median traded value
>= Rs 2 crore/day over 126 days, >= ~1 year of history, no >50 % one-day data jump in the last 21 days.

Portfolio per indicator: equal weight across every stock currently in a BUY state (cash when none).
Costs (Groww delivery, Rs ~10,000 per position): buy 0.24 %, sell 0.38 % (incl. Rs 15.93 DP charge), plus slippage
0.1 / 0.2 / 0.4 % per side by liquidity.  Daily re-balancing friction between held stocks is ignored.

Benchmarks: "Hold all liquid stocks" (same universe, always invested, same costs) and NIFTY 50.
Periods: full 2010 -> today, 2010-2019 and 2020 -> today (nothing was fitted, so both halves are out-of-sample).

Market-breadth indicators (93-104) are market-timing rules: when bullish, hold all liquid stocks; otherwise cash.
Indicators that need intraday/tick data or are drawing tools are run with a clearly-labelled daily-bar proxy.

    python3 -m ml_lab.indicator_backtest            # all 140
    python3 -m ml_lab.indicator_backtest --only 1,12,26
"""
from __future__ import annotations

import argparse
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
def nb_hold(entry, exit_, max_hold, stop, target, trail, close):
    """Generic position state machine. stop/target are captured at entry; trail = distance below the highest close."""
    T, S = entry.shape
    out = np.zeros((T, S), np.bool_)
    for s in nb.prange(S):
        on, age, st, tg, peak = False, 0, np.nan, np.nan, 0.0
        for t in range(T):
            c = close[t, s]
            if on:
                age += 1
                if c > peak:
                    peak = c
                ex = exit_[t, s]
                if max_hold > 0 and age >= max_hold:
                    ex = True
                if not np.isnan(st) and c < st:
                    ex = True
                if not np.isnan(tg) and c >= tg:
                    ex = True
                if not np.isnan(trail[t, s]) and c < peak - trail[t, s]:
                    ex = True
                if ex:
                    on = False
            elif entry[t, s] and not np.isnan(c):
                on, age, st, tg, peak = True, 0, stop[t, s], target[t, s], c
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


@nb.njit(cache=True, parallel=True)
def nb_linreg(x, n):
    """Rolling least-squares line over n bars -> (slope, end value, r^2)."""
    T, S = x.shape
    slope, endv, r2 = np.full((T, S), np.nan), np.full((T, S), np.nan), np.full((T, S), np.nan)
    si = n * (n - 1) / 2.0
    sii = (n - 1) * n * (2 * n - 1) / 6.0
    den = n * sii - si * si
    for s in nb.prange(S):
        for t in range(n - 1, T):
            sx = sxx = six = 0.0
            ok = True
            for k in range(n):
                v = x[t - n + 1 + k, s]
                if np.isnan(v):
                    ok = False
                    break
                sx += v
                sxx += v * v
                six += k * v
            if not ok:
                continue
            b = (n * six - si * sx) / den
            a = (sx - b * si) / n
            slope[t, s], endv[t, s] = b, a + b * (n - 1)
            vx = n * sxx - sx * sx
            if vx > 0:
                r = (n * six - si * sx) / np.sqrt(den * vx)
                r2[t, s] = r * r
    return slope, endv, r2


@nb.njit(cache=True, parallel=True)
def nb_band_trend(lower, upper, close):
    """Supertrend/Chandelier skeleton: ratcheting long stop (lower) and short stop (upper). Returns +1/-1 trend."""
    T, S = close.shape
    out = np.zeros((T, S))
    for s in nb.prange(S):
        lo = hi = np.nan
        d = 1.0
        for t in range(T):
            c, l0, u0 = close[t, s], lower[t, s], upper[t, s]
            if np.isnan(c) or np.isnan(l0) or np.isnan(u0):
                out[t, s] = 0.0 if np.isnan(lo) else d
                continue
            if np.isnan(lo):
                lo, hi = l0, u0
                out[t, s] = d
                continue
            if d == 1.0 and c < lo:
                d = -1.0
            elif d == -1.0 and c > hi:
                d = 1.0
            lo = max(l0, lo) if (d == 1.0 and c > lo) else l0
            hi = min(u0, hi) if (d == -1.0 and c < hi) else u0
            out[t, s] = d
    return out


@nb.njit(cache=True, parallel=True)
def nb_volstop(close, atr, mult):
    T, S = close.shape
    out = np.zeros((T, S))
    for s in nb.prange(S):
        up = True
        ext = stop = np.nan
        for t in range(T):
            c, a = close[t, s], atr[t, s]
            if np.isnan(c) or np.isnan(a):
                out[t, s] = 0.0 if np.isnan(stop) else (1.0 if up else -1.0)
                continue
            if np.isnan(ext):
                ext, stop = c, c - mult * a
            if up:
                ext = max(ext, c)
                stop = max(stop, ext - mult * a)
                if c < stop:
                    up, ext, stop = False, c, c + mult * a
            else:
                ext = min(ext, c)
                stop = min(stop, ext + mult * a)
                if c > stop:
                    up, ext, stop = True, c, c - mult * a
            out[t, s] = 1.0 if up else -1.0
    return out


@nb.njit(cache=True, parallel=True)
def nb_psar(high, low, af0, afmax):
    T, S = high.shape
    out = np.zeros((T, S))
    for s in nb.prange(S):
        up = True
        sar = ep = np.nan
        af = af0
        ph = pl = np.nan
        for t in range(T):
            h, l = high[t, s], low[t, s]
            if np.isnan(h) or np.isnan(l):
                continue
            if np.isnan(sar):
                sar, ep, ph, pl = l, h, h, l
                out[t, s] = 1.0
                continue
            sar = sar + af * (ep - sar)
            if up:
                sar = min(sar, pl, low[t - 1, s] if t > 0 and not np.isnan(low[t - 1, s]) else pl)
                if l < sar:
                    up, sar, ep, af = False, ep, l, af0
                elif h > ep:
                    ep, af = h, min(af + af0, afmax)
            else:
                sar = max(sar, ph, high[t - 1, s] if t > 0 and not np.isnan(high[t - 1, s]) else ph)
                if h > sar:
                    up, sar, ep, af = True, ep, h, af0
                elif l < ep:
                    ep, af = l, min(af + af0, afmax)
            ph, pl = h, l
            out[t, s] = 1.0 if up else -1.0
    return out


@nb.njit(cache=True, parallel=True)
def nb_adaptive(x, alpha):
    """y[t] = y[t-1] + alpha[t] * (x[t] - y[t-1])  (KAMA)."""
    T, S = x.shape
    out = np.full((T, S), np.nan)
    for s in nb.prange(S):
        y = np.nan
        for t in range(T):
            v, a = x[t, s], alpha[t, s]
            if np.isnan(v):
                out[t, s] = y
                continue
            if np.isnan(y):
                y = v
            elif not np.isnan(a):
                y = y + a * (v - y)
            out[t, s] = y
    return out


@nb.njit(cache=True, parallel=True)
def nb_mcginley(x, n):
    T, S = x.shape
    out = np.full((T, S), np.nan)
    for s in nb.prange(S):
        y = np.nan
        for t in range(T):
            v = x[t, s]
            if np.isnan(v):
                out[t, s] = y
                continue
            if np.isnan(y):
                y = v
            else:
                y = y + (v - y) / (0.6 * n * (v / y) ** 4)
            out[t, s] = y
    return out


@nb.njit(cache=True, parallel=True)
def nb_fisher(x):
    """x = position of hl2 in its 9-bar range scaled to [-1, 1]."""
    T, S = x.shape
    fish, trig = np.full((T, S), np.nan), np.full((T, S), np.nan)
    for s in nb.prange(S):
        v = f = 0.0
        prev = np.nan
        for t in range(T):
            if np.isnan(x[t, s]):
                continue
            v = 0.66 * x[t, s] + 0.67 * v
            v = min(max(v, -0.999), 0.999)
            f = 0.5 * np.log((1 + v) / (1 - v)) + 0.5 * f
            fish[t, s], trig[t, s] = f, prev
            prev = f
    return fish, trig


@nb.njit(cache=True, parallel=True)
def nb_streak(close):
    T, S = close.shape
    out = np.zeros((T, S))
    for s in nb.prange(S):
        k = 0.0
        for t in range(1, T):
            a, b = close[t, s], close[t - 1, s]
            if np.isnan(a) or np.isnan(b):
                k = 0.0
            elif a > b:
                k = k + 1 if k > 0 else 1.0
            elif a < b:
                k = k - 1 if k < 0 else -1.0
            else:
                k = 0.0
            out[t, s] = k
    return out


@nb.njit(cache=True, parallel=True)
def nb_pctrank(x, n):
    T, S = x.shape
    out = np.full((T, S), np.nan)
    for s in nb.prange(S):
        for t in range(n, T):
            v = x[t, s]
            if np.isnan(v):
                continue
            c = m = 0
            for k in range(t - n, t):
                if not np.isnan(x[k, s]):
                    m += 1
                    if x[k, s] < v:
                        c += 1
            if m > n // 2:
                out[t, s] = 100.0 * c / m
    return out


@nb.njit(cache=True, parallel=True)
def nb_since_ext(x, n, is_max):
    """Bars since the highest (lowest) value within the last n+1 bars."""
    T, S = x.shape
    out = np.full((T, S), np.nan)
    for s in nb.prange(S):
        for t in range(n, T):
            best, bi = np.nan, -1
            for k in range(t - n, t + 1):
                v = x[k, s]
                if np.isnan(v):
                    continue
                if bi < 0 or (v >= best if is_max else v <= best):
                    best, bi = v, k
            if bi >= 0:
                out[t, s] = t - bi
    return out


@nb.njit(cache=True, parallel=True)
def nb_zigzag(close, pct):
    """Non-repainting zig zag on closes: a pivot is confirmed only after a pct reversal.
    Returns trend (+1 up / -1 down), last confirmed pivot high, last confirmed pivot low."""
    T, S = close.shape
    trend, ph, pl = np.zeros((T, S)), np.full((T, S), np.nan), np.full((T, S), np.nan)
    for s in nb.prange(S):
        d = 0.0
        ext = lh = ll = np.nan
        for t in range(T):
            c = close[t, s]
            if np.isnan(c):
                trend[t, s], ph[t, s], pl[t, s] = d, lh, ll
                continue
            if np.isnan(ext):
                ext, d = c, 1.0
            if d == 1.0:
                if c > ext:
                    ext = c
                elif c < ext * (1 - pct):
                    lh, d, ext = ext, -1.0, c
            else:
                if c < ext:
                    ext = c
                elif c > ext * (1 + pct):
                    ll, d, ext = ext, 1.0, c
            trend[t, s], ph[t, s], pl[t, s] = d, lh, ll
    return trend, ph, pl


@nb.njit(cache=True, parallel=True)
def nb_structure(piv_hi, piv_lo):
    """+1 when the last two swing highs AND lows are rising (HH + HL), -1 when both falling (LH + LL), else 0."""
    T, S = piv_hi.shape
    out = np.zeros((T, S))
    for s in nb.prange(S):
        h1 = h2 = l1 = l2 = np.nan
        st = 0.0
        for t in range(T):
            if not np.isnan(piv_hi[t, s]):
                h1, h2 = h2, piv_hi[t, s]
            if not np.isnan(piv_lo[t, s]):
                l1, l2 = l2, piv_lo[t, s]
            if not (np.isnan(h1) or np.isnan(l1)):
                if h2 > h1 and l2 > l1:
                    st = 1.0
                elif h2 < h1 and l2 < l1:
                    st = -1.0
            out[t, s] = st
    return out


@nb.njit(cache=True, parallel=True)
def nb_order_block(o, h, l, c, look, life):
    """Bullish order block: on a break of structure (close > highest high of the last `look` bars) the last bearish
    candle before it is the block.  Entry = first later bar that dips into the block and closes above its low.
    Returns entry flags and the stop (block low)."""
    T, S = c.shape
    entry, stop = np.zeros((T, S), np.bool_), np.full((T, S), np.nan)
    for s in nb.prange(S):
        ob_lo = ob_hi = np.nan
        born = -1
        for t in range(look, T):
            if np.isnan(c[t, s]):
                continue
            if not np.isnan(ob_lo) and t - born > life:
                ob_lo = ob_hi = np.nan
            if not np.isnan(ob_lo) and t > born and l[t, s] <= ob_hi and c[t, s] > ob_lo:
                entry[t, s], stop[t, s] = True, ob_lo
                ob_lo = ob_hi = np.nan
                continue
            mx = -1e18
            for k in range(t - look, t):
                if not np.isnan(h[k, s]) and h[k, s] > mx:
                    mx = h[k, s]
            if c[t, s] > mx:
                for k in range(t - 1, t - look - 1, -1):
                    if c[k, s] < o[k, s]:
                        ob_lo, ob_hi, born = l[k, s], h[k, s], t
                        break
    return entry, stop


@nb.njit(cache=True, parallel=True)
def nb_poc(close, vol, start, end, bins):
    """Point of control (price with the most volume) of the closes in rows [start[t], end[t]] for every t."""
    T, S = close.shape
    out = np.full((T, S), np.nan)
    for s in nb.prange(S):
        hist = np.zeros(bins)
        for t in range(T):
            a, b = start[t], end[t]
            if a < 0 or b < a:
                continue
            lo, hi, m = 1e18, -1e18, 0
            for k in range(a, b + 1):
                v = close[k, s]
                if not np.isnan(v) and not np.isnan(vol[k, s]):
                    lo, hi, m = min(lo, v), max(hi, v), m + 1
            if m < (b - a + 1) * 0.8 or hi <= lo:
                continue
            hist[:] = 0.0
            for k in range(a, b + 1):
                v = close[k, s]
                if not np.isnan(v) and not np.isnan(vol[k, s]):
                    i = min(bins - 1, int((v - lo) / (hi - lo) * bins))
                    hist[i] += vol[k, s]
            i = np.argmax(hist)
            out[t, s] = lo + (i + 0.5) * (hi - lo) / bins
    return out


@nb.njit(cache=True, parallel=True)
def nb_maxvol_close(close, vol, n):
    """Close of the highest-volume day within the last n bars (volume-based support/resistance level)."""
    T, S = close.shape
    out = np.full((T, S), np.nan)
    for s in nb.prange(S):
        for t in range(n - 1, T):
            bv, bc = -1.0, np.nan
            for k in range(t - n + 1, t + 1):
                if not np.isnan(vol[k, s]) and vol[k, s] > bv and not np.isnan(close[k, s]):
                    bv, bc = vol[k, s], close[k, s]
            out[t, s] = bc
    return out


# ============================================================ pandas helpers
def F(x) -> np.ndarray:
    return np.ascontiguousarray(x.to_numpy(np.float64) if isinstance(x, pd.DataFrame) else x, dtype=np.float64)


def like(arr, ref: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(arr, index=ref.index, columns=ref.columns)


def sma(x, n): return x.rolling(n, min_periods=n).mean()
def ema(x, n): return x.ewm(span=n, adjust=False, min_periods=n).mean()
def rma(x, n): return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
def hh(x, n): return x.rolling(n, min_periods=n).max()
def ll(x, n): return x.rolling(n, min_periods=n).min()
def wsum(x, n): return x.rolling(n, min_periods=n).sum()
def wma(x, n): return like(nb_wdot(F(x), np.arange(1, n + 1, dtype=np.float64)), x)
def rising(x, k=1): return x > x.shift(k)


def xup(a, b):
    b1 = b.shift(1) if isinstance(b, (pd.DataFrame, pd.Series)) else b
    return (a > b) & (a.shift(1) <= b1)


def xdn(a, b):
    b1 = b.shift(1) if isinstance(b, (pd.DataFrame, pd.Series)) else b
    return (a < b) & (a.shift(1) >= b1)


def rsi(x, n):
    d = x.diff()
    up, dn = rma(d.clip(lower=0), n), rma((-d).clip(lower=0), n)
    return 100 - 100 / (1 + up / dn)


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
        fill = lambda k: P[k][cols].astype("float64").ffill(limit=5).fillna(self.C).where(self.C.notna())
        self.O, self.H, self.L = fill("open"), fill("high"), fill("low")
        self.H = np.maximum(self.H, self.C).where(self.C.notna())
        self.L = np.minimum(self.L, self.C).where(self.C.notna())
        self.V = P["volume"][cols].astype("float64").fillna(0).where(self.C.notna())
        self.elig = elig[cols].to_numpy() & (self.idx >= START_EVAL)[:, None]
        self.tv = tv[cols]
        nc = nifty["close"].reindex(self.idx).ffill()
        self.nifty = nc
        self.mret = nc.pct_change()
        self.hl2, self.tp = (self.H + self.L) / 2, (self.H + self.L + self.C) / 3
        self.ret = self.C.pct_change(fill_method=None)
        self._c = {}
        # forward open-to-open return earned by a position decided at the close of t
        of = self.O
        R = (of.shift(-2) / of.shift(-1) - 1)
        J = jump[cols].reindex(self.idx).fillna(False)
        R = R.mask(J.shift(-1, fill_value=False) | J.shift(-2, fill_value=False))
        self.R = R.fillna(0.0).to_numpy()
        tvn = self.tv.to_numpy()
        self.slip = np.where(tvn >= 5e8, 0.001, np.where(tvn >= 1e8, 0.002, 0.004))
        self.T, self.S = self.C.shape
        self.NAN = np.full((self.T, self.S), np.nan)
        print(f"{self.S} stocks ever liquid since {START_EVAL}; {self.T} days "
              f"{self.idx[0].date()} -> {self.idx[-1].date()}", flush=True)

    def cache(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    def atr(self, n=14):
        def f():
            pc = self.C.shift(1)
            tr = np.maximum(self.H - self.L, np.maximum((self.H - pc).abs(), (self.L - pc).abs()))
            return rma(tr, n)
        return self.cache(("atr", n), f)

    def tr(self):
        pc = self.C.shift(1)
        return np.maximum(self.H, pc) - np.minimum(self.L, pc)

    def hold(self, entry, exit_=None, max_hold=0, stop=None, target=None, trail=None):
        z = lambda a: self.NAN if a is None else F(a)
        ex = np.zeros((self.T, self.S), np.bool_) if exit_ is None else np.asarray(exit_.fillna(False) if hasattr(exit_, "fillna") else exit_, dtype=np.bool_)
        en = np.asarray(entry.fillna(False) if hasattr(entry, "fillna") else entry, dtype=np.bool_)
        return nb_hold(en, ex, max_hold, z(stop), z(target), z(trail), F(self.C))

    # breadth inputs (eligible universe only)
    def breadth(self):
        def f():
            E = pd.DataFrame(self.elig, index=self.idx, columns=self.C.columns)
            ch = self.C.diff()
            adv, dec = ((ch > 0) & E), ((ch < 0) & E)
            uv, dv = self.V.where(adv, 0).sum(1), self.V.where(dec, 0).sum(1)
            return E, adv.sum(1).astype(float), dec.sum(1).astype(float), uv, dv, E.sum(1).astype(float)
        return self.cache("breadth", f)

    def swings(self, n):
        def f():
            w = 2 * n + 1
            ph = self.H.shift(n).where(self.H.shift(n) >= hh(self.H, w))
            pl = self.L.shift(n).where(self.L.shift(n) <= ll(self.L, w))
            return ph, pl
        return self.cache(("swing", n), f)

    def period_levels(self, freq):
        """Previous completed week/month: open, high, low, close mapped onto each day."""
        def f():
            key = self.idx.to_period(freq)
            g = lambda df, how: df.groupby(key).agg(how).shift(1)
            o, h, l, c = g(self.O, "first"), g(self.H, "max"), g(self.L, "min"), g(self.C, "last")
            m = lambda df: pd.DataFrame(df.loc[key].to_numpy(), index=self.idx, columns=self.C.columns)
            return m(o), m(h), m(l), m(c)
        return self.cache(("lvl", freq), f)


# ============================================================ indicator registry
REG: list[dict] = []


def ind(no, name, cat, rule, note=""):
    def deco(f):
        REG.append({"no": no, "name": name, "category": cat, "rule": rule, "note": note, "f": f})
        return f
    return deco


TREND, MOM, VOLA, VOLU, BREADTH, PA, STAT = ("Trend", "Momentum", "Volatility", "Volume", "Market breadth",
                                              "Price action", "Statistical")
PROXY_INTRADAY = "proxy: needs intraday/tick data; daily-bar version used"

# ---------------- A. trend
@ind(1, "Simple Moving Average (SMA)", TREND, "close > SMA(50)")
def _(d): return d.C > sma(d.C, 50)
@ind(2, "Exponential Moving Average (EMA)", TREND, "close > EMA(50)")
def _(d): return d.C > ema(d.C, 50)
@ind(3, "Weighted Moving Average (WMA)", TREND, "close > WMA(50)")
def _(d): return d.C > wma(d.C, 50)
@ind(4, "Volume Weighted Moving Average (VWMA)", TREND, "close > VWMA(50)")
def _(d): return d.C > wsum(d.C * d.V, 50) / wsum(d.V, 50)
@ind(5, "Hull Moving Average (HMA)", TREND, "close > HMA(50)")
def _(d): return d.C > wma(2 * wma(d.C, 25) - wma(d.C, 50), 7)
@ind(6, "Double Exponential Moving Average (DEMA)", TREND, "close > DEMA(50)")
def _(d):
    e = ema(d.C, 50)
    return d.C > 2 * e - ema(e, 50)
@ind(7, "Triple Exponential Moving Average (TEMA)", TREND, "close > TEMA(50)")
def _(d):
    e1 = ema(d.C, 50); e2 = ema(e1, 50)
    return d.C > 3 * e1 - 3 * e2 + ema(e2, 50)
@ind(8, "Kaufman's Adaptive Moving Average (KAMA)", TREND, "close > KAMA(10,2,30)")
def _(d):
    er = (d.C - d.C.shift(10)).abs() / wsum(d.C.diff().abs(), 10)
    sc = (er * (2 / 3 - 2 / 31) + 2 / 31) ** 2
    return d.C > like(nb_adaptive(F(d.C), F(sc)), d.C)
@ind(9, "Arnaud Legoux Moving Average (ALMA)", TREND, "close > ALMA(50, 0.85, 6)")
def _(d):
    n = 50; m = 0.85 * (n - 1); s = n / 6
    w = np.exp(-((np.arange(n) - m) ** 2) / (2 * s * s))
    return d.C > like(nb_wdot(F(d.C), w), d.C)
@ind(10, "McGinley Dynamic", TREND, "close > McGinley(14)")
def _(d): return d.C > like(nb_mcginley(F(d.C), 14.0), d.C)
@ind(11, "Parabolic SAR", TREND, "SAR(0.02, 0.2) below price (uptrend)")
def _(d): return like(nb_psar(F(d.H), F(d.L), 0.02, 0.2), d.C) > 0
@ind(12, "Supertrend", TREND, "Supertrend(10, 3) in uptrend")
def _(d):
    a = rma(d.tr(), 10)
    return like(nb_band_trend(F(d.hl2 - 3 * a), F(d.hl2 + 3 * a), F(d.C)), d.C) > 0
@ind(13, "Ichimoku Cloud", TREND, "close above the cloud and Tenkan > Kijun")
def _(d):
    ten = (hh(d.H, 9) + ll(d.L, 9)) / 2; kij = (hh(d.H, 26) + ll(d.L, 26)) / 2
    a = ((ten + kij) / 2).shift(26); b = ((hh(d.H, 52) + ll(d.L, 52)) / 2).shift(26)
    return (d.C > np.maximum(a, b)) & (ten > kij)
@ind(14, "Aroon", TREND, "BUY: Aroon-Up > 70 and Aroon-Down < 30; SELL: Aroon-Up < Aroon-Down (25)")
def _(d):
    up = 100 * (25 - like(nb_since_ext(F(d.H), 25, True), d.C)) / 25
    dn = 100 * (25 - like(nb_since_ext(F(d.L), 25, False), d.C)) / 25
    return d.hold((up > 70) & (dn < 30), up < dn)
def dmi(d, n=14):
    def f():
        um, dm = d.H.diff(), -d.L.diff()
        pdm = um.where((um > dm) & (um > 0), 0.0); mdm = dm.where((dm > um) & (dm > 0), 0.0)
        a = rma(d.tr(), n)
        pdi, mdi = 100 * rma(pdm, n) / a, 100 * rma(mdm, n) / a
        adx = rma(100 * (pdi - mdi).abs() / (pdi + mdi), n)
        return pdi, mdi, adx
    return d.cache(("dmi", n), f)
@ind(15, "Average Directional Index (ADX)", TREND, "ADX(14) > 25 and +DI > -DI")
def _(d):
    p, m, a = dmi(d)
    return (a > 25) & (p > m)
@ind(16, "Directional Movement Index (DMI)", TREND, "+DI(14) > -DI(14)")
def _(d):
    p, m, _a = dmi(d)
    return p > m
@ind(17, "Vortex Indicator", TREND, "VI+(14) > VI-(14)")
def _(d):
    trs = wsum(d.tr(), 14)
    return wsum((d.H - d.L.shift(1)).abs(), 14) / trs > wsum((d.L - d.H.shift(1)).abs(), 14) / trs
def lreg(d, n=50):
    return d.cache(("lreg", n), lambda: tuple(like(a, d.C) for a in nb_linreg(F(d.C), n)))
@ind(18, "Linear Regression", TREND, "close > linear-regression line end value (50) (LSMA)")
def _(d): return d.C > lreg(d)[1]
@ind(19, "Chande Trend Meter", TREND, "trend score >= 60 / 100 (bands of 20/50/100 + RSI)", "approximation of the TradingView script")
def _(d):
    pts = []
    for n in (20, 50, 100):
        m, s = sma(d.C, n), d.C.rolling(n).std()
        pts += [(d.C > m), (d.C > m + s), (d.C > m + 2 * s), (d.C > d.C.shift(n))]
    pts.append(rsi(d.C, 14) > 50)
    score = sum(p.astype(float) for p in pts) / len(pts) * 100
    return score >= 60
@ind(20, "Moving Average Ribbon", TREND, "EMA 20 > 30 > 40 > 50 > 60 (ribbon stacked up)")
def _(d):
    e = [ema(d.C, n) for n in (20, 30, 40, 50, 60)]
    out = e[0] > e[1]
    for a, b in zip(e[1:], e[2:]):
        out &= a > b
    return out
@ind(21, "Guppy Multiple Moving Average (GMMA)", TREND, "mean of short EMAs (3-15) > mean of long EMAs (30-60)")
def _(d):
    s = sum(ema(d.C, n) for n in (3, 5, 8, 10, 12, 15)) / 6
    l = sum(ema(d.C, n) for n in (30, 35, 40, 45, 50, 60)) / 6
    return s > l
def zz(d, pct):
    return d.cache(("zz", pct), lambda: tuple(like(a, d.C) for a in nb_zigzag(F(d.C), pct)))
@ind(22, "Zig Zag", TREND, "last confirmed 5 % swing is up", "non-repainting version (a swing counts only after it is confirmed)")
def _(d): return zz(d, 0.05)[0] > 0
@ind(23, "Chandelier Exit", TREND, "Chandelier Exit(22, 3) long")
def _(d):
    a = d.atr(22)
    return like(nb_band_trend(F(hh(d.H, 22) - 3 * a), F(ll(d.L, 22) + 3 * a), F(d.C)), d.C) > 0
@ind(24, "Donchian Channels", TREND, "BUY: close > prior 20-day high; SELL: close < channel middle")
def _(d):
    u, l = hh(d.H, 20).shift(1), ll(d.L, 20).shift(1)
    return d.hold(d.C > u, d.C < (u + l) / 2)
def macd(d, x=None, f=12, s=26, g=9):
    x = d.C if x is None else x
    m = ema(x, f) - ema(x, s)
    return m, ema(m, g)
@ind(25, "MACD", TREND, "MACD(12,26) > signal(9)")
def _(d):
    m, s = macd(d)
    return m > s

# ---------------- B. momentum
@ind(26, "Relative Strength Index (RSI)", MOM, "BUY: RSI(14) crosses up 30; SELL: RSI > 70")
def _(d):
    r = rsi(d.C, 14)
    return d.hold(xup(r, 30), r > 70)
def stoch(d, n=14):
    return 100 * (d.C - ll(d.L, n)) / (hh(d.H, n) - ll(d.L, n))
@ind(27, "Stochastic Oscillator", MOM, "BUY: %K(14,3) crosses above %D below 20; SELL: %K > 80")
def _(d):
    k = sma(stoch(d), 3); dd = sma(k, 3)
    return d.hold(xup(k, dd) & (k < 20), k > 80)
@ind(28, "Stochastic RSI", MOM, "BUY: StochRSI %K crosses up 20; SELL: %K > 80")
def _(d):
    r = rsi(d.C, 14)
    k = sma(100 * (r - ll(r, 14)) / (hh(r, 14) - ll(r, 14)), 3)
    return d.hold(xup(k, 20), k > 80)
@ind(29, "Commodity Channel Index (CCI)", MOM, "BUY: CCI(20) crosses up -100; SELL: CCI > 100")
def _(d):
    m = sma(d.tp, 20); md = (d.tp - m).abs().rolling(20).mean()
    c = (d.tp - m) / (0.015 * md)
    return d.hold(xup(c, -100), c > 100)
@ind(30, "Williams %R", MOM, "BUY: %R(14) crosses up -80; SELL: %R > -20")
def _(d):
    w = stoch(d) - 100
    return d.hold(xup(w, -80), w > -20)
@ind(31, "Rate of Change (ROC)", MOM, "ROC(12) > 0")
def _(d): return d.C.pct_change(12, fill_method=None) > 0
@ind(32, "Momentum Indicator", MOM, "close - close 10 days ago > 0")
def _(d): return d.C - d.C.shift(10) > 0
@ind(33, "Money Flow Index (MFI)", MOM, "BUY: MFI(14) crosses up 20; SELL: MFI > 80")
def _(d):
    mf = d.tp * d.V; up = d.tp > d.tp.shift(1)
    pos, neg = wsum(mf.where(up, 0), 14), wsum(mf.where(~up, 0), 14)
    m = 100 - 100 / (1 + pos / neg)
    return d.hold(xup(m, 20), m > 80)
@ind(34, "True Strength Index (TSI)", MOM, "TSI(25,13) > signal(13)")
def _(d):
    m = d.C.diff()
    t = 100 * ema(ema(m, 25), 13) / ema(ema(m.abs(), 25), 13)
    return t > ema(t, 13)
@ind(35, "Relative Vigor Index (RVI)", MOM, "RVI(10) > signal")
def _(d):
    co, hl = d.C - d.O, d.H - d.L
    sym = lambda x: (x + 2 * x.shift(1) + 2 * x.shift(2) + x.shift(3)) / 6
    r = wsum(sym(co), 10) / wsum(sym(hl), 10)
    return r > sym(r)
@ind(36, "Ultimate Oscillator", MOM, "BUY: UO(7,14,28) crosses up 30; SELL: UO > 70")
def _(d):
    pc = d.C.shift(1)
    bp = d.C - np.minimum(d.L, pc); tr = np.maximum(d.H, pc) - np.minimum(d.L, pc)
    a = lambda n: wsum(bp, n) / wsum(tr, n)
    u = 100 * (4 * a(7) + 2 * a(14) + a(28)) / 7
    return d.hold(xup(u, 30), u > 70)
def ao(d): return d.cache("ao", lambda: sma(d.hl2, 5) - sma(d.hl2, 34))
@ind(37, "Awesome Oscillator", MOM, "AO(5,34) > 0")
def _(d): return ao(d) > 0
@ind(38, "Accelerator Oscillator", MOM, "AC > 0 (AO - SMA5(AO))")
def _(d): return ao(d) - sma(ao(d), 5) > 0
@ind(39, "Chande Momentum Oscillator (CMO)", MOM, "BUY: CMO(14) crosses up -50; SELL: CMO > 50")
def _(d):
    m = d.C.diff(); su, sd = wsum(m.clip(lower=0), 14), wsum((-m).clip(lower=0), 14)
    c = 100 * (su - sd) / (su + sd)
    return d.hold(xup(c, -50), c > 50)
@ind(40, "Fisher Transform", MOM, "Fisher(9) > trigger (previous value)")
def _(d):
    x = 2 * ((d.hl2 - ll(d.hl2, 9)) / (hh(d.hl2, 9) - ll(d.hl2, 9)).replace(0, np.nan) - 0.5)
    f, t = nb_fisher(F(x))
    return like(f, d.C) > like(t, d.C)
@ind(41, "Detrended Price Oscillator (DPO)", MOM, "DPO(20) > 0 (non-centred)")
def _(d): return d.C - sma(d.C, 20).shift(11) > 0
@ind(42, "Percentage Price Oscillator (PPO)", MOM, "PPO(12,26) > signal(9)")
def _(d):
    p = 100 * (ema(d.C, 12) - ema(d.C, 26)) / ema(d.C, 26)
    return p > ema(p, 9)
@ind(43, "Relative Momentum Index (RMI)", MOM, "BUY: RMI(20,5) crosses up 30; SELL: RMI > 70")
def _(d):
    m = d.C - d.C.shift(5)
    r = 100 - 100 / (1 + rma(m.clip(lower=0), 20) / rma((-m).clip(lower=0), 20))
    return d.hold(xup(r, 30), r > 70)
@ind(44, "Know Sure Thing (KST)", MOM, "KST > signal(9)")
def _(d):
    roc = lambda n: d.C.pct_change(n, fill_method=None) * 100
    k = sma(roc(10), 10) + 2 * sma(roc(15), 10) + 3 * sma(roc(20), 10) + 4 * sma(roc(30), 15)
    return k > sma(k, 9)
@ind(45, "Connors RSI", MOM, "BUY: CRSI(3,2,100) < 10; SELL: CRSI > 70")
def _(d):
    st = like(nb_streak(F(d.C)), d.C)
    pr = like(nb_pctrank(F(d.ret), 100), d.C)
    c = (rsi(d.C, 3) + rsi(st, 2) + pr) / 3
    return d.hold(c < 10, c > 70)
@ind(46, "Elder Ray Index", MOM, "BUY: EMA13 rising, bear power < 0 and rising; SELL: EMA13 falling")
def _(d):
    e = ema(d.C, 13); bear = d.L - e
    return d.hold(rising(e) & (bear < 0) & rising(bear), ~rising(e))
@ind(47, "Balance of Power (BOP)", MOM, "SMA14 of BOP > 0")
def _(d): return sma((d.C - d.O) / (d.H - d.L).replace(0, np.nan), 14) > 0
@ind(48, "Psychological Line (PSY)", MOM, "BUY: PSY(12) < 25 (oversold); SELL: PSY > 75")
def _(d):
    p = 100 * wsum((d.C > d.C.shift(1)).astype(float), 12) / 12
    return d.hold(p < 25, p > 75)
@ind(49, "TRIX", MOM, "TRIX(15) > signal(9)")
def _(d):
    t = ema(ema(ema(d.C, 15), 15), 15).pct_change(fill_method=None) * 100
    return t > ema(t, 9)
@ind(50, "Coppock Curve", MOM, "Coppock (daily version of 14/11/10 months) rising vs a month ago")
def _(d):
    c = wma(d.C.pct_change(294, fill_method=None) + d.C.pct_change(231, fill_method=None), 210)
    return c > c.shift(21)

# ---------------- C. volatility
def calm(x, n=252):
    return x < x.rolling(n, min_periods=n // 2).median()
@ind(51, "Average True Range (ATR)", VOLA, "BUY: close > prior close + ATR(14); SELL: 3 x ATR trailing stop")
def _(d):
    a = d.atr(14)
    return d.hold(d.C > d.C.shift(1) + a.shift(1), trail=3 * a)
def bb(d, n=20, k=2):
    m, s = sma(d.C, n), d.C.rolling(n).std()
    return m - k * s, m, m + k * s
@ind(52, "Bollinger Bands", VOLA, "BUY: close < lower band(20,2); SELL: close > middle band (mean reversion)")
def _(d):
    lo, m, _u = bb(d)
    return d.hold(d.C < lo, d.C > m)
@ind(53, "Bollinger Bands %B", VOLA, "BUY: %B > 1 (breakout); SELL: %B < 0.5")
def _(d):
    lo, m, u = bb(d)
    b = (d.C - lo) / (u - lo)
    return d.hold(b > 1, b < 0.5)
@ind(54, "Bollinger Band Width", VOLA, "BUY: squeeze (width in lowest 20 % of 6 months) then close > upper band; SELL: < middle")
def _(d):
    lo, m, u = bb(d)
    w = (u - lo) / m
    sq = (w <= w.rolling(126, min_periods=100).quantile(0.2)).rolling(5).max() > 0
    return d.hold(sq & (d.C > u), d.C < m)
@ind(55, "Keltner Channels", VOLA, "BUY: close > upper (EMA20 + 2 ATR10); SELL: close < EMA20")
def _(d):
    m = ema(d.C, 20)
    return d.hold(d.C > m + 2 * d.atr(10), d.C < m)
@ind(56, "Donchian Channels Width", VOLA, "BUY: narrow channel (lowest 20 % of 6 months) then breakout above it; SELL: < middle")
def _(d):
    u, l = hh(d.H, 20).shift(1), ll(d.L, 20).shift(1)
    w = (u - l) / ((u + l) / 2)
    sq = w <= w.rolling(126, min_periods=100).quantile(0.2)
    return d.hold(sq & (d.C > u), d.C < (u + l) / 2)
@ind(57, "Historical Volatility", VOLA, "hold while HV(20) < its 1-year median (calm regime)")
def _(d): return calm(d.ret.rolling(20).std())
@ind(58, "Chaikin Volatility", VOLA, "BUY: Chaikin volatility(10,10) > 0 (expanding) and close > SMA(20); SELL: close < SMA(20)")
def _(d):
    e = ema(d.H - d.L, 10); cv = e / e.shift(10) - 1
    m = sma(d.C, 20)
    return d.hold((cv > 0) & (d.C > m), d.C < m)
@ind(59, "Standard Deviation", VOLA, "BUY: StdDev(20) > StdDev(100) (expansion) and close > SMA(20); SELL: close < SMA(20)")
def _(d):
    m = sma(d.C, 20)
    return d.hold((d.ret.rolling(20).std() > d.ret.rolling(100).std()) & (d.C > m), d.C < m)
@ind(60, "Average Daily Range (ADR)", VOLA, "BUY: up bar with range > 1.5 x ADR(20) closing above prior high; SELL after 10 days")
def _(d):
    adr = sma(d.H - d.L, 20).shift(1)
    return d.hold((d.H - d.L > 1.5 * adr) & (d.C > d.H.shift(1)), max_hold=10)
@ind(61, "Relative Volatility Index (RVI)", VOLA, "RVI(10, 14) > 50")
def _(d):
    s = d.C.rolling(10).std(); ch = d.C.diff()
    u, dn = ema(s.where(ch > 0, 0), 14), ema(s.where(ch <= 0, 0), 14)
    return 100 * u / (u + dn) > 50
@ind(62, "Ulcer Index", VOLA, "hold while Ulcer Index(14) < its 1-year median (shallow drawdowns)")
def _(d):
    dd = 100 * (d.C / hh(d.C, 14) - 1)
    return calm(np.sqrt(sma(dd ** 2, 14)))
@ind(63, "Mass Index", VOLA, "BUY: reversal bulge (MI > 27 then < 26.5) with close > EMA(9); SELL: close < EMA(20)")
def _(d):
    e = ema(d.H - d.L, 9); mi = wsum(e / ema(e, 9), 25)
    bulge = (mi > 27).rolling(25).max() > 0
    return d.hold(bulge & xdn(mi, 26.5) & (d.C > ema(d.C, 9)), d.C < ema(d.C, 20))
@ind(64, "Squeeze Momentum Indicator", VOLA, "BUY: squeeze (BB inside KC) releases with momentum > 0; SELL: momentum < 0")
def _(d):
    lo, m, u = bb(d, 20, 2)
    a = sma(d.tr(), 20)
    sq = (lo > m - 1.5 * a) & (u < m + 1.5 * a)
    x = d.C - ((hh(d.H, 20) + ll(d.L, 20)) / 2 + m) / 2
    mom = like(nb_linreg(F(x), 20)[1], d.C)
    return d.hold(sq.shift(1, fill_value=False) & ~sq & (mom > 0), mom < 0)
@ind(65, "Volatility Stop", VOLA, "Volatility Stop(20, 2 ATR) in uptrend")
def _(d): return like(nb_volstop(F(d.C), F(d.atr(20)), 2.0), d.C) > 0
@ind(66, "Choppiness Index", VOLA, "BUY: CHOP(14) < 38.2 (trending) and close > SMA(50); SELL: CHOP > 61.8 or close < SMA(50)")
def _(d):
    ch = 100 * np.log10(wsum(d.tr(), 14) / (hh(d.H, 14) - ll(d.L, 14))) / np.log10(14)
    m = sma(d.C, 50)
    return d.hold((ch < 38.2) & (d.C > m), (ch > 61.8) | (d.C < m))
@ind(67, "Normalized Average True Range (NATR)", VOLA, "hold while NATR(14) < 1-year median and close > SMA(200) (calm uptrend)")
def _(d): return calm(d.atr(14) / d.C) & (d.C > sma(d.C, 200))
@ind(68, "Garman-Klass Volatility", VOLA, "hold while GK vol(20) < its 1-year median")
def _(d):
    x = 0.5 * np.log(d.H / d.L) ** 2 - (2 * np.log(2) - 1) * np.log(d.C / d.O) ** 2
    return calm(np.sqrt(sma(x, 20)))
@ind(69, "Parkinson Volatility", VOLA, "hold while Parkinson vol(20) < its 1-year median")
def _(d): return calm(np.sqrt(sma(np.log(d.H / d.L) ** 2, 20) / (4 * np.log(2))))
@ind(70, "Rogers-Satchell Volatility", VOLA, "hold while Rogers-Satchell vol(20) < its 1-year median")
def _(d):
    x = np.log(d.H / d.C) * np.log(d.H / d.O) + np.log(d.L / d.C) * np.log(d.L / d.O)
    return calm(np.sqrt(sma(x, 20).clip(lower=0)))

# ---------------- D. volume
@ind(71, "Volume", VOLU, "BUY: volume > 1.5 x 20-day average on a 20-day closing high; SELL: close < SMA(20)")
def _(d):
    return d.hold((d.V > 1.5 * sma(d.V, 20).shift(1)) & (d.C >= hh(d.C, 20)), d.C < sma(d.C, 20))
def cumline(x): return x.fillna(0).cumsum().where(x.notna() | x.isna())
@ind(72, "On-Balance Volume (OBV)", VOLU, "OBV > EMA(20) of OBV")
def _(d):
    o = (np.sign(d.C.diff()) * d.V).fillna(0).cumsum()
    return o > ema(o, 20)
@ind(73, "Volume Weighted Average Price (VWAP)", VOLU, "close > rolling 20-day VWAP", PROXY_INTRADAY)
def _(d): return d.C > wsum(d.tp * d.V, 20) / wsum(d.V, 20)
@ind(74, "Anchored VWAP", VOLU, "close > VWAP anchored at the latest 52-week low")
def _(d):
    T = d.T
    newlow = (d.L <= ll(d.L, 252)).to_numpy()
    a = np.where(newlow, np.arange(T)[:, None], -1)
    a = np.maximum.accumulate(a, axis=0)
    pv, v = F((d.tp * d.V).fillna(0)).cumsum(0), F(d.V.fillna(0)).cumsum(0)
    pv0, v0 = np.vstack([np.zeros((1, d.S)), pv]), np.vstack([np.zeros((1, d.S)), v])
    ai = np.where(a < 0, 0, a)
    av = (pv - np.take_along_axis(pv0, ai, 0)) / (v - np.take_along_axis(v0, ai, 0))
    av[a < 0] = np.nan
    return d.C > like(av, d.C)
def poc(d, kind):
    def f():
        T = d.T; ti = np.arange(T)
        if kind == "fixed":                                   # previous calendar quarter
            q = d.idx.to_period("Q")
            first = pd.Series(ti, index=q).groupby(level=0).min()
            last = pd.Series(ti, index=q).groupby(level=0).max()
            prev_q = q - 1
            st = first.reindex(prev_q).fillna(-1).to_numpy().astype(np.int64)
            en = last.reindex(prev_q).fillna(-1).to_numpy().astype(np.int64)
        else:
            n = {"vp": 252, "vrvp": 60}[kind]
            st, en = ti - n + 1, ti.copy()
            st[st < 0] = -1
        return like(nb_poc(F(d.C), F(d.V), st.astype(np.int64), en.astype(np.int64), 24), d.C)
    return d.cache(("poc", kind), f)
@ind(75, "Volume Profile", VOLU, "close > point of control (POC) of the last 252 days")
def _(d): return d.C > poc(d, "vp")
@ind(76, "Visible Range Volume Profile", VOLU, "close > POC of the last 60 days (visible range)")
def _(d): return d.C > poc(d, "vrvp")
@ind(77, "Fixed Range Volume Profile", VOLU, "close > POC of the previous calendar quarter")
def _(d): return d.C > poc(d, "fixed")
def adl(d):
    def f():
        mfm = ((d.C - d.L) - (d.H - d.C)) / (d.H - d.L).replace(0, np.nan)
        return (mfm.fillna(0) * d.V.fillna(0)).cumsum()
    return d.cache("adl", f)
@ind(78, "Accumulation/Distribution Line", VOLU, "A/D line > EMA(20) of A/D")
def _(d): return adl(d) > ema(adl(d), 20)
@ind(79, "Chaikin Money Flow (CMF)", VOLU, "CMF(20) > 0")
def _(d):
    mfm = ((d.C - d.L) - (d.H - d.C)) / (d.H - d.L).replace(0, np.nan)
    return wsum(mfm.fillna(0) * d.V, 20) / wsum(d.V, 20) > 0
@ind(80, "Chaikin Oscillator", VOLU, "EMA3(A/D) - EMA10(A/D) > 0")
def _(d): return ema(adl(d), 3) - ema(adl(d), 10) > 0
@ind(81, "Volume Rate of Change", VOLU, "VROC(14) > 0 and price ROC(14) > 0")
def _(d): return (d.V / sma(d.V, 5).shift(14) - 1 > 0) & (d.C.pct_change(14, fill_method=None) > 0)
@ind(82, "Ease of Movement (EOM)", VOLU, "SMA14 of EOM > 0")
def _(d):
    e = d.hl2.diff() * (d.H - d.L) / d.V.replace(0, np.nan)
    return sma(e, 14) > 0
@ind(83, "Force Index", VOLU, "EMA13 of force index > 0")
def _(d): return ema(d.C.diff() * d.V, 13) > 0
def vol_index(d, neg):
    cond = d.V < d.V.shift(1) if neg else d.V > d.V.shift(1)
    x = (1 + d.ret.where(cond, 0).fillna(0)).cumprod()
    return x.where(d.C.notna())
@ind(84, "Negative Volume Index (NVI)", VOLU, "NVI > EMA(255) of NVI")
def _(d):
    x = vol_index(d, True)
    return x > ema(x, 255)
@ind(85, "Positive Volume Index (PVI)", VOLU, "PVI > EMA(255) of PVI")
def _(d):
    x = vol_index(d, False)
    return x > ema(x, 255)
@ind(86, "Klinger Volume Oscillator", VOLU, "KVO(34,55) > signal(13)")
def _(d):
    sv = d.V * np.where(d.tp > d.tp.shift(1), 1.0, -1.0)
    k = ema(sv, 34) - ema(sv, 55)
    return k > ema(k, 13)
@ind(87, "Volume Oscillator", VOLU, "EMA5(volume) > EMA10(volume) and close > SMA(20)")
def _(d): return (ema(d.V, 5) > ema(d.V, 10)) & (d.C > sma(d.C, 20))
@ind(88, "Relative Volume (RVOL)", VOLU, "BUY: RVOL > 2 on an up day; SELL after 10 days")
def _(d): return d.hold((d.V / sma(d.V, 20).shift(1) > 2) & (d.C > d.C.shift(1)), max_hold=10)
@ind(89, "Money Flow", VOLU, "20-day net money flow (typical price x volume, signed) > 0")
def _(d):
    mf = d.tp * d.V * np.sign(d.tp.diff())
    return wsum(mf, 20) > 0
@ind(90, "Price Volume Trend (PVT)", VOLU, "PVT > EMA(20) of PVT")
def _(d):
    p = (d.ret * d.V).fillna(0).cumsum()
    return p > ema(p, 20)
@ind(91, "Volume Delta", VOLU, "5-day sum of estimated volume delta > 0", PROXY_INTRADAY)
def _(d):
    delta = d.V * ((d.C - d.L) - (d.H - d.C)) / (d.H - d.L).replace(0, np.nan)
    return wsum(delta.fillna(0), 5) > 0
@ind(92, "Cumulative Volume Delta (CVD)", VOLU, "CVD > EMA(50) of CVD", PROXY_INTRADAY)
def _(d): return adl(d) > ema(adl(d), 50)

# ---------------- E. market breadth (market timing: hold all liquid stocks when bullish)
@ind(93, "Advance-Decline Line", BREADTH, "A/D line > its EMA(21)")
def _(d):
    _E, a, dc, *_ = d.breadth(); x = (a - dc).cumsum()
    return x > ema(x, 21)
@ind(94, "Advance-Decline Ratio", BREADTH, "10-day advances / declines > 1")
def _(d):
    _E, a, dc, *_ = d.breadth()
    return wsum(a, 10) / wsum(dc, 10) > 1
@ind(95, "Advance-Decline Volume", BREADTH, "cumulative up-volume minus down-volume > its EMA(21)")
def _(d):
    *_x, uv, dv, _n = d.breadth(); x = (uv - dv).cumsum()
    return x > ema(x, 21)
def mcc(d):
    _E, a, dc, *_ = d.breadth()
    r = 1000 * (a - dc) / (a + dc).replace(0, np.nan)
    return ema(r, 19) - ema(r, 39)
@ind(96, "McClellan Oscillator", BREADTH, "McClellan oscillator (ratio-adjusted) > 0")
def _(d): return mcc(d) > 0
@ind(97, "McClellan Summation Index", BREADTH, "summation index > its EMA(10)")
def _(d):
    s = mcc(d).fillna(0).cumsum()
    return s > ema(s, 10)
@ind(98, "Arms Index (TRIN)", BREADTH, "BUY: TRIN(10) > 1.2 (oversold); SELL: TRIN(10) < 0.85")
def _(d):
    _E, a, dc, uv, dv, _n = d.breadth()
    t = sma((a / dc) / (uv / dv), 10)
    st, on = [], False
    for v in t.to_numpy():
        on = (v > 1.2) if not on else not (v < 0.85)
        st.append(on)
    return pd.Series(st, index=t.index)
@ind(99, "New Highs-New Lows", BREADTH, "10-day average of (52-week highs - lows) > 0")
def _(d):
    E = d.breadth()[0]
    nh = ((d.C >= hh(d.C, 252)) & E).sum(1); nl = ((d.C <= ll(d.C, 252)) & E).sum(1)
    return sma((nh - nl).astype(float), 10) > 0
def pct_above(d, n):
    E = d.breadth()[0]
    return 100 * ((d.C > sma(d.C, n)) & E).sum(1) / E.sum(1).replace(0, np.nan)
@ind(100, "Percentage of Stocks Above 50 DMA", BREADTH, "% of stocks above 50-DMA > 50")
def _(d): return pct_above(d, 50) > 50
@ind(101, "Percentage of Stocks Above 200 DMA", BREADTH, "% of stocks above 200-DMA > 50")
def _(d): return pct_above(d, 200) > 50
@ind(102, "Bullish Percent Index", BREADTH, "% of stocks on a breakout buy signal > 50", "approximation: 20-day breakout state instead of point & figure charts")
def _(d):
    E = d.breadth()[0]
    st = like(d.hold(d.C > hh(d.H, 20).shift(1), d.C < ll(d.L, 20).shift(1)), d.C)
    return 100 * (st & E).sum(1) / E.sum(1).replace(0, np.nan) > 50
@ind(103, "Up/Down Volume Ratio", BREADTH, "10-day up volume / down volume > 1")
def _(d):
    *_x, uv, dv, _n = d.breadth()
    return wsum(uv, 10) / wsum(dv, 10) > 1
@ind(104, "Absolute Breadth Index", BREADTH, "hold while 10-day ABI (|adv - dec| / issues) < its 1-year median (calm market)")
def _(d):
    _E, a, dc, _u, _v, n = d.breadth()
    return calm(sma((a - dc).abs() / n.replace(0, np.nan), 10))

# ---------------- F. price action / support-resistance
def pivots(d, freq="M"):
    o, h, l, c = d.period_levels(freq)
    return o, h, l, c, (h + l + c) / 3
@ind(105, "Pivot Points Standard", PA, "close > monthly pivot P = (H + L + C) / 3 of last month")
def _(d): return d.C > pivots(d)[4]
@ind(106, "Fibonacci Retracement", PA, "in an uptrend (close > SMA200) BUY when close crosses up the 61.8 % retracement of the 6-month range; target = range high, stop = 78.6 %")
def _(d):
    h, l = hh(d.H, 126).shift(1), ll(d.L, 126).shift(1)
    lv = h - 0.618 * (h - l)
    return d.hold(xup(d.C, lv) & (d.C > sma(d.C, 200)), stop=h - 0.786 * (h - l), target=h)
@ind(107, "Fibonacci Extension", PA, "BUY: close > 6-month high; target = 161.8 % extension, stop = 50 % retracement")
def _(d):
    h, l = hh(d.H, 126).shift(1), ll(d.L, 126).shift(1)
    return d.hold(d.C > h, stop=h - 0.5 * (h - l), target=h + 0.618 * (h - l))
@ind(108, "Auto Fibonacci", PA, "after a confirmed 10 % zig-zag swing high, BUY when close crosses up the 61.8 % retracement of the last up-leg; target = swing high, stop = swing low")
def _(d):
    tr, ph, pl = zz(d, 0.10)
    lv = ph - 0.618 * (ph - pl)
    return d.hold((tr < 0) & (ph > pl) & xup(d.C, lv), stop=pl, target=ph)
@ind(109, "Support and Resistance", PA, "BUY: close > last swing-high resistance (10-bar pivots); SELL: close < last swing-low support")
def _(d):
    ph, pl = d.swings(10)
    return d.hold(d.C > ph.ffill(), d.C < pl.ffill())
@ind(110, "Donchian Breakout", PA, "Turtle: BUY close > prior 55-day high; SELL close < prior 20-day low")
def _(d): return d.hold(d.C > hh(d.H, 55).shift(1), d.C < ll(d.L, 20).shift(1))
@ind(111, "Opening Range", PA, "BUY when close breaks above the month's first-day high; SELL below first-day low or at month end", "proxy: true opening range needs intraday bars; monthly first-day range used")
def _(d):
    key = d.idx.to_period("M")
    first = pd.Series(np.arange(d.T), index=d.idx).groupby(key).transform("min").to_numpy()
    fh = pd.DataFrame(d.H.to_numpy()[first], index=d.idx, columns=d.C.columns)
    fl = pd.DataFrame(d.L.to_numpy()[first], index=d.idx, columns=d.C.columns)
    last_day = pd.Series(key, index=d.idx).ne(pd.Series(key, index=d.idx).shift(-1)).to_numpy()
    not_first = (np.arange(d.T) != first)[:, None]
    return d.hold((d.C > fh) & not_first, (d.C < fl) | last_day[:, None])
@ind(112, "Previous Day High/Low", PA, "BUY: close > previous day high; SELL: close < previous day low")
def _(d): return d.hold(d.C > d.H.shift(1), d.C < d.L.shift(1))
@ind(113, "Previous Week High/Low", PA, "BUY: close > previous week high; SELL: close < previous week low")
def _(d):
    _o, h, l, _c = d.period_levels("W")
    return d.hold(d.C > h, d.C < l)
@ind(114, "Previous Month High/Low", PA, "BUY: close > previous month high; SELL: close < previous month low")
def _(d):
    _o, h, l, _c = d.period_levels("M")
    return d.hold(d.C > h, d.C < l)
@ind(115, "Camarilla Pivot Points", PA, "weekly Camarilla: BUY close > H4; SELL close < L3")
def _(d):
    _o, h, l, c = d.period_levels("W")
    return d.hold(d.C > c + 1.1 * (h - l) / 2, d.C < c - 1.1 * (h - l) / 4)
@ind(116, "Woodie's Pivot Points", PA, "close > monthly Woodie pivot (H + L + 2C) / 4")
def _(d):
    _o, h, l, c = d.period_levels("M")
    return d.C > (h + l + 2 * c) / 4
@ind(117, "DeMark Pivot Points", PA, "close > monthly DeMark pivot X / 4")
def _(d):
    o, h, l, c = d.period_levels("M")
    x = np.where(c < o, h + 2 * l + c, np.where(c > o, 2 * h + l + c, h + l + 2 * c))
    return d.C > like(x / 4, d.C)
@ind(118, "Floor Pivot Points", PA, "monthly floor pivots: BUY close > R1; SELL close < P")
def _(d):
    _o, h, l, _c, p = pivots(d)
    return d.hold(d.C > 2 * p - l, d.C < p)
def fractals(d):
    def f():
        H, L = d.H, d.L
        up = (H.shift(2) > H.shift(3)) & (H.shift(2) > H.shift(4)) & (H.shift(2) > H.shift(1)) & (H.shift(2) > H)
        dn = (L.shift(2) < L.shift(3)) & (L.shift(2) < L.shift(4)) & (L.shift(2) < L.shift(1)) & (L.shift(2) < L)
        return H.shift(2).where(up).ffill(), L.shift(2).where(dn).ffill()
    return d.cache("fract", f)
@ind(119, "Fractal Indicator", PA, "BUY: close > last up-fractal (5-bar); SELL: close < last down-fractal")
def _(d):
    u, l = fractals(d)
    return d.hold(d.C > u, d.C < l)
@ind(120, "Williams Fractals", PA, "Bill Williams: BUY close > last up-fractal while above the Alligator teeth; SELL close < teeth")
def _(d):
    u, _l = fractals(d)
    teeth = rma(d.hl2, 8).shift(5)
    return d.hold((d.C > u) & (d.C > teeth), d.C < teeth)
@ind(121, "Market Structure (HH, HL, LH, LL)", PA, "hold while the last swing high and low are higher (HH + HL), 5-bar pivots")
def _(d):
    ph, pl = d.swings(5)
    return like(nb_structure(F(ph), F(pl)), d.C) > 0
@ind(122, "Fair Value Gap", PA, "BUY on a bullish FVG (low > high 2 bars ago); SELL if the gap fills or after 20 days")
def _(d): return d.hold((d.L > d.H.shift(2)) & (d.C > d.O), stop=d.H.shift(2), max_hold=20)
@ind(123, "Order Blocks", PA, "BUY on the first retest of a bullish order block; SELL below the block or after 20 days", "rule-based version of a discretionary concept")
def _(d):
    e, st = nb_order_block(F(d.O), F(d.H), F(d.L), F(d.C), 10, 30)
    return d.hold(like(e, d.C), stop=like(st, d.C), max_hold=20)
@ind(124, "Volume-Based Support and Resistance", PA, "close > close of the highest-volume day of the last 60 days")
def _(d): return d.C > like(nb_maxvol_close(F(d.C), F(d.V), 60), d.C)

# ---------------- G. statistical
@ind(125, "Z-Score", STAT, "BUY: 20-day z-score < -2; SELL: z > 0 (mean reversion)")
def _(d):
    z = (d.C - sma(d.C, 20)) / d.C.rolling(20).std()
    return d.hold(z < -2, z > 0)
def roll_cov(d, n):
    m = d.mret
    er, em = d.ret.rolling(n).mean(), m.rolling(n).mean()
    cov = d.ret.mul(m, axis=0).rolling(n).mean() - er.mul(em, axis=0)
    return cov, er, em, m.rolling(n).var(ddof=0)
@ind(126, "Rolling Correlation", STAT, "hold while 63-day correlation with NIFTY < 0.3 and close > SMA(50)")
def _(d):
    cov, *_x, vm = roll_cov(d, 63)
    corr = cov.div(np.sqrt(vm), axis=0) / d.ret.rolling(63).std(ddof=0)
    return (corr < 0.3) & (d.C > sma(d.C, 50))
@ind(127, "Rolling Beta", STAT, "hold while 126-day beta vs NIFTY < 0.8 (low-beta)")
def _(d):
    cov, *_x, vm = roll_cov(d, 126)
    return cov.div(vm, axis=0) < 0.8
@ind(128, "Rolling Regression", STAT, "hold while 63-day alpha vs NIFTY (regression intercept) > 0")
def _(d):
    cov, er, em, vm = roll_cov(d, 63)
    beta = cov.div(vm, axis=0)
    return er - beta.mul(em, axis=0) > 0
@ind(129, "Linear Regression Slope", STAT, "50-day regression slope > 0")
def _(d): return lreg(d)[0] > 0
@ind(130, "R-Squared", STAT, "50-day R^2 > 0.6 with positive slope (strong clean uptrend)")
def _(d):
    s, _e, r2 = lreg(d)
    return (r2 > 0.6) & (s > 0)
@ind(131, "Standardized Momentum", STAT, "12-1 month return / volatility > 0.5")
def _(d):
    m = d.C.shift(21) / d.C.shift(252) - 1
    return m / (d.ret.rolling(252).std() * np.sqrt(231)) > 0.5
@ind(132, "Relative Strength vs Benchmark", STAT, "stock / NIFTY ratio > its 50-day average")
def _(d):
    r = d.C.div(d.nifty, axis=0)
    return r > sma(r, 50)
@ind(133, "Spread / Ratio", STAT, "BUY: stock/NIFTY ratio 63-day z-score < -2; SELL: z > 0")
def _(d):
    r = d.C.div(d.nifty, axis=0)
    z = (r - sma(r, 63)) / r.rolling(63).std()
    return d.hold(z < -2, z > 0)
@ind(134, "Mean Reversion Score", STAT, "BUY: 5-day return z-score < -2 while close > SMA(200); SELL: close > SMA(5)")
def _(d):
    r5 = d.C.pct_change(5, fill_method=None)
    z = (r5 - r5.rolling(252, min_periods=126).mean()) / r5.rolling(252, min_periods=126).std()
    return d.hold((z < -2) & (d.C > sma(d.C, 200)), d.C > sma(d.C, 5))
@ind(135, "Hurst Exponent", STAT, "hold while Hurst(100) > 0.55 (trending) and close > SMA(50)")
def _(d):
    lp = np.log(d.C)
    lags = np.array([2, 4, 8, 16])
    ys = [np.log((lp - lp.shift(k)).rolling(100, min_periods=80).std()) for k in lags]
    x = np.log(lags); xm = x.mean()
    h = sum((xi - xm) * y for xi, y in zip(x, ys)) / ((x - xm) ** 2).sum()
    return (h > 0.55) & (d.C > sma(d.C, 50))
@ind(136, "Entropy", STAT, "hold while 20-day up/down entropy < 0.9 bits with mostly up days (orderly uptrend)")
def _(d):
    p = sma((d.C > d.C.shift(1)).astype(float).where(d.C.notna()), 20).clip(1e-6, 1 - 1e-6)
    e = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return (e < 0.9) & (p > 0.5)
@ind(137, "Autocorrelation", STAT, "63-day lag-1 autocorrelation > 0.1 and 5-day return > 0 (momentum regime)")
def _(d):
    r = d.ret
    a, b = r, r.shift(1)
    cov = (a * b).rolling(63).mean() - a.rolling(63).mean() * b.rolling(63).mean()
    ac = cov / (a.rolling(63).std(ddof=0) * b.rolling(63).std(ddof=0))
    return (ac > 0.1) & (d.C.pct_change(5, fill_method=None) > 0)
@ind(138, "Kaufman Efficiency Ratio", STAT, "ER(10) > 0.5 and close > close 10 days ago")
def _(d):
    er = (d.C - d.C.shift(10)).abs() / wsum(d.C.diff().abs(), 10)
    return (er > 0.5) & (d.C > d.C.shift(10))
@ind(139, "Rolling Sharpe Ratio", STAT, "BUY: 126-day Sharpe > 1; SELL: Sharpe < 0")
def _(d):
    s = d.ret.rolling(126).mean() / d.ret.rolling(126).std() * np.sqrt(252)
    return d.hold(s > 1, s < 0)
@ind(140, "Rolling Sortino Ratio", STAT, "BUY: 126-day Sortino > 1.5; SELL: Sortino < 0")
def _(d):
    dn = np.sqrt((d.ret.clip(upper=0) ** 2).rolling(126).mean())
    s = d.ret.rolling(126).mean() / dn * np.sqrt(252)
    return d.hold(s > 1.5, s < 0)


# ============================================================ backtest engine
def run(d: Data, sig) -> tuple[pd.Series, dict]:
    if isinstance(sig, pd.Series):                                     # market-timing rule
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
    rp = gross - cin - cout
    dates = d.idx[1:]                                                   # return earned from the next open
    r = pd.Series(rp[:-1], index=dates)
    r = r[r.index >= START_EVAL]
    rg = pd.Series(gross[:-1], index=dates)
    rg = rg[rg.index >= START_EVAL]

    # trades (closed only)
    P = pos.T.ravel()                                                   # stock-major
    E = ent.T.ravel()
    tid = np.cumsum(E) * P
    lr = np.log1p(np.clip(d.R.T.ravel(), -0.95, None)) * P
    k = tid.max() + 1
    sum_lr = np.bincount(tid, weights=lr, minlength=k)[1:]
    length = np.bincount(tid, minlength=k)[1:]
    slip_in = (d.slip.T.ravel())[E]
    net = np.exp(sum_lr) * (1 - BUY_COST - SELL_COST - 2 * slip_in) - 1      # open trades marked at the last price
    closed = np.ones(len(net), bool)
    yrs = (d.idx[-1] - pd.Timestamp(START_EVAL)).days / 365.25
    stats = {"trades": int(closed.sum()),
             "trades_per_year": round(float(closed.sum() / yrs), 1),
             "win_rate_pct": round(float((net[closed] > 0).mean() * 100), 1) if closed.any() else np.nan,
             "avg_trade_pct": round(float(net[closed].mean() * 100), 2) if closed.any() else np.nan,
             "median_hold_days": float(np.median(length[closed])) if closed.any() else np.nan,
             "avg_stocks_held": round(float(n[d.idx >= START_EVAL].mean()), 1),
             "pct_time_invested": round(float((n[d.idx >= START_EVAL] > 0).mean() * 100), 1),
             "gross_cagr_pct": perf(rg).get("cagr_pct"), "gross_sharpe": perf(rg).get("sharpe"),
             "cost_drag_pct_per_year": round(float((cin + cout)[d.idx >= START_EVAL].sum() / yrs * 100), 2)}
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = {int(x) for x in a.only.split(",") if x.strip()}
    t0 = datetime.now()
    d = Data()
    out = ROOT / "results" / f"indicators_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    rows, yearly, monthly = [], {}, {}
    bench = {}
    r_ew, st_ew = run(d, pd.Series(True, index=d.idx))
    bench["Hold all liquid stocks"] = (r_ew, st_ew)
    nr = d.nifty.pct_change().reindex(r_ew.index).fillna(0)
    bench["NIFTY 50"] = (nr, {})
    for name, (r, st) in bench.items():
        rows.append({"no": 0, "name": name, "category": "Benchmark", "rule": "", "note": "", **periods(r), **st})
        yearly[name] = (1 + r).groupby(r.index.year).prod() - 1
        monthly[name] = (1 + r).cumprod().resample("ME").last()
    ew_y = yearly["Hold all liquid stocks"]

    for spec in sorted(REG, key=lambda x: x["no"]):
        if only and spec["no"] not in only:
            continue
        t1 = datetime.now()
        try:
            sig = spec["f"](d)
            r, st = run(d, sig)
        except Exception as e:                                          # keep going, report the failure honestly
            print(f"#{spec['no']:>3} {spec['name']}: FAILED {type(e).__name__}: {e}", flush=True)
            rows.append({k: spec[k] for k in ("no", "name", "category", "rule", "note")} | {"error": str(e)})
            continue
        y = (1 + r).groupby(r.index.year).prod() - 1
        p = periods(r)
        row = {k: spec[k] for k in ("no", "name", "category", "rule", "note")} | p | st
        row["years_beat_hold_all"] = f"{int((y > ew_y.reindex(y.index)).sum())}/{len(y)}"
        rows.append(row)
        yearly[f"{spec['no']}. {spec['name']}"] = y
        monthly[f"{spec['no']}. {spec['name']}"] = (1 + r).cumprod().resample("ME").last()
        print(f"#{spec['no']:>3} {spec['name'][:38]:38s} CAGR {p.get('cagr_pct'):>6} Sharpe {p.get('sharpe'):>5} "
              f"DD {p.get('max_dd_pct'):>6} | win {st['win_rate_pct']}% trades {st['trades']} "
              f"| IS {p.get('is_sharpe')} OOS {p.get('oos_sharpe')}  ({(datetime.now() - t1).seconds}s)", flush=True)

    S = pd.DataFrame(rows)
    S.to_csv(out / "summary.csv", index=False)
    Y = pd.DataFrame(yearly).T.mul(100).round(1)
    Y.columns = [str(c) for c in Y.columns]
    Y.to_csv(out / "yearly.csv")
    pd.DataFrame(monthly).round(4).to_csv(out / "equity_monthly.csv")
    (out / "report.json").write_text(json.dumps({
        "as_of": str(d.idx[-1].date()), "start": START_EVAL, "split_year": SPLIT, "stocks": int(d.S),
        "costs": {"buy": BUY_COST, "sell": SELL_COST, "slippage": "0.1/0.2/0.4 % by liquidity"},
        "method": __doc__}, indent=1))
    print("results ->", out, f"({(datetime.now() - t0).seconds // 60} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
