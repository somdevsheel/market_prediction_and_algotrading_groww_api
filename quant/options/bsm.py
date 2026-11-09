"""Black-Scholes-Merton pricing, implied volatility and Greeks for European index options.

Conventions: ``sigma`` and ``r``/``q`` are decimals (0.15 = 15%); ``t`` in years
(ACT/365, see ``year_fraction``). NIFTY index options are European and cash-settled.
Theta is returned per calendar day, vega and rho per 1 percentage point.
Model prices are never used as executable prices.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional

from scipy.optimize import brentq
from scipy.stats import norm

from core.clock import IST

EXPIRY_TIME = time(15, 30)


def year_fraction(now: datetime, expiry_date, expiry_time: time = EXPIRY_TIME) -> float:
    exp = datetime.combine(expiry_date, expiry_time, IST)
    now = now if now.tzinfo else now.replace(tzinfo=IST)
    return max((exp - now).total_seconds(), 0.0) / (365.0 * 86400)


def _d1d2(s, k, t, r, q, sigma):
    v = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / v
    return d1, d1 - v


def price(s: float, k: float, t: float, r: float, sigma: float, opt: str, q: float = 0.0) -> float:
    if t <= 0 or sigma <= 0:
        return max(0.0, s - k) if opt == "CE" else max(0.0, k - s)
    d1, d2 = _d1d2(s, k, t, r, q, sigma)
    if opt == "CE":
        return s * math.exp(-q * t) * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)
    return k * math.exp(-r * t) * norm.cdf(-d2) - s * math.exp(-q * t) * norm.cdf(-d1)


@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float   # per calendar day
    vega: float    # per 1 vol point
    rho: float     # per 1 rate point


def greeks(s: float, k: float, t: float, r: float, sigma: float, opt: str, q: float = 0.0) -> Greeks:
    if t <= 0 or sigma <= 0:
        itm = (s > k) if opt == "CE" else (s < k)
        return Greeks(delta=(1.0 if itm else 0.0) * (1 if opt == "CE" else -1), gamma=0, theta=0, vega=0, rho=0)
    d1, d2 = _d1d2(s, k, t, r, q, sigma)
    sq = math.sqrt(t)
    pdf = norm.pdf(d1)
    dq, dr = math.exp(-q * t), math.exp(-r * t)
    gamma = dq * pdf / (s * sigma * sq)
    vega = s * dq * pdf * sq / 100
    if opt == "CE":
        delta = dq * norm.cdf(d1)
        theta = (-s * dq * pdf * sigma / (2 * sq) - r * k * dr * norm.cdf(d2) + q * s * dq * norm.cdf(d1)) / 365
        rho = k * t * dr * norm.cdf(d2) / 100
    else:
        delta = -dq * norm.cdf(-d1)
        theta = (-s * dq * pdf * sigma / (2 * sq) + r * k * dr * norm.cdf(-d2) - q * s * dq * norm.cdf(-d1)) / 365
        rho = -k * t * dr * norm.cdf(-d2) / 100
    return Greeks(delta, gamma, theta, vega, rho)


def implied_vol(market_price: float, s: float, k: float, t: float, r: float, opt: str, q: float = 0.0,
                lo: float = 1e-4, hi: float = 5.0) -> Optional[float]:
    """Brent root-finder. Returns None for prices outside no-arbitrage bounds or t<=0."""
    if market_price is None or market_price <= 0 or t <= 0:
        return None
    intrinsic = max(0.0, s * math.exp(-q * t) - k * math.exp(-r * t)) if opt == "CE" else \
        max(0.0, k * math.exp(-r * t) - s * math.exp(-q * t))
    upper = s * math.exp(-q * t) if opt == "CE" else k * math.exp(-r * t)
    if market_price < intrinsic - 1e-9 or market_price >= upper:
        return None
    f = lambda v: price(s, k, t, r, v, opt, q) - market_price  # noqa: E731
    try:
        if f(lo) > 0:
            return lo
        if f(hi) < 0:
            return None
        return brentq(f, lo, hi, xtol=1e-7, maxiter=200)
    except (ValueError, RuntimeError):
        return None
