"""IV by strike / expiry: smile, skew, ATM IV, term structure, IV rank/percentile."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from core.models import OptionChain
from quant.options.bsm import greeks, implied_vol, year_fraction


def chain_iv_table(chain: OptionChain, now: datetime, r: float = 0.065, min_price: float = 0.5) -> pd.DataFrame:
    """Model IV per strike from LTPs, alongside provider IV. Illiquid / invalid quotes give NaN."""
    t = year_fraction(now, chain.expiry)
    rows = []
    for row in chain.rows:
        iv = None
        if row.ltp and row.ltp >= min_price and row.volume > 0:
            iv = implied_vol(row.ltp, chain.underlying_ltp, row.strike, t, r, row.option_type)
        g = greeks(chain.underlying_ltp, row.strike, t, r, iv, row.option_type) if iv else None
        rows.append({"strike": row.strike, "type": row.option_type, "ltp": row.ltp, "volume": row.volume,
                     "oi": row.open_interest, "model_iv": iv * 100 if iv else np.nan,
                     "provider_iv": row.iv if row.iv else np.nan,
                     "model_delta": g.delta if g else np.nan, "provider_delta": row.delta,
                     "model_gamma": g.gamma if g else np.nan, "provider_gamma": row.gamma,
                     "model_theta": g.theta if g else np.nan, "provider_theta": row.theta,
                     "model_vega": g.vega if g else np.nan, "provider_vega": row.vega})
    return pd.DataFrame(rows).sort_values(["strike", "type"]).reset_index(drop=True)


def atm_strike(spot: float, step: float = 50) -> float:
    return round(spot / step) * step


def atm_iv(table: pd.DataFrame, spot: float, step: float = 50, col: str = "model_iv") -> float:
    k = atm_strike(spot, step)
    v = table[table.strike == k][col].dropna()
    return float(v.mean()) if len(v) else float("nan")


def smile(table: pd.DataFrame, spot: float, col: str = "model_iv") -> pd.DataFrame:
    """OTM-side smile: puts below spot, calls above spot."""
    otm = table[((table.type == "PE") & (table.strike <= spot)) | ((table.type == "CE") & (table.strike >= spot))]
    return otm[["strike", "type", col]].dropna().reset_index(drop=True)


def skew_25d_proxy(table: pd.DataFrame, spot: float, pct: float = 0.02, col: str = "model_iv") -> float:
    """IV(OTM put at spot*(1-pct)) - IV(OTM call at spot*(1+pct)), nearest strikes."""
    puts = table[(table.type == "PE")].dropna(subset=[col])
    calls = table[(table.type == "CE")].dropna(subset=[col])
    if puts.empty or calls.empty:
        return float("nan")
    p = puts.iloc[(puts.strike - spot * (1 - pct)).abs().argsort().iloc[0]][col]
    c = calls.iloc[(calls.strike - spot * (1 + pct)).abs().argsort().iloc[0]][col]
    return float(p - c)


def iv_rank(series: pd.Series, lookback: int = 252) -> float:
    s = series.dropna().iloc[-lookback:]
    if len(s) < 2 or s.max() == s.min():
        return float("nan")
    return float((s.iloc[-1] - s.min()) / (s.max() - s.min()) * 100)


def iv_percentile(series: pd.Series, lookback: int = 252) -> float:
    s = series.dropna().iloc[-lookback:]
    if s.empty:
        return float("nan")
    return float((s <= s.iloc[-1]).mean() * 100)


def rv_iv_comparison(forecast_rv: float, iv: float) -> dict:
    """Research signal only — never a standalone trade trigger."""
    return {"forecast_rv": forecast_rv, "iv": iv, "rv_minus_iv": forecast_rv - iv,
            "vol_ratio": forecast_rv / iv if iv else float("nan")}


def greeks_comparison(table: pd.DataFrame) -> dict:
    """Mean absolute difference model vs provider Greeks/IV over strikes where both exist."""
    out = {}
    for g in ("iv", "delta", "gamma", "theta", "vega"):
        m, p = (f"model_{g}", f"provider_{g}")
        if m in table and p in table:
            d = (table[m] - table[p]).abs().dropna()
            out[g] = {"mean_abs_diff": round(float(d.mean()), 5) if len(d) else None, "n": int(len(d))}
    return out
