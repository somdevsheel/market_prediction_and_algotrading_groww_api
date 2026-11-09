"""Historical ATM implied-volatility series from REAL option candles (gold/implied_volatility).

For each trading day: nearest non-expiry-day expiry, ATM strike from the 09:55 index close, CE and PE
5-minute candles; IV solved from option closes at 09:55 and at the last candle (EOD) with BSM.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time

import numpy as np
import pandas as pd

from core.clock import IST
from execution.contract_selector import choose_expiry
from ingestion.historical.downloader import ensure_candles
from ingestion.providers.groww import groww_option_symbol
from quant.options.bsm import implied_vol, year_fraction

log = logging.getLogger(__name__)


def build_iv_history(provider, lake, calendar, nifty5: pd.DataFrame, start: date, end: date, r: float = 0.065,
                     step: int = 50, existing: pd.DataFrame | None = None) -> pd.DataFrame:
    done = set(pd.to_datetime(existing["date"]).dt.date) if existing is not None and len(existing) else set()
    rows = [] if existing is None else existing.to_dict("records")
    days = sorted(d for d in nifty5["timestamp"].dt.date.unique() if start <= d <= end and d not in done)
    for d in days:
        day = nifty5[nifty5["timestamp"].dt.date == d]
        bar = day[day["timestamp"].dt.time == time(9, 55)]
        if bar.empty or len(day) < 10:
            continue
        spot_am, spot_eod = float(bar.close.iloc[0]), float(day.close.iloc[-1])
        exp = choose_expiry(calendar.around(d), d, True)
        if exp is None:
            continue
        k = round(spot_am / step) * step
        rec = {"date": d.isoformat(), "expiry": exp.isoformat(), "strike": k, "spot_1000": spot_am, "spot_eod": spot_eod,
               "days_to_expiry": (exp - d).days}
        ivs_am, ivs_eod = [], []
        for opt in ("CE", "PE"):
            sym = groww_option_symbol("NIFTY", exp, k, opt)
            try:
                oc = ensure_candles(provider, lake, "options", sym, "FNO", "5minute", d, d)
            except Exception as e:  # noqa: BLE001
                log.warning("iv history %s %s: %s", sym, d, e)
                continue
            if oc.empty:
                continue
            ob = oc[oc["timestamp"] == bar["timestamp"].iloc[0]]
            if len(ob):
                t = year_fraction(datetime.combine(d, time(10, 0), IST), exp)
                iv = implied_vol(float(ob.close.iloc[0]), spot_am, k, t, r, opt)
                rec[f"iv_{opt.lower()}_1000"] = iv * 100 if iv else np.nan
                if iv:
                    ivs_am.append(iv * 100)
            last = oc.iloc[-1]
            t2 = year_fraction(last["timestamp"].to_pydatetime() + pd.Timedelta(minutes=5), exp)
            iv2 = implied_vol(float(last.close), spot_eod, k, t2, r, opt)
            if iv2:
                ivs_eod.append(iv2 * 100)
        rec["iv_atm_1000"] = float(np.mean(ivs_am)) if ivs_am else np.nan
        rec["iv_atm_close"] = float(np.mean(ivs_eod)) if ivs_eod else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.sort_values("date").reset_index(drop=True) if len(out) else out
