"""Gold daily feature table. Every column for day d uses information available BEFORE d's open
(i.e. through d-1's close), except the ``event_*`` tags which are scheduled in advance.

Columns: hv5..hv60, hv_ratio (hv10/hv60), ewma_fc (λ=0.94), garch_fc / gjr_fc (rolling refit, trailing window),
vix_prev, vix_chg, ret_prev, trend_prev (close vs EMA20), atm_iv_prev (if IV history exists), iv_pctile,
rv_iv_ratio (garch_fc / atm_iv_prev), hmm_p0..p{K-1}, hmm_state, hmm_label, event tags.
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from ingestion.macro.events import EventCalendar
from quant.features.indicators import ema
from quant.regime.hmm import GaussianHMM, label_states
from quant.volatility import ewma, garch, hv

log = logging.getLogger(__name__)


def daily_closes(df5: pd.DataFrame) -> pd.DataFrame:
    g = df5.groupby(df5["timestamp"].dt.date)
    return pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last()})


def _rolling_garch(r: pd.Series, model: str, window: int = 750, refit: int = 21, min_obs: int = 250) -> pd.Series:
    """Forecast vol (% ann.) for day i using returns < i. Refit every ``refit`` days."""
    out = pd.Series(np.nan, index=r.index)
    fit = None
    for i in range(min_obs, len(r)):
        if fit is None or (i - min_obs) % refit == 0:
            fit = garch.fit(r.iloc[max(0, i - window):i], model)
        else:  # update filter with the latest return without refitting
            p = fit.params
            last_r = r.iloc[i - 1] * 100
            if model == "garch":
                fit.last_variance = p["omega"] + p["alpha"] * fit.last_return ** 2 + p["beta"] * fit.last_variance
            elif model == "gjr":
                fit.last_variance = (p["omega"] + (p["alpha"] + p["gamma"] * (fit.last_return < 0)) * fit.last_return ** 2
                                     + p["beta"] * fit.last_variance)
            fit.last_return = last_r
        out.iloc[i] = garch.forecast_annual_vol(fit, 1)
    return out


def _rolling_hmm(X: pd.DataFrame, k: int = 3, window: int = 750, refit: int = 63, min_obs: int = 300):
    probs = np.full((len(X), k), np.nan)
    labels = [None] * len(X)
    model = None
    lab = {}
    arr = X.to_numpy()
    for i in range(min_obs, len(X)):
        if model is None or (i - min_obs) % refit == 0:
            model = GaussianHMM(n_states=k, seed=0).fit(arr[max(0, i - window):i])
            lab = label_states(model)
        p = model.predict_next(arr[max(0, i - window):i])  # state for day i given data < i
        probs[i] = p
        labels[i] = lab[int(np.argmax(p))]
    return probs, labels


def build_daily_features(nifty5: pd.DataFrame, vix5: pd.DataFrame | None = None,
                         atm_iv: pd.DataFrame | None = None, is_trading_day=None, hmm_states: int = 3,
                         next_session: date | None = None) -> pd.DataFrame:
    """``next_session`` appends a row for the upcoming session (features from data through the last close)."""
    d = daily_closes(nifty5)
    d.index = pd.to_datetime(d.index)
    if next_session is not None and pd.Timestamp(next_session) > d.index[-1]:
        d.loc[pd.Timestamp(next_session)] = np.nan
    r = hv.log_returns(d.close)
    f = pd.DataFrame(index=d.index)
    tbl = hv.hv_table(d.close)
    for c in tbl:
        f[c.lower()] = tbl[c].shift(1)
    f["hv_ratio"] = f["hv10"] / f["hv60"]
    f["ret_prev"] = r.shift(1)
    f["trend_prev"] = (d.close / ema(d.close, 20) - 1).shift(1)
    f["range_prev"] = ((d.high - d.low) / d.close).shift(1)
    f["gap_known_at_open"] = np.nan  # filled intraday by the engine/backtester if used
    rr = r.iloc[1:].copy()  # keeps a trailing NaN for the next-session row
    known = rr.dropna()
    ev_ = ewma.ewma_variance(known, 0.94)
    ew = np.sqrt(ev_ * 252) * 100  # forecast for t uses r < t
    if rr.isna().iloc[-1] and len(known) and ev_.notna().any():
        nxt = 0.94 * ev_.iloc[-1] + 0.06 * known.iloc[-1] ** 2
        ew = pd.concat([ew, pd.Series([np.sqrt(nxt * 252) * 100], index=[rr.index[-1]])])
    f["ewma_fc"] = ew.reindex(f.index)
    f["garch_fc"] = _rolling_garch(rr, "garch").reindex(f.index)
    f["gjr_fc"] = _rolling_garch(rr, "gjr").reindex(f.index)
    if vix5 is not None and len(vix5):
        v = daily_closes(vix5).close
        v.index = pd.to_datetime(v.index)
        v = v.reindex(f.index)
        f["vix_prev"] = v.shift(1)
        f["vix_chg"] = (v / v.shift(1) - 1).shift(1)
    if atm_iv is not None and len(atm_iv):
        iv = atm_iv.set_index(pd.to_datetime(atm_iv["date"]))["iv_atm_close"]
        f["atm_iv_prev"] = iv.reindex(f.index).shift(1)
        f["iv_pctile"] = f["atm_iv_prev"].rolling(252, min_periods=60).apply(lambda w: (w <= w[-1]).mean() * 100, raw=True)
        f["rv_iv_ratio"] = f["garch_fc"] / f["atm_iv_prev"]
    X = pd.DataFrame({"ret": r, "absret": r.abs(), "range": (d.high - d.low) / d.close}).dropna()
    if next_session is not None and pd.Timestamp(next_session) not in X.index:
        X.loc[pd.Timestamp(next_session)] = np.nan
    probs, labels = _rolling_hmm(X, hmm_states)
    hp = pd.DataFrame(probs, index=X.index, columns=[f"hmm_p{i}" for i in range(hmm_states)])
    hp["hmm_label"] = labels
    # probs[i] uses X < i, i.e. through day i-1 -> already aligned to day i
    f = f.join(hp)
    pc = f[[f"hmm_p{i}" for i in range(hmm_states)]]
    f["hmm_state"] = pc.fillna(-1).idxmax(axis=1).str.replace("hmm_p", "").where(pc.notna().all(axis=1))
    ev = EventCalendar(is_trading_day=is_trading_day)
    f["events"] = [",".join(ev.risk(x.date())) for x in f.index]
    f.index.name = "date"
    return f.reset_index()


def features_for(feats: pd.DataFrame | None, d: date) -> dict:
    if feats is None or feats.empty:
        return {}
    row = feats[feats["date"] == pd.Timestamp(d)]
    if row.empty:
        return {}
    out = row.iloc[0].to_dict()
    return {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in out.items()}
