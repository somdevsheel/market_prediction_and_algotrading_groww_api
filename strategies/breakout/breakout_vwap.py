"""Strategy B — ``breakout_vwap_v1``: opening-range breakout + futures VWAP + volume.

Bullish (mirror for bearish) on the latest *completed* 5-minute NIFTY candle:

1. close > OR_high + buffer, and (if ``require_fresh_breakout``) the previous
   candle closed at or below that level — first confirmed close beyond the level.
2. candle body / range >= ``min_body_ratio`` (breakout strength).
3. close - level <= ``max_extension_points`` (do not chase).
4. near-month futures close > futures session VWAP (``use_vwap_filter``).
5. futures volume of the same candle >= ``volume_multiplier`` x mean of the
   previous ``volume_lookback`` candles (``use_volume_filter``).
6. 15-minute context: last completed 15m close > EMA(context_ema) (``use_context_filter``).
7. regime not in ``block_regimes``.

Invalidation = breakout candle low (high for bearish); target = 2R on the underlying.
Each filter is a parameter so experiments A/B/C... reuse this code unchanged.
"""
from __future__ import annotations

from datetime import time
from typing import Optional

import pandas as pd

from quant.features.indicators import ema, opening_range, resample, session_vwap, volume_expansion
from strategies.base import MarketContext, Signal, Strategy


class BreakoutVWAPStrategy(Strategy):
    strategy_id = "breakout_vwap_v1"

    def __init__(self, params: dict, version: str = "1.0.0"):
        super().__init__(params)
        self.version = version

    def evaluate(self, ctx: MarketContext) -> Optional[Signal]:
        p = self.params
        R = self.last_reasons = []
        idx = ctx.index_5m
        if idx is None or idx.empty:
            R.append("no_index_data")
            return None
        today = ctx.now.date()
        day = idx[idx["timestamp"].dt.date == today].reset_index(drop=True)
        orm = int(p.get("opening_range_minutes", 15))
        if len(day) < orm // 5 + 1:
            R.append("opening_range_incomplete")
            return None
        rng = opening_range(day, orm)
        if rng is None:
            R.append("no_opening_range")
            return None
        or_high, or_low = rng
        width = or_high - or_low
        if not (p.get("min_or_width_points", 0) <= width <= p.get("max_or_width_points", 1e9)):
            R.append(f"or_width_out_of_bounds:{width:.1f}")
            return None
        last, prev = day.iloc[-1], day.iloc[-2]
        if last["timestamp"].time() < (pd.Timestamp("09:15") + pd.Timedelta(minutes=orm)).time():
            R.append("still_inside_opening_range_window")
            return None
        buf = float(p.get("breakout_buffer_points", 0))
        up_lvl, dn_lvl = or_high + buf, or_low - buf
        direction = None
        if last["close"] > up_lvl and (not p.get("require_fresh_breakout", True) or prev["close"] <= up_lvl):
            direction = "BULLISH"
        elif last["close"] < dn_lvl and (not p.get("require_fresh_breakout", True) or prev["close"] >= dn_lvl):
            direction = "BEARISH"
        if direction is None:
            R.append("no_fresh_breakout")
            return None
        reasons = [f"or_breakout_{'up' if direction == 'BULLISH' else 'down'}"]
        rng_c = last["high"] - last["low"]
        body = abs(last["close"] - last["open"])
        body_ok = rng_c > 0 and body / rng_c >= p.get("min_body_ratio", 0.5)
        dir_ok = (last["close"] > last["open"]) if direction == "BULLISH" else (last["close"] < last["open"])
        if not (body_ok and dir_ok):
            R.append("weak_breakout_candle")
            return None
        reasons.append("strong_candle")
        ext = (last["close"] - up_lvl) if direction == "BULLISH" else (dn_lvl - last["close"])
        if ext > p.get("max_extension_points", 1e9):
            R.append(f"overextended:{ext:.1f}")
            return None

        diag = {"or_high": or_high, "or_low": or_low, "close": float(last["close"]), "extension": round(ext, 2)}
        fut = ctx.futures_5m
        if p.get("use_vwap_filter", True) or p.get("use_volume_filter", True):
            if fut is None or fut.empty:
                R.append("no_futures_data")
                return None
            fday = fut[fut["timestamp"].dt.date == today].reset_index(drop=True)
            fbar = fday[fday["timestamp"] == last["timestamp"]]
            if fbar.empty:
                R.append("futures_candle_missing")
                return None
            i = fbar.index[0]
            vw = session_vwap(fday).iloc[i]
            diag["fut_close"], diag["fut_vwap"] = float(fday["close"].iloc[i]), float(vw)
            if p.get("use_vwap_filter", True):
                ok = fday["close"].iloc[i] > vw if direction == "BULLISH" else fday["close"].iloc[i] < vw
                if not ok:
                    R.append("vwap_filter_failed")
                    return None
                reasons.append("futures_vwap_confirm")
            if p.get("use_volume_filter", True):
                vx = volume_expansion(fday["volume"], int(p.get("volume_lookback", 6))).iloc[i]
                diag["volume_ratio"] = None if pd.isna(vx) else round(float(vx), 3)
                if pd.isna(vx) or vx < p.get("volume_multiplier", 1.2):
                    R.append(f"volume_filter_failed:{vx}")
                    return None
                reasons.append("futures_volume_expansion")

        if p.get("use_context_filter", True):
            m15 = resample(idx, 15)
            m15 = m15[m15["_n"] == 3]
            m15 = m15[m15["timestamp"] + pd.Timedelta(minutes=15) <= last["timestamp"] + pd.Timedelta(minutes=5)]
            e = ema(m15["close"], int(p.get("context_ema", 20)))
            if e.dropna().empty:
                R.append("context_insufficient_history")
                return None
            c15, e15 = m15["close"].iloc[-1], e.iloc[-1]
            diag["ctx15_close"], diag["ctx15_ema"] = float(c15), float(e15)
            if (direction == "BULLISH" and c15 <= e15) or (direction == "BEARISH" and c15 >= e15):
                R.append("context_filter_failed")
                return None
            reasons.append("15m_ema_context")

        regime = (ctx.regime or {}).get("regime", "unknown")
        if regime in p.get("block_regimes", []):
            R.append(f"regime_blocked:{regime}")
            return None

        if direction == "BULLISH":
            inval = float(last["low"])
            risk = last["close"] - inval
            target = float(last["close"] + 2 * risk)
        else:
            inval = float(last["high"])
            risk = inval - last["close"]
            target = float(last["close"] - 2 * risk)
        if risk <= 0:
            R.append("non_positive_risk")
            return None
        strength = min(1.0, 0.4 * min(1, body / max(rng_c, 1e-9)) + 0.3 * min(1, ext / max(buf, 1) / 4 + 0.5)
                       + 0.3 * min(1, (diag.get("volume_ratio") or 1) / 2))
        R.append("signal")
        decision_ts = last["timestamp"] + pd.Timedelta(minutes=5)
        return Signal(strategy_id=self.strategy_id, timestamp=decision_ts.to_pydatetime(), underlying="NIFTY",
                      direction=direction, signal_strength=round(float(strength), 3),
                      entry_condition=f"5m close {'>' if direction == 'BULLISH' else '<'} OR "
                                      f"{'high' if direction == 'BULLISH' else 'low'}+buffer ({up_lvl if direction == 'BULLISH' else dn_lvl:.2f})",
                      invalidation_level=inval, target_level=target, reference_price=float(last["close"]),
                      regime=regime, model_version=self.version, reason_codes=reasons, diagnostics=diag)
