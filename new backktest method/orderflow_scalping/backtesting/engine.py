"""Event-driven bar-replay backtester (1-minute bars).

Timeline per bar t:  (1) pending entry fills at bar t's open (or limit trade-through within ttl)
                     (2) open position: stop / target checked against bar t's range (stop first if both)
                     (3) at bar t's CLOSE: strategy exit signal / max holding / square-off schedule an exit
                         for bar t+1's open; flat + entry signal schedules an entry for bar t+1
No decision uses information after bar t's close; fills never use bar t's close for an order decided at t.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from orderflow_scalping.backtesting.costs import futures_charges
from orderflow_scalping.backtesting.execution import market_fill, stop_fill
from orderflow_scalping.backtesting.risk_manager import DayRisk


def lot_size_on(d: date, schedule: list[dict]) -> int:
    lot = schedule[0]["lot_size"]
    for s in sorted(schedule, key=lambda x: x["from"]):
        if d >= date.fromisoformat(s["from"]):
            lot = s["lot_size"]
    return int(lot)


def _mins(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def run(df: pd.DataFrame, entry: np.ndarray, settings: dict, scenario: str = "base",
        exit_long: np.ndarray | None = None, exit_short: np.ndarray | None = None, label: str = "") -> pd.DataFrame:
    ex, rk, cs = settings["execution"], settings["risk"], settings["costs"]
    sc = settings["scenarios"][scenario]
    hs, sl = sc["half_spread"], sc["slippage"]
    tick = ex["tick"]
    first, last, sq = _mins(settings["session"]["first_entry"]), _mins(settings["session"]["last_entry"]), _mins(settings["session"]["square_off"])
    limit_mode = ex["order_type"] == "limit"
    ttl, thru = ex["limit_ttl_bars"], ex["limit_trade_through_ticks"] * tick
    O, H, L, C = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    A = df["atr"].to_numpy(float)
    M = df["minute"].to_numpy()
    D = df["date"].to_numpy()
    TS = df["timestamp"].to_numpy()
    REG = df["regime"].to_numpy() if "regime" in df else np.full(len(df), "")
    xl = exit_long if exit_long is not None else np.zeros(len(df), bool)
    xs = exit_short if exit_short is not None else np.zeros(len(df), bool)
    lots = settings["lots"]
    sched = settings["lot_size_schedule"]
    trades, nonfills = [], 0
    risk = DayRisk(rk)
    pos = 0
    pend = None          # (side, decision_idx, stop_dist, tgt_dist, limit_px)
    pend_exit = None     # reason string
    cur_day = None
    qty = 0
    n = len(df)
    for t in range(n):
        if D[t] != cur_day:
            if pos != 0:  # safety: never carry overnight (should already be squared off)
                pend_exit = "eod_safety"
            cur_day = D[t]
            risk.reset()
            qty = lot_size_on(cur_day, sched) * lots
        # (1a) scheduled exit at this bar's open
        if pos != 0 and pend_exit is not None:
            px = market_fill(O[t], -pos, hs, sl)
            trades.append(_close(tr, t, px, pend_exit, TS, qty, cs, hs + sl))
            risk.record(trades[-1]["net_pnl"])
            pos, pend_exit = 0, None
        # (1b) pending entry
        if pend is not None and pos == 0:
            side, di, sd, td, lpx = pend
            filled = None
            if not limit_mode:
                if t == di + 1:
                    filled = market_fill(O[t], side, hs, sl)
                pend = None
            else:
                if (side > 0 and L[t] <= lpx - thru) or (side < 0 and H[t] >= lpx + thru):
                    filled = lpx
                    pend = None
                elif t - di >= ttl:
                    nonfills += 1
                    pend = None
            if filled is not None:
                pos = side
                tr = {"entry_idx": t, "entry_time": TS[t], "decision_time": TS[di], "side": side, "entry": filled,
                      "stop": filled - side * sd, "target": filled + side * td, "regime": REG[di],
                      "mae": 0.0, "mfe": 0.0, "qty": qty, "entry_cost_pts": (hs + sl) if not limit_mode else 0.0}
        # (2) stop / target inside this bar
        if pos != 0:
            tr["mae"] = min(tr["mae"], (L[t] - tr["entry"]) if pos > 0 else (tr["entry"] - H[t]))
            tr["mfe"] = max(tr["mfe"], (H[t] - tr["entry"]) if pos > 0 else (tr["entry"] - L[t]))
            hit_stop = L[t] <= tr["stop"] if pos > 0 else H[t] >= tr["stop"]
            hit_tgt = H[t] >= tr["target"] + thru if pos > 0 else L[t] <= tr["target"] - thru
            if hit_stop:
                px = stop_fill(tr["stop"], O[t], pos, hs, sl)
                trades.append(_close(tr, t, px, "stop", TS, qty, cs, hs + sl))
                risk.record(trades[-1]["net_pnl"])
                pos, pend_exit = 0, None
            elif hit_tgt:
                trades.append(_close(tr, t, tr["target"], "target", TS, qty, cs, 0.0))
                risk.record(trades[-1]["net_pnl"])
                pos, pend_exit = 0, None
        # (3) decisions at this bar's close
        nxt_same_day = t + 1 < n and D[t + 1] == D[t]
        if pos != 0:
            if not nxt_same_day or M[t] + 1 >= sq:
                pend_exit = "square_off"
            elif (pos > 0 and xl[t]) or (pos < 0 and xs[t]):
                pend_exit = "exit_signal"
            elif t - tr["entry_idx"] + 1 >= rk["max_holding_bars"]:
                pend_exit = "max_holding"
        elif pend is None and entry[t] != 0 and nxt_same_day and first <= M[t] < last and risk.can_enter() \
                and np.isfinite(A[t]) and A[t] > 0:
            pend = (int(entry[t]), t, rk["stop_atr_mult"] * A[t], rk["target_atr_mult"] * A[t], C[t])
    out = pd.DataFrame(trades)
    if len(out):
        out["strategy"] = label
        out.attrs["nonfills"] = nonfills
    return out


def _close(tr: dict, t: int, px: float, reason: str, TS, qty: int, cs: dict, exit_cost_pts: float) -> dict:
    side = tr["side"]
    buy_px, sell_px = (tr["entry"], px) if side > 0 else (px, tr["entry"])
    ch = futures_charges(buy_px * qty, sell_px * qty, cs)
    gross = side * (px - tr["entry"]) * qty
    return {"entry_time": tr["entry_time"], "decision_time": tr["decision_time"], "exit_time": TS[t],
            "side": side, "entry": round(tr["entry"], 2), "exit": round(px, 2), "qty": qty, "exit_reason": reason,
            "gross_pnl": round(gross, 2), "charges": round(ch["total"], 2), "net_pnl": round(gross - ch["total"], 2),
            "slippage_cost": round((tr["entry_cost_pts"] + exit_cost_pts) * qty, 2),
            **{f"ch_{k}": round(v, 2) for k, v in ch.items() if k != "total"},
            "mae_pts": round(tr["mae"], 2), "mfe_pts": round(tr["mfe"], 2), "regime": tr["regime"],
            "hold_minutes": int((TS[t] - tr["entry_time"]) / np.timedelta64(1, "m"))}
