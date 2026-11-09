"""Options execution for 9/21 EMA crossover spot trades (three modes), with the prompt's friction model.

Modes: ITM5_BUY (bullish -> buy CE 5 strikes ITM; bearish -> buy PE 5 strikes ITM), ATM_BUY (buy ATM CE/PE),
ATM_SELL (bullish -> SELL ATM PE; bearish -> SELL ATM CE). Strikes from the spot price at the fill bar
(ATM = nearest 50). Expiry: nearest expiry that is not the trade date (configurable).
Fills: option 1-minute bar OPEN at the entry/exit fill bar (same bars as the spot rules). If the option has
no bar at that minute, the last traded close before it is used (recorded as ``stale_fill``).
Slippage: buys pay px*(1+0.5%) + fixed_pts; sells receive max(0.05, px*(1-0.5%) - fixed_pts), tick-rounded.
Costs: full Indian options cost stack via backtesting.costs.charges. Lot: 65 x 1 (fixed, as specified).
Gross P&L uses raw fill prices; friction = slippage + charges; net = gross - friction.
"""
from __future__ import annotations

import math
from datetime import date

import pandas as pd

from backtesting.costs.charges import option_charges
from execution.contract_selector import choose_expiry
from ingestion.historical.downloader import ensure_candles
from ingestion.providers.groww import groww_option_symbol

MODES = ("ITM5_BUY", "ATM_BUY", "ATM_SELL")
TICK = 0.05


def _tick(x: float, up: bool) -> float:
    n = x / TICK
    return round((math.ceil(n - 1e-9) if up else math.floor(n + 1e-9)) * TICK, 2)


def contract_for(mode: str, side: int, spot: float, step: int = 50) -> tuple[int, str, str]:
    atm = int(round(spot / step) * step)
    if mode == "ITM5_BUY":
        return (atm - 5 * step, "CE", "BUY") if side > 0 else (atm + 5 * step, "PE", "BUY")
    if mode == "ATM_BUY":
        return (atm, "CE" if side > 0 else "PE", "BUY")
    return (atm, "PE" if side > 0 else "CE", "SELL")


class EMAOptionSim:
    def __init__(self, provider, lake, calendar, charges_cfg: dict, slip_pct: float = 0.5, slip_pts: float = 0.5,
                 lot: int = 65, avoid_expiry_day: bool = True):
        self.p, self.lake, self.cal, self.ch = provider, lake, calendar, charges_cfg
        self.slip_pct, self.slip_pts, self.lot, self.avoid = slip_pct / 100, slip_pts, lot, avoid_expiry_day
        self.cache: dict = {}

    def candles(self, sym: str, d: date) -> pd.DataFrame:
        k = (sym, d)
        if k not in self.cache:
            try:
                self.cache[k] = ensure_candles(self.p, self.lake, "options", sym, "FNO", "1minute", d, d)
            except Exception:  # noqa: BLE001
                self.cache[k] = pd.DataFrame()
        return self.cache[k]

    @staticmethod
    def px_at(oc: pd.DataFrame, ts) -> tuple[float | None, bool]:
        row = oc[oc["timestamp"] == ts]
        if len(row):
            return float(row["open"].iloc[0]), False
        prev = oc[oc["timestamp"] < ts]
        return (float(prev["close"].iloc[-1]), True) if len(prev) else (None, True)

    def buy_fill(self, px):
        return _tick(px * (1 + self.slip_pct) + self.slip_pts, True)

    def sell_fill(self, px):
        return max(TICK, _tick(px * (1 - self.slip_pct) - self.slip_pts, False))

    def simulate(self, t: dict, mode: str) -> dict:
        d = date.fromisoformat(t["date"])
        exp = choose_expiry(self.cal.around(d), d, self.avoid)
        if exp is None:
            return {"skip": "no_expiry"}
        strike, ot, action = contract_for(mode, t["side"], t["entry_spot"])
        sym = groww_option_symbol("NIFTY", exp, strike, ot)
        oc = self.candles(sym, d)
        if oc.empty:
            return {"skip": "no_option_data", "symbol": sym}
        e_raw, s1 = self.px_at(oc, t["entry_ts"])
        x_raw, s2 = self.px_at(oc, t["exit_ts"])
        if e_raw is None or x_raw is None:
            return {"skip": "no_option_price", "symbol": sym}
        q = self.lot
        if action == "BUY":
            e_fill, x_fill = self.buy_fill(e_raw), self.sell_fill(x_raw)
            gross = (x_raw - e_raw) * q
            ch = option_charges(e_fill * q, x_fill * q, self.ch)
            after_slip = (x_fill - e_fill) * q
        else:
            e_fill, x_fill = self.sell_fill(e_raw), self.buy_fill(x_raw)
            gross = (e_raw - x_raw) * q
            ch = option_charges(x_fill * q, e_fill * q, self.ch)
            after_slip = (e_fill - x_fill) * q
        slippage = gross - after_slip
        net = after_slip - ch.total
        return {"mode": mode, "symbol": sym, "strike": strike, "option_type": ot, "action": action,
                "expiry": str(exp), "dte": (exp - d).days, "opt_entry_raw": e_raw, "opt_exit_raw": x_raw,
                "opt_entry_fill": e_fill, "opt_exit_fill": x_fill, "stale_fill": s1 or s2,
                "gross_pnl": round(gross, 2), "slippage": round(slippage, 2), "charges": ch.to_dict(),
                "friction": round(slippage + ch.total, 2), "net_pnl": round(net, 2)}


def metrics(tr: pd.DataFrame, capital: float = 100_000) -> dict:
    if tr.empty:
        return {"trades": 0}
    n = tr["net_pnl"]
    w, l = n[n > 0], n[n <= 0]
    eq = capital + n.cumsum()
    peak = eq.cummax().clip(lower=capital)
    dd = eq - peak
    return {"trades": int(len(tr)), "win_rate": round(float((n > 0).mean() * 100), 1),
            "net_pnl": round(float(n.sum()), 2), "gross_pnl": round(float(tr["gross_pnl"].sum()), 2),
            "friction": round(float(tr["friction"].sum()), 2),
            "profit_factor": round(float(w.sum() / -l.sum()), 3) if l.sum() < 0 else None,
            "payoff_ratio": round(float(w.mean() / -l.mean()), 3) if len(w) and len(l) and l.mean() < 0 else None,
            "expectancy": round(float(n.mean()), 2), "max_dd_rs": round(float(dd.min()), 2),
            "max_dd_pct": round(float((dd / peak).min() * 100), 2), "capital_basis": capital}
