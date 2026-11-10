"""Daily portfolio backtester for swing / positional equity strategies (delivery, long-only).

Timeline: signals computed from data up to day t's CLOSE; orders execute at day t+1's OPEN with slippage
(buys pay up, sells receive less). Whole shares only, cash never negative, max N positions, equal-sized slots.
Positions are marked to market at each close. Charges = Groww delivery schedule (equity/costs.py) per order.
Survivorship bias: the universe is today's listed stocks only (delisted names absent) -> results are optimistic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from equity.costs import DELIVERY, order_charges


@dataclass
class Result:
    equity: pd.Series
    trades: pd.DataFrame
    params: dict = field(default_factory=dict)


def run(panel: dict, strategy, params: dict, capital: float = 100_000, slippage_bps: float = 10,
        max_positions: int = 10, start: str | None = None, end: str | None = None) -> Result:
    O, C = panel["open"], panel["close"]
    dates = C.index
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    if end:
        dates = dates[dates <= pd.Timestamp(end)]
    sig = strategy.prepare(panel, params)          # precomputed causal indicators
    cash, pos = capital, {}                          # pos: sym -> dict(qty, entry_px, entry_date)
    eq, trades = [], []
    pend_buy, pend_sell = [], []
    slip = slippage_bps / 1e4
    for i, d in enumerate(dates):
        o = O.loc[d]
        # 1) execute yesterday's decisions at today's open
        for s in pend_sell:
            if s in pos and np.isfinite(o.get(s, np.nan)):
                p = pos.pop(s)
                px = o[s] * (1 - slip)
                val = px * p["qty"]
                ch = order_charges(val, "sell", DELIVERY)
                cash += val - ch
                pnl = (px - p["entry_px"]) * p["qty"] - ch - p["buy_charges"]
                trades.append({"sym": s, "entry_date": p["entry_date"], "exit_date": d, "qty": p["qty"],
                               "entry_px": round(p["entry_px"], 2), "exit_px": round(px, 2),
                               "charges": round(ch + p["buy_charges"], 2), "net_pnl": round(pnl, 2),
                               "ret_pct": round(pnl / (p["entry_px"] * p["qty"]) * 100, 2),
                               "hold_days": (d - p["entry_date"]).days})
        pend_sell = []
        slots = max_positions - len(pos)
        if pend_buy and slots > 0:
            port_val = cash + sum(p["qty"] * p["last_px"] for p in pos.values())  # last known prices (no NaN)
            alloc = port_val / max_positions
            for s in pend_buy[:slots]:
                px = o.get(s, np.nan)
                if not np.isfinite(px) or s in pos:
                    continue
                px *= (1 + slip)
                qty = int(min(alloc, cash) // (px * 1.002))
                if qty < 1:
                    continue
                val = qty * px
                ch = order_charges(val, "buy", DELIVERY)
                if val + ch > cash:
                    continue
                cash -= val + ch
                pos[s] = {"qty": qty, "entry_px": px, "entry_date": d, "buy_charges": ch, "last_px": px}
        pend_buy = []
        # 2) mark to market at close
        c = C.loc[d]
        for s, p in pos.items():  # carry the last valid close for suspended / missing days
            if np.isfinite(c.get(s, np.nan)):
                p["last_px"] = c[s]
        mv = sum(p["qty"] * p["last_px"] for p in pos.values())
        eq.append((d, cash + mv))
        # 3) decisions at close (data up to and including d)
        sells, buys = strategy.decide(d, sig, set(pos), pos, params)
        pend_sell = [s for s in sells if s in pos]
        pend_buy = [s for s in buys if s not in pos]
    for s, p in pos.items():  # open positions at the end: valued at last close (no forced sale)
        trades.append({"sym": s, "entry_date": p["entry_date"], "exit_date": None, "qty": p["qty"],
                       "entry_px": round(p["entry_px"], 2), "exit_px": None, "charges": round(p["buy_charges"], 2),
                       "net_pnl": None, "ret_pct": None, "hold_days": None})
    return Result(pd.Series(dict(eq)), pd.DataFrame(trades), params)


def metrics(equity: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    if equity.empty:
        return {}
    r = equity.pct_change().dropna()
    yrs = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / yrs) - 1
    dd = equity / equity.cummax() - 1
    out = {"start_value": round(float(equity.iloc[0])), "end_value": round(float(equity.iloc[-1])),
           "total_return_pct": round(float((equity.iloc[-1] / equity.iloc[0] - 1) * 100), 1),
           "cagr_pct": round(float(cagr * 100), 1), "max_dd_pct": round(float(dd.min() * 100), 1),
           "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2) if r.std() > 0 else None,
           "sortino": round(float(r.mean() / r[r < 0].std() * np.sqrt(252)), 2) if (r < 0).sum() > 1 else None,
           "calmar": round(float(cagr / abs(dd.min())), 2) if dd.min() < 0 else None}
    if trades is not None and len(trades):
        t = trades.dropna(subset=["net_pnl"])
        out.update({"trades": int(len(t)), "win_rate_pct": round(float((t.net_pnl > 0).mean() * 100), 1) if len(t) else None,
                    "avg_hold_days": round(float(t.hold_days.mean()), 1) if len(t) else None,
                    "charges": round(float(trades.charges.sum())),
                    "profit_factor": round(float(t[t.net_pnl > 0].net_pnl.sum() / -t[t.net_pnl <= 0].net_pnl.sum()), 2)
                    if (t.net_pnl <= 0).any() else None})
    return out
