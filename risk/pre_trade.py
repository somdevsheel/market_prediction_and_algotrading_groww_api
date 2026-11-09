"""Independent pre-trade risk engine. It can veto any proposed trade.

All checks run and are reported (not just the first failure) so every rejection
is fully explainable in the ledger.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from backtesting.costs.charges import estimate_round_trip


@dataclass
class TradePlan:
    mode: str
    trading_symbol: str
    option_type: str
    lot_size: int
    lots: int
    entry_price: float          # expected fill (ask + buffer)
    stop_price: float
    target_price: float
    quote_time: Optional[datetime]
    tick_size: float = 0.05

    @property
    def qty(self) -> int:
        return self.lot_size * self.lots


@dataclass
class RiskDecision:
    approved: bool
    checks: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    planned_loss: float = 0.0
    worst_case_loss: float = 0.0
    est_charges: float = 0.0


@dataclass
class DayState:
    trades_today: int
    realized_pnl_today: float
    open_positions: int
    pending_orders: int
    reconciliation_required: bool


def compute_stop_points(entry: float, delta: Optional[float], underlying_risk: float, qty: int,
                        risk_cfg: dict, charges_cfg: dict, tick: float = 0.05) -> tuple[float, str]:
    """Option-premium stop distance.

    Structural stop = |delta| * underlying distance to invalidation. It is capped
    so that (points * qty + charges + slippage) <= ``risk_per_trade_max`` and
    floored at ``min_stop_points``.
    """
    slip = risk_cfg["slippage_estimate_ticks"] * tick * qty
    charges = estimate_round_trip(entry, entry, qty, charges_cfg)
    budget_pts = (risk_cfg["risk_per_trade_max"] - charges - slip) / qty
    d = abs(delta) if delta else 0.5
    structural = d * abs(underlying_risk)
    pts, how = structural, "structural"
    if pts > budget_pts:
        pts, how = budget_pts, "capped_by_risk_budget"
    if pts < risk_cfg["min_stop_points"]:
        pts, how = risk_cfg["min_stop_points"], "floored_min_stop"
    if pts > entry - tick:  # a long option cannot fall below zero: stop never below one tick
        pts, how = entry - tick, "capped_at_premium"
    import math

    steps = pts / tick
    steps = math.floor(steps + 1e-9) if how == "capped_by_risk_budget" else round(steps)
    pts = max(tick, round(steps * tick, 2))
    return pts, how


def evaluate(plan: TradePlan, now: datetime, day: DayState, funds_available: float, kill_active: bool,
             in_entry_window: bool, risk_cfg: dict, charges_cfg: dict, live_approved: bool) -> RiskDecision:
    f: list[str] = []
    c: dict = {}

    def check(name: str, ok: bool, detail=None):
        c[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            f.append(name if detail is None else f"{name}:{detail}")

    check("mode_valid", plan.mode in ("paper", "live"), plan.mode)
    if plan.mode == "live":
        check("live_approved", live_approved)
    check("kill_switch_off", not kill_active)
    check("entry_window", in_entry_window)
    check("no_reconciliation_pending", not day.reconciliation_required)
    age = (now - plan.quote_time).total_seconds() if plan.quote_time else None
    check("quote_fresh", age is not None and age <= risk_cfg["max_quote_age_seconds"], age)
    check("max_lots", 1 <= plan.lots <= risk_cfg["max_lots"], plan.lots)
    check("daily_trade_count", day.trades_today < risk_cfg["max_trades_per_day"], day.trades_today)
    check("no_existing_position", day.open_positions == 0 and day.pending_orders == 0,
          {"open": day.open_positions, "pending": day.pending_orders})
    check("daily_loss_limit", day.realized_pnl_today > -risk_cfg["daily_loss_limit"], day.realized_pnl_today)
    check("stop_below_entry", 0 < plan.stop_price < plan.entry_price, (plan.stop_price, plan.entry_price))
    check("target_above_entry", plan.target_price > plan.entry_price, plan.target_price)

    qty = plan.qty
    charges = estimate_round_trip(plan.entry_price, max(plan.stop_price, 0), qty, charges_cfg)
    charges *= 1 + risk_cfg.get("charges_buffer_pct", 0) / 100
    slip = risk_cfg["slippage_estimate_ticks"] * plan.tick_size * qty
    planned = (plan.entry_price - plan.stop_price) * qty + charges + slip
    outlay = plan.entry_price * qty + charges
    worst = plan.entry_price * qty + estimate_round_trip(plan.entry_price, 0.0, qty, charges_cfg)
    check("planned_loss_within_budget", planned <= risk_cfg["risk_per_trade_max"] + 1e-6, round(planned, 2))
    check("premium_outlay_limit", outlay <= risk_cfg["max_premium_outlay"], round(outlay, 2))
    check("capital_limit", outlay <= risk_cfg["capital"], round(outlay, 2))
    check("sufficient_funds", outlay <= funds_available, {"needed": round(outlay, 2), "available": funds_available})
    return RiskDecision(approved=not f, checks=c, failures=f, planned_loss=round(planned, 2),
                        worst_case_loss=round(worst, 2), est_charges=round(charges, 2))
