"""Per-day risk state: daily loss limit, max consecutive losses, max trades/day, entry-window checks.
Position size is fixed (config lots); it is never increased after losses."""
from __future__ import annotations


class DayRisk:
    def __init__(self, r: dict):
        self.r = r
        self.reset()

    def reset(self):
        self.pnl, self.trades, self.streak, self.halted, self.reason = 0.0, 0, 0, False, ""

    def record(self, net: float):
        self.pnl += net
        self.trades += 1
        self.streak = self.streak + 1 if net <= 0 else 0
        if self.pnl <= -self.r["daily_loss_limit"]:
            self.halted, self.reason = True, "daily_loss_limit"
        elif self.streak >= self.r["max_consecutive_losses"]:
            self.halted, self.reason = True, "max_consecutive_losses"
        elif self.trades >= self.r["max_trades_per_day"]:
            self.halted, self.reason = True, "max_trades_per_day"

    def can_enter(self) -> bool:
        return not self.halted
