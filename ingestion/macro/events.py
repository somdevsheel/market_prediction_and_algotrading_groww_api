"""Event-risk calendar (filter only). ``EventCalendar.risk(d)`` -> list of event tags affecting session d."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import yaml

from core.config import CONFIG_DIR


class EventCalendar:
    def __init__(self, path: Path | None = None, is_trading_day=None):
        cfg = yaml.safe_load((path or CONFIG_DIR / "events.yaml").read_text()) or {}
        self.fixed: dict[date, list[str]] = {}
        for key, tag in (("rbi_policy_decisions", "rbi_policy"), ("union_budget", "budget"),
                         ("election_results", "election"), ("custom", "custom")):
            for d in cfg.get(key) or []:
                self.fixed.setdefault(date.fromisoformat(str(d)), []).append(tag)
        self.cpi_dom = cfg.get("cpi_release_day_of_month")
        self.cpi_next = cfg.get("cpi_affects_next_session", True)
        self.is_trading_day = is_trading_day or (lambda d: d.weekday() < 5)

    def _cpi_session(self, y: int, m: int) -> date | None:
        if not self.cpi_dom:
            return None
        d = date(y, m, self.cpi_dom)
        if self.cpi_next:
            d += timedelta(days=1)
            while not self.is_trading_day(d):
                d += timedelta(days=1)
        return d

    def risk(self, d: date) -> list[str]:
        tags = list(self.fixed.get(d, []))
        for y, m in ((d.year, d.month), ((d.replace(day=1) - timedelta(days=1)).year, (d.replace(day=1) - timedelta(days=1)).month)):
            if self._cpi_session(y, m) == d:
                tags.append("cpi")
        return tags
