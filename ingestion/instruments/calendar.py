"""Contract calendar: expiries per month (from the provider, cached on disk) and near-month futures mapping.

Near-month future on date d = contract with the *last* expiry of d's month if it is >= d, else next month's.
(Valid across the Thursday→Tuesday expiry-day change because it uses the provider's actual expiry list.)
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from ingestion.providers.groww import groww_future_symbol


def next_month(d: date) -> tuple[int, int]:
    n = d.replace(day=28) + timedelta(days=7)
    return n.year, n.month


class ExpiryCalendar:
    def __init__(self, provider, underlying: str = "NIFTY", cache_file: Optional[Path] = None,
                 preload: Optional[dict] = None):
        self.provider, self.underlying = provider, underlying
        self.cache_file = cache_file
        self._c: dict[str, list[str]] = dict(preload or {})
        if cache_file and cache_file.exists():
            self._c = {**json.loads(cache_file.read_text()), **self._c}
        self.errors: list = []

    def month(self, y: int, m: int) -> list[date]:
        k = f"{y}-{m:02d}"
        if k not in self._c:
            if self.provider is None:
                self.errors.append({"month": k, "error": "offline: month not preloaded"})
                return []
            try:
                ex = self.provider.get_expiries(self.underlying, y, m)
            except Exception as e:  # noqa: BLE001
                self.errors.append({"month": k, "error": str(e)})
                return []
            self._c[k] = [e.isoformat() for e in ex]
            if self.cache_file and date(y, m, 1) < date.today().replace(day=1):  # only cache closed months
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                self.cache_file.write_text(json.dumps(self._c))
        return [date.fromisoformat(x) for x in self._c[k]]

    def around(self, d: date) -> list[date]:
        y2, m2 = next_month(d)
        return sorted(set(self.month(d.year, d.month) + self.month(y2, m2)))

    def monthly_expiry(self, y: int, m: int) -> Optional[date]:
        ex = [e for e in self.month(y, m) if e.month == m]
        return max(ex) if ex else None

    def futures_symbol_for(self, d: date) -> Optional[str]:
        for y, m in ((d.year, d.month), next_month(d)):
            e = self.monthly_expiry(y, m)
            if e and e >= d:
                return groww_future_symbol(self.underlying, e)
        return None
