"""Market clock: IST time, trading sessions and holidays.

A ``Clock`` can be frozen for tests/backtests (``FrozenClock``).
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


class Clock:
    def now(self) -> datetime:
        return datetime.now(IST)

    def today(self) -> date:
        return self.now().date()


class FrozenClock(Clock):
    def __init__(self, dt: datetime):
        self.dt = dt if dt.tzinfo else dt.replace(tzinfo=IST)

    def now(self) -> datetime:
        return self.dt

    def advance(self, **kw) -> None:
        self.dt = self.dt + timedelta(**kw)


class MarketCalendar:
    """Session times and holiday list from config."""

    def __init__(self, settings: dict, instruments: dict):
        s = settings["session"]
        self.open = parse_hhmm(s["market_open"])
        self.close = parse_hhmm(s["market_close"])
        self.entry_start = parse_hhmm(s["entry_start"])
        self.entry_end = parse_hhmm(s["entry_end"])
        self.square_off = parse_hhmm(s["square_off"])
        self.stop_trading = parse_hhmm(s["stop_trading"])
        self.holidays = {date.fromisoformat(d) for d in instruments.get("holidays", [])}

    def is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self.holidays

    def in_session(self, dt: datetime) -> bool:
        t = dt.astimezone(IST).time()
        return self.is_trading_day(dt.date()) and self.open <= t < self.close

    def in_entry_window(self, dt: datetime) -> bool:
        t = dt.astimezone(IST).time()
        return self.is_trading_day(dt.date()) and self.entry_start <= t < self.entry_end

    def past_square_off(self, dt: datetime) -> bool:
        return dt.astimezone(IST).time() >= self.square_off

    def next_trading_day(self, d: date) -> date:
        n = d + timedelta(days=1)
        while not self.is_trading_day(n):
            n += timedelta(days=1)
        return n

    def at(self, d: date, t: time) -> datetime:
        return datetime.combine(d, t, tzinfo=IST)
