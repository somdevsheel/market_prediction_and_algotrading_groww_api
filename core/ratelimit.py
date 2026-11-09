"""Thread-safe sliding-window rate limiter (per-second and per-minute caps)."""
from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, per_second: int, per_minute: int, sleep=time.sleep, clock=time.monotonic):
        self.per_second, self.per_minute = per_second, per_minute
        self._hits: deque[float] = deque()
        self._lock = threading.Lock()
        self._sleep, self._clock = sleep, clock

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = self._clock()
                while self._hits and now - self._hits[0] > 60:
                    self._hits.popleft()
                last_sec = sum(1 for t in self._hits if now - t < 1)
                if last_sec < self.per_second and len(self._hits) < self.per_minute:
                    self._hits.append(now)
                    return
                wait = 0.05 if last_sec >= self.per_second else max(0.05, 60 - (now - self._hits[0]))
            self._sleep(wait)
