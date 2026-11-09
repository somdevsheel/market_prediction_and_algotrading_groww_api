"""Minimal in-process metrics (counters + latency summaries) rendered in Prometheus text format.

The worker snapshots its metrics into the ledger (``system_status['metrics']``) each step; the API's
``/metrics`` endpoint renders that snapshot plus ledger-derived gauges.
"""
from __future__ import annotations

import threading
from collections import defaultdict


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.counters: dict[str, float] = defaultdict(float)
        self.lat: dict[str, list[float]] = defaultdict(list)

    def inc(self, name: str, v: float = 1.0) -> None:
        with self._lock:
            self.counters[name] += v

    def observe(self, name: str, seconds: float) -> None:
        with self._lock:
            l = self.lat[name]
            l.append(seconds)
            if len(l) > 500:
                del l[:250]

    def snapshot(self) -> dict:
        with self._lock:
            out = {"counters": dict(self.counters), "latency": {}}
            for k, l in self.lat.items():
                if l:
                    s = sorted(l)
                    out["latency"][k] = {"count": len(s), "p50": s[len(s) // 2], "p95": s[int(len(s) * 0.95) - 1 if len(s) > 1 else 0],
                                         "max": s[-1]}
            return out


METRICS = Metrics()


def _name(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)


def render_prometheus(snapshot: dict, gauges: dict) -> str:
    lines = []
    for k, v in (snapshot or {}).get("counters", {}).items():
        lines += [f"# TYPE nq_{_name(k)}_total counter", f"nq_{_name(k)}_total {v}"]
    for k, d in (snapshot or {}).get("latency", {}).items():
        n = f"nq_{_name(k)}_latency_seconds"
        lines.append(f"# TYPE {n} summary")
        lines.append(f'{n}{{quantile="0.5"}} {d["p50"]}')
        lines.append(f'{n}{{quantile="0.95"}} {d["p95"]}')
        lines.append(f"{n}_count {d['count']}")
    for k, v in gauges.items():
        if v is None:
            continue
        lines += [f"# TYPE nq_{_name(k)} gauge", f"nq_{_name(k)} {float(v)}"]
    return "\n".join(lines) + "\n"
