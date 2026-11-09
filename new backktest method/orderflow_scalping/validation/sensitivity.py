"""Parameter-grid expansion and stability summary."""
from __future__ import annotations

import itertools

import pandas as pd


def expand(default: dict, grid: dict) -> list[dict]:
    keys = list(grid)
    return [{**default, **dict(zip(keys, combo))} for combo in itertools.product(*grid.values())]


def key(p: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(p.items()))


def stability(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    if df.empty or "test_net" not in df:
        return {}
    return {"combos": len(df), "train_positive": int((df.train_net > 0).sum()), "test_positive": int((df.test_net > 0).sum()),
            "test_net_median": round(float(df.test_net.median()), 0), "test_net_best": round(float(df.test_net.max()), 0),
            "test_net_worst": round(float(df.test_net.min()), 0)}
