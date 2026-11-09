"""Strategy 3 — Volume delta scalping (PROXY: bulk volume classification; genuine needs aggressor trades)."""
from __future__ import annotations

from orderflow_scalping.indicators.volume_delta import cumulative
from orderflow_scalping.strategies.common import fresh, rz

LABEL = "Volume Delta (proxy)"


def signals(df, p: dict):
    w = int(p["window"])
    cd = cumulative(df["delta_bvc"], w)
    z = rz(cd)
    acc = cd.diff(max(1, w // 2))
    active = df["rel_volume"] >= 0.8  # avoid insufficient activity
    lc = (z > p["entry_z"]) & active & ((acc > 0) if p["require_accel"] else True)
    sc = (z < -p["entry_z"]) & active & ((acc < 0) if p["require_accel"] else True)
    return fresh(lc, sc), None, None
