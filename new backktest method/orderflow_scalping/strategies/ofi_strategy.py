"""Strategy 1 — OFI scalping (PROXY mode on candles: CLV x volume; genuine CKS OFI needs L2 snapshots)."""
from __future__ import annotations

from orderflow_scalping.indicators.ofi import normalized
from orderflow_scalping.strategies.common import fresh, persist, rz

LABEL = "OFI (proxy)"


def signals(df, p: dict):
    nofi = normalized(df["ofi_proxy"], df["volume"], int(p["window"]))
    z = rz(nofi)
    liquid = df["rel_volume"] >= 0.5
    lc = persist(z > p["entry_z"], int(p["persistence"])) & liquid
    sc = persist(z < -p["entry_z"], int(p["persistence"])) & liquid
    return fresh(lc, sc), None, None
