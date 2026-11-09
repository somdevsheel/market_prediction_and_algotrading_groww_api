"""Load 1-minute near-month futures from this experiment's store, validate, and compute all causal features."""
from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

from orderflow_scalping._paths import STORE
from orderflow_scalping.data.validation.validate import validate
from orderflow_scalping.indicators import ofi, regime, volume_delta, vwap


def load_futures(start: date, end: date) -> pd.DataFrame:
    root = STORE / "lake" / "silver" / "futures"
    files = sorted(root.glob("symbol=*/interval=1minute/year=*/month=*/part.parquet"))
    frames = []
    for f in files:
        sym = f.parts[-5].split("=", 1)[1]
        x = pd.read_parquet(f, columns=["timestamp", "open", "high", "low", "close", "volume"])
        x["contract"] = sym
        frames.append(x)
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata") if df["timestamp"].dt.tz is not None \
        else df["timestamp"].dt.tz_localize("Asia/Kolkata")
    df = df[(df.timestamp.dt.date >= start) & (df.timestamp.dt.date <= end)]
    # one contract per day = the near-month contract that was downloaded for that day (most volume wins on overlap)
    vol = df.groupby([df.timestamp.dt.date, "contract"])["volume"].sum().reset_index()
    vol.columns = ["d", "contract", "v"]
    pick = vol.sort_values("v").drop_duplicates("d", keep="last").set_index("d")["contract"]
    df = df[df["contract"].to_numpy() == pick.reindex(df.timestamp.dt.date).to_numpy()]
    return df.sort_values("timestamp").reset_index(drop=True)


def build(start: date, end: date, cache: bool = True) -> tuple[pd.DataFrame, dict]:
    cp = STORE / f"features_{start}_{end}.parquet"
    rp = STORE / f"validation_{start}_{end}.json"
    if cache and cp.exists() and rp.exists():
        import json

        return pd.read_parquet(cp), json.loads(rp.read_text())
    df, rep = validate(load_futures(start, end))
    df["date"] = df["timestamp"].dt.date
    df["minute"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["atr"] = regime.atr(df)
    df["vwap"] = vwap.session_vwap(df)
    df["vwap_z"] = vwap.deviation_z(df, df["vwap"])
    df["vwap_slope"] = vwap.slope(df["vwap"], 10)
    df["ofi_proxy"] = ofi.ofi_proxy(df)
    df["delta_bvc"] = volume_delta.delta_bvc(df)
    df["vol_ma"] = df["volume"].rolling(60, min_periods=20).mean().shift(1)
    df["rel_volume"] = df["volume"] / df["vol_ma"]  # relative volume vs the previous 60 bars (liquidity filter)
    df = regime.add_regime(df, df["vwap_slope"])
    # first bars of each day have no intraday history for VWAP z / rolling stats -> they stay NaN (no trading)
    if cache:
        import json

        cp.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cp, index=False)
        rp.write_text(json.dumps(rep, indent=1, default=str))
    return df, rep
