"""Data-quality checks. Invalid rows are *quarantined*, never silently dropped."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import time

import pandas as pd

SESSION_START, SESSION_END = time(9, 15), time(15, 30)


@dataclass
class QualityReport:
    rows_in: int = 0
    rows_valid: int = 0
    duplicates: int = 0
    invalid_ohlc: int = 0
    non_positive_prices: int = 0
    negative_volume: int = 0
    outside_session: int = 0
    missing_intervals: int = 0
    trading_days: int = 0
    missing_by_day: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_candles(df: pd.DataFrame, interval_minutes: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, QualityReport]:
    """Return ``(valid, quarantined, report)``.

    Missing intervals are counted against the regular 09:15-15:30 grid for
    intraday intervals; nothing is fabricated to fill them.
    """
    rep = QualityReport(rows_in=len(df))
    if df.empty:
        return df, df, rep
    df = df.copy()
    dup_mask = df.duplicated("timestamp", keep="first")
    rep.duplicates = int(dup_mask.sum())
    df = df[~dup_mask]
    reasons = pd.Series("", index=df.index)
    nonpos = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | df[["open", "high", "low", "close"]].isna().any(axis=1)
    bad = (df["high"] < df[["open", "close", "low"]].max(axis=1)) | (df["low"] > df[["open", "close", "high"]].min(axis=1))
    negvol = df["volume"].fillna(0) < 0
    t = df["timestamp"].dt.time
    outside = (t < SESSION_START) | (t >= SESSION_END) if interval_minutes and interval_minutes < 1440 else pd.Series(False, index=df.index)
    reasons[nonpos] += "non_positive_price;"
    reasons[bad & ~nonpos] += "invalid_ohlc;"
    reasons[negvol] += "negative_volume;"
    reasons[outside] += "outside_session;"
    rep.non_positive_prices, rep.invalid_ohlc = int(nonpos.sum()), int((bad & ~nonpos).sum())
    rep.negative_volume, rep.outside_session = int(negvol.sum()), int(outside.sum())
    q_mask = reasons != ""
    quarantined = df[q_mask].assign(quarantine_reason=reasons[q_mask])
    valid = df[~q_mask].sort_values("timestamp").reset_index(drop=True)
    rep.rows_valid = len(valid)
    if interval_minutes and interval_minutes < 1440 and not valid.empty:
        per_day = 375 // interval_minutes + (1 if 375 % interval_minutes else 0)
        counts = valid.groupby(valid["timestamp"].dt.date).size()
        rep.trading_days = len(counts)
        for d, n in counts.items():
            if n < per_day:
                rep.missing_by_day[str(d)] = int(per_day - n)
        rep.missing_intervals = int(sum(rep.missing_by_day.values()))
    return valid, quarantined, rep
