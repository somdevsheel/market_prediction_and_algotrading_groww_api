"""Per-run outputs: trades CSV, daily/monthly P&L, equity & drawdown PNGs."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from orderflow_scalping.analytics.performance import daily, monthly  # noqa: E402


def write_run(out: Path, trades: pd.DataFrame, capital: float, title: str, split_date: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out / "trades.csv", index=False)
    d = daily(trades)
    d.to_csv(out / "daily_pnl.csv", header=["net_pnl"])
    monthly(trades).to_csv(out / "monthly_returns.csv", header=["net_pnl"])
    if d.empty:
        return
    eq = capital + d.cumsum()
    dd = eq - eq.cummax().clip(lower=capital)
    for name, series, color in (("equity", eq, "#2357d6"), ("drawdown", dd, "#c73a32")):
        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=110)
        ax.plot(pd.to_datetime(series.index), series.values, color=color, lw=1.2)
        if name == "drawdown":
            ax.fill_between(pd.to_datetime(series.index), series.values, 0, color=color, alpha=0.15)
        ax.axvline(pd.Timestamp(split_date), color="#888", ls="--", lw=1)
        ax.text(pd.Timestamp(split_date), ax.get_ylim()[1], "  out-of-sample →", va="top", fontsize=8, color="#555")
        ax.set_title(f"{title} — {name} (₹, 1 lot, net of costs)", fontsize=10)
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / f"{name}.png")
        plt.close(fig)
