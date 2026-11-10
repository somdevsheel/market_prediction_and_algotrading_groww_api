"""ML / DL research on NIFTY (yfinance daily data, 2008 -> today), walk-forward 2013 -> today.

    python3 -m ml_lab.data.yahoo     # refresh data
    python3 -m ml_lab.run            # train + evaluate everything, write ml_lab/results/<ts>/
"""
from __future__ import annotations

import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from ml_lab import walk_forward as wf  # noqa: E402
from ml_lab.features import build_dataset  # noqa: E402

ROOT = __import__("pathlib").Path(__file__).resolve().parent


def main() -> int:
    t0 = datetime.now()
    out = ROOT / "results" / t0.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    d = build_dataset()
    print(f"dataset {d.shape}, {d.date.min().date()} -> {d.date.max().date()}, device {wf.deep.DEV}", flush=True)
    pc = wf.run(d, "clf")
    pr = wf.run(d, "reg")
    pc.to_parquet(out / "pred_direction.parquet")
    pr.to_parquet(out / "pred_volatility.parquet")
    cm = list(wf.classic.classifiers()) + wf.DEEP
    rm = list(wf.classic.regressors()) + wf.DEEP
    rc, rr = wf.clf_report(pc, cm), wf.reg_report(pr, rm)
    rc.to_csv(out / "direction_metrics.csv", index=False)
    rr.to_csv(out / "volatility_metrics.csv", index=False)
    # recent-period check (2024+)
    rc24 = wf.clf_report(pc[pc.date >= "2024-01-01"], cm)
    rr24 = wf.reg_report(pr[pr.date >= "2024-01-01"], rm)
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), dpi=110)
    best = rr.iloc[0]["model"]
    s = pr[pr.date >= "2024-01-01"]
    ax[0].plot(s.date, np.exp(s.y) * 100, lw=0.8, color="#999", label="actual day range %")
    ax[0].plot(s.date, np.exp(s[best]) * 100, lw=1.2, color="#2357d6", label=f"{best} forecast")
    ax[0].plot(s.date, np.exp(s["har_baseline"]) * 100, lw=1.0, color="#e5a93a", label="HAR baseline")
    ax[0].set_title("NIFTY daily range: forecast vs actual (out-of-sample)")
    ax[0].legend(fontsize=8)
    for m in ["logistic", "lightgbm", "lstm", "transformer"]:
        side = np.where(pc[m] > 0.55, 1, np.where(pc[m] < 0.45, -1, 0))
        ax[1].plot(pc.date, np.cumsum(side * pc.y_ret - (side != 0) * 6e-4) * 100, lw=1, label=m)
    ax[1].plot(pc.date, np.cumsum(pc.y_ret) * 100, lw=1, color="#999", label="always long (open->close)")
    ax[1].set_title("Direction models as intraday NIFTY trades (cum. %, 6 bps cost)")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "charts.png")
    rep = {"generated_at": t0.isoformat(), "runtime_min": round((datetime.now() - t0).seconds / 60, 1),
           "dataset": {"rows": len(d), "from": str(d.date.min().date()), "to": str(d.date.max().date()),
                       "features": d.attrs["features"]},
           "direction_all": rc.to_dict("records"), "direction_2024plus": rc24.to_dict("records"),
           "volatility_all": rr.to_dict("records"), "volatility_2024plus": rr24.to_dict("records"),
           "torch": torch.__version__, "device": str(wf.deep.DEV)}
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    pd.set_option("display.width", 200)
    print("\nDIRECTION (open->close), walk-forward 2013->:\n", rc.to_string(index=False))
    print("\nDIRECTION 2024+:\n", rc24[["model", "accuracy", "auc", "strategy_return_pct"]].to_string(index=False))
    print("\nVOLATILITY (log day range), walk-forward 2013->:\n", rr.to_string(index=False))
    print("\nVOLATILITY 2024+:\n", rr24.to_string(index=False))
    print("\nresults ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
