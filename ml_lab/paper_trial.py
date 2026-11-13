"""One-month monitoring trial of the ML stock model (no real money).

Frozen rules (written to <results>/paper_trial_<id>/plan.json on first run, never changed afterwards):
  * signals: XGBoost BUY/HOLD/SELL from the close of SIGNAL_DATE (saved snapshot)
  * portfolio: top-10 BUY stocks ranked by P(BUY) - P(SELL), Rs 1,00,000, Rs 10,000 each, whole shares,
    bought at the open of START, held to the open of END (monthly rule = no trades in between, like the backtest)
  * charges: Groww delivery (brokerage, STT, stamp, exchange, GST, DP Rs 15.93 per sell) + 0.1-0.4 % slippage
  * compared with: every BUY stock, every HOLD stock, every SELL stock, all liquid stocks (equal weight) and NIFTY 50
  * also: the daily next-day track record of every signal snapshot saved during the trial

    python3 -m ml_lab.paper_trial            # status now (uses the latest prices in the data lake)
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from ml_lab.stocks_bhs import LAKE, ROOT, order_cost, slippage

TRIAL_ID = "2026-11"
SIGNAL_DATE, START, END = "2026-11-11", "2026-11-12", "2026-12-14"   # signals after the 11 Nov close -> first possible buy = 12 Nov open
CAPITAL, SLOTS, MODEL = 100_000.0, 10, "xgboost"


def run_dir():
    return sorted(p for p in (ROOT / "results").glob("bhs_*") if (p / "summary.csv").exists())[-1]


def plan() -> dict:
    out = ROOT / "results" / f"paper_trial_{TRIAL_ID}"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "plan.json"
    if p.exists():
        return json.loads(p.read_text())
    s = pd.read_csv(run_dir() / "signals_history" / f"{SIGNAL_DATE}.csv")
    s["score"] = s[f"{MODEL}_p_buy"] - s[f"{MODEL}_p_sell"]
    top = s[s[f"{MODEL}_signal"] == "BUY"].nlargest(SLOTS, "score")
    pl = {"trial": TRIAL_ID, "frozen_at": datetime.now().isoformat(timespec="seconds"), "model": MODEL,
          "signal_date": SIGNAL_DATE, "start_open": START, "end_open": END, "capital": CAPITAL, "slots": SLOTS,
          "picks": [{"sym": r.sym, "p_buy": float(r[f"{MODEL}_p_buy"]), "score": round(float(r.score), 3), "tv": float(r.tv)}
                    for _, r in top.iterrows()],
          "groups": {g: s.loc[s[f"{MODEL}_signal"] == g, "sym"].tolist() for g in ("BUY", "HOLD", "SELL")},
          "rules": __doc__}
    p.write_text(json.dumps(pl, indent=1))
    return pl


def status() -> dict:
    pl = plan()
    px = pd.read_parquet(LAKE / "silver" / "yahoo_stocks" / "all_daily.parquet", columns=["sym", "date", "open", "close", "name"],
                         filters=[("date", ">=", pd.Timestamp(pl["signal_date"]))])
    names = px.drop_duplicates("sym").set_index("sym").name
    O = px.pivot_table(index="date", columns="sym", values="open")
    C = px.pivot_table(index="date", columns="sym", values="close")
    start, end = pd.Timestamp(pl["start_open"]), pd.Timestamp(pl["end_open"])
    if C.index.max() < start:
        return {"state": "waiting", "msg": f"Trial starts at the open of {pl['start_open']}; prices so far end {C.index.max().date()}.",
                "picks": [names.get(p["sym"], p["sym"]) for p in pl["picks"]]}
    done = C.index.max() >= end
    mark_day = end if done else C.index.max()
    mark = O.loc[end] if done else C.loc[mark_day]                 # final mark = open of END, otherwise latest close
    rows, inv, val, ch_buy, ch_sell = [], 0.0, 0.0, 0.0, 0.0
    per = pl["capital"] / pl["slots"]
    for p in pl["picks"]:
        b = float(O.loc[start].get(p["sym"], np.nan))
        m = float(mark.get(p["sym"], np.nan))
        if not np.isfinite(b):
            continue
        q = int(per // (b * (1 + slippage(p["tv"]))))
        sl = slippage(p["tv"])
        cb, cs = order_cost(q * b, "buy", sl), order_cost(q * m, "sell", sl)
        rows.append({"stock": names.get(p["sym"], p["sym"]), "sym": p["sym"], "qty": q, "buy": round(b, 2), "now": round(m, 2),
                     "invested": round(q * b), "value": round(q * m), "pnl": round(q * (m - b)), "ret_pct": round((m / b - 1) * 100, 2)})
        inv += q * b; val += q * m; ch_buy += cb; ch_sell += cs
    cash = pl["capital"] - inv - ch_buy
    net_if_sold = val - ch_sell + cash
    grp = {}
    for g, syms in pl["groups"].items():
        r = (mark.reindex(syms) / O.loc[start].reindex(syms) - 1).dropna()
        r = r[r.abs() < 0.6]
        grp[g] = {"stocks": int(len(r)), "avg_pct": round(float(r.mean() * 100), 2), "up": int((r > 0).sum()), "down": int((r < 0).sum())}
    allsyms = sum(pl["groups"].values(), [])
    r = (mark.reindex(allsyms) / O.loc[start].reindex(allsyms) - 1).dropna()
    r = r[r.abs() < 0.6]
    grp["ALL liquid stocks"] = {"stocks": int(len(r)), "avg_pct": round(float(r.mean() * 100), 2), "up": int((r > 0).sum()), "down": int((r < 0).sum())}
    n = pd.read_parquet(LAKE / "silver" / "yahoo" / "NIFTY.parquet").set_index("date")
    nm = n.open.get(end) if done else n.close.get(mark_day)
    grp["NIFTY 50"] = {"avg_pct": round(float(nm / n.open[start] - 1) * 100, 2)} if start in n.index and nm else {}
    # daily next-day track record of the snapshots saved during the trial
    hd = run_dir() / "signals_history"
    track = []
    for f in sorted(hd.glob("*.csv")):
        d = pd.Timestamp(f.stem)
        if d < pd.Timestamp(pl["signal_date"]) or d >= C.index.max() or d not in C.index:
            continue
        nx = C.index[C.index > d][0]
        s = pd.read_csv(f)
        s["r"] = s.sym.map(C.loc[nx] / C.loc[d] - 1) * 100
        s = s[s.r.abs() < 40]
        s["score"] = s[f"{pl['model']}_p_buy"] - s[f"{pl['model']}_p_sell"]
        sg = s[f"{pl['model']}_signal"]
        track.append({"signal": f.stem, "checked": str(nx.date()), "top10": round(s[sg == "BUY"].nlargest(10, "score").r.mean(), 2),
                      "BUY": round(s[sg == "BUY"].r.mean(), 2), "HOLD": round(s[sg == "HOLD"].r.mean(), 2),
                      "SELL": round(s[sg == "SELL"].r.mean(), 2), "ALL": round(s.r.mean(), 2)})
    T = pd.DataFrame(track)
    days = int(((C.index >= start) & (C.index <= mark_day)).sum())
    return {"state": "finished" if done else "running", "marked_at": f"{'open' if done else 'close'} of {mark_day.date()}",
            "trading_days": days, "of_days": 21, "portfolio": rows,
            "invested": round(inv), "value": round(val), "gain_before_charges": round(val - inv),
            "gain_pct": round((val / inv - 1) * 100, 2) if inv else None, "buy_charges": round(ch_buy),
            "sell_charges_if_sold": round(ch_sell), "net_profit_if_sold": round(net_if_sold - pl["capital"]),
            "net_return_pct_on_capital": round((net_if_sold / pl["capital"] - 1) * 100, 2), "groups": grp,
            "daily_track": T.to_dict("records"),
            "daily_track_avg": {k: round(float(v), 2) for k, v in T.drop(columns=["signal", "checked"]).mean().items()} if len(T) else {}}


def main() -> int:
    st = status()
    if st["state"] == "waiting":
        print(st["msg"]); print("frozen top-10:", ", ".join(st["picks"])); return 0
    print(f"Trial {TRIAL_ID}: {st['state']} · {st['trading_days']}/{st['of_days']} trading days · marked at {st['marked_at']}")
    print(pd.DataFrame(st["portfolio"]).drop(columns=["sym"]).to_string(index=False))
    print(f"invested Rs {st['invested']:,} -> Rs {st['value']:,} | {st['gain_pct']:+.2f}% (Rs {st['gain_before_charges']:+,}) before charges | "
          f"net if sold now Rs {st['net_profit_if_sold']:+,} ({st['net_return_pct_on_capital']:+.2f}% on Rs 1 lakh, after all charges)")
    print("since the start:", json.dumps(st["groups"]))
    if st["daily_track"]:
        print(pd.DataFrame(st["daily_track"]).to_string(index=False)); print("average:", st["daily_track_avg"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
