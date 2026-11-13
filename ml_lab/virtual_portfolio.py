"""Virtual (simulated) portfolios: one per ML / DL model, Rs 1,00,000 each, no real money.

Rules (the same as the walk-forward backtest, applied to the daily signal snapshots saved by the refresh):
  * start: signals of START_SIGNAL (close), first buy at the next trading day's open
  * on every rebalance day (the start, then every REBAL_DAYS trading days), using the latest snapshot before that open:
      - sell holdings whose signal is SELL (or that left the liquid universe)
      - keep the rest (HOLD / BUY)
      - fill free slots (max 10) with the highest-ranked BUY stocks (P(BUY) - P(SELL)), equity / 10 each, whole shares
  * trades at the open; Groww delivery charges + 0.1-0.4 % slippage (same cost model as the backtest)
  * every portfolio is marked at each day's close
Two portfolio types per model:
  * top10   : the 10 highest-ranked BUY stocks, whole shares (what you could really buy with Rs 1 lakh)
  * all_buy : EVERY BUY stock, equal amount each, Rs 2 crore per model (CAPITAL_BY_MODE) so that even ~950 BUY stocks get
              >= ~Rs 21,000 each -> whole shares, and Groww's fixed charges (Rs 5 min brokerage, Rs 15.93 DP) stay small
Benchmarks: NIFTY 50 and an equal-weight basket of all liquid stocks (both from the same first open, no charges).

    python3 -m ml_lab.virtual_portfolio
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml_lab.stocks_bhs import LAKE, ROOT, order_cost, slippage

START_SIGNAL = "2026-11-11"
CAPITAL, SLOTS, REBAL_DAYS = 100_000.0, 10, 21
CAPITAL_BY_MODE = {"top10": 100_000.0, "all_buy": 20_000_000.0}   # all-BUY: Rs 2 crore -> >= ~Rs 21k per stock, whole shares
FAMILY = {"logistic": "ML", "knn": "ML", "random_forest": "ML", "extra_trees": "ML", "hist_gb": "ML", "xgboost": "ML",
          "lightgbm": "ML", "catboost": "ML", "mlp": "DL", "lstm": "DL", "gru": "DL", "cnn": "DL", "transformer": "DL",
          "ensemble": "ML+DL"}


def _run_dir() -> Path:
    return sorted(p for p in (ROOT / "results").glob("bhs_*") if (p / "summary.csv").exists())[-1]


def _snapshots() -> dict[pd.Timestamp, Path]:
    hd = _run_dir() / "signals_history"
    return {pd.Timestamp(p.stem): p for p in sorted(hd.glob("*.csv"))} if hd.exists() else {}


def _next_rebalance(first: pd.Timestamp, done: int) -> str | None:
    """Date of the next rebalance open: trading day number k * REBAL_DAYS counted from the first buy (NSE calendar)."""
    try:
        from core.clock import MarketCalendar
        from core.config import load_config

        cfg = load_config()
        cal = MarketCalendar(cfg["settings"], cfg["instruments"])
        k = -(-done // REBAL_DAYS) * REBAL_DAYS          # next multiple of REBAL_DAYS >= days done
        if k == done:
            k += REBAL_DAYS
        x = first.date()
        for _ in range(k):
            x = cal.next_trading_day(x)
        return str(x)
    except Exception:  # noqa: BLE001
        return None


def _run_model(m, mode, days, rebal, snap_before, O, C, names) -> dict:
    """mode 'top10': top-10 BUY, whole shares.  mode 'all_buy': every BUY stock, equal amount each, fractional shares."""
    cap = CAPITAL_BY_MODE[mode]
    cash, hold, trades, charges = cap, {}, [], 0.0               # hold: sym -> {qty, cost, buy_price, ...}
    gross_cash = cap                                         # same trades, but no charges / slippage
    curve, gcurve = [], []
    skipped = 0
    for t in days:
        if t in rebal:
            sd, s = snap_before(t)
            if f"{m}_signal" in s.columns:
                sig = dict(zip(s.sym, s[f"{m}_signal"]))
                tv = dict(zip(s.sym, s.tv))
                for sym in list(hold):                           # sell SELL / out-of-universe at the open
                    p = O.loc[t].get(sym, np.nan)
                    if sig.get(sym, "SELL") == "SELL" and np.isfinite(p):
                        h = hold.pop(sym)
                        val = h["qty"] * p
                        c = order_cost(val, "sell", slippage(tv.get(sym, 0)))
                        cash += val - c
                        gross_cash += val
                        charges += c
                        trades.append({"date": str(t.date()), "side": "SELL", "sym": sym, "qty": h["qty"], "price": round(float(p), 2),
                                       "pnl": round(val - c - h["cost"]), "signal_date": str(sd.date())})
                eq = cash + sum(h["qty"] * O.loc[t].get(k, np.nan) for k, h in hold.items() if np.isfinite(O.loc[t].get(k, np.nan)))
                s = s.assign(score=s[f"{m}_p_buy"] - s[f"{m}_p_sell"])
                cand = s[(s[f"{m}_signal"] == "BUY") & ~s.sym.isin(list(hold))].sort_values("score", ascending=False)
                cand = cand[[np.isfinite(O.loc[t].get(x, np.nan)) and O.loc[t].get(x, 0) > 0 for x in cand.sym]]
                slots = SLOTS if mode == "top10" else len(hold) + len(cand)
                for _, r in cand.iterrows():
                    if len(hold) >= slots:
                        break
                    p = float(O.loc[t][r.sym])
                    sl = slippage(r.tv)
                    budget = min(eq / max(slots, 1), cash)
                    q = int(budget // (p * (1 + sl) * 1.003))                    # whole shares (a stock priced above the budget is skipped)
                    if q <= 0:
                        skipped += 1
                        continue
                    val = q * p
                    c = order_cost(val, "buy", sl)
                    if val + c > cash + 1e-6:
                        continue
                    cash -= val + c
                    gross_cash -= val
                    charges += c
                    hold[r.sym] = {"qty": q, "cost": val + c, "buy_price": round(p, 2), "entry_date": str(t.date()),
                                   "p_buy": round(float(r[f"{m}_p_buy"]), 3)}
                    trades.append({"date": str(t.date()), "side": "BUY", "sym": r.sym, "qty": q, "price": round(p, 2),
                                   "pnl": None, "signal_date": str(sd.date())})
        mv = 0.0
        for k, h in hold.items():
            p = C.loc[t].get(k, np.nan)
            if not np.isfinite(p):                              # no trade today: last known close
                col = C[k].loc[:t].dropna()
                p = col.iloc[-1] if len(col) else h["buy_price"]
            h["last"] = float(p)
            mv += h["qty"] * p
        curve.append({"date": str(t.date()), "value": round(cash + mv, 2)})
        gcurve.append({"date": str(t.date()), "value": round(gross_cash + mv, 2)})
    sell_cost = sum(order_cost(h["qty"] * h["last"], "sell", 0.002) for h in hold.values())
    value, gvalue = curve[-1]["value"], gcurve[-1]["value"]
    return {"model": m, "family": FAMILY.get(m, ""), "mode": mode, "value": round(value),
            "capital": cap, "return_pct": round((value / cap - 1) * 100, 2), "gross_return_pct": round((gvalue / cap - 1) * 100, 2),
            "pnl": round(value - cap), "charges_paid": round(charges), "sell_charges_now": round(sell_cost),
            "net_if_sold": round(value - sell_cost - cap), "net_if_sold_pct": round((value - sell_cost - cap) / cap * 100, 2),
            "skipped_too_expensive": skipped,
            "cash": round(cash), "n_holdings": len(hold), "trades": len(trades),
            "holdings": [{"sym": k, "name": names.get(k, k), "qty": h["qty"], "buy_price": h["buy_price"], "last": round(h["last"], 2),
                          "value": round(h["qty"] * h["last"]), "pnl": round(h["qty"] * h["last"] - h["cost"]),
                          "ret_pct": round((h["last"] / h["buy_price"] - 1) * 100, 2), "entry_date": h["entry_date"], "p_buy": h["p_buy"]}
                         for k, h in sorted(hold.items(), key=lambda kv: -(kv[1]["last"] / kv[1]["buy_price"]))],
            # all-BUY: hundreds of identical opening buys -> keep sells only (each holding shows its own buy date / price)
            "trade_log": trades if mode == "top10" else [x for x in trades if x["side"] == "SELL"],
            "buys": sum(x["side"] == "BUY" for x in trades), "curve": curve, "gross_curve": gcurve}


def simulate(start_signal: str = START_SIGNAL) -> dict:
    snaps = _snapshots()
    if pd.Timestamp(start_signal) not in snaps:
        return {"state": "no_snapshot", "msg": f"No saved signals for {start_signal}."}
    px = pd.read_parquet(LAKE / "silver" / "yahoo_stocks" / "all_daily.parquet", columns=["sym", "date", "open", "close", "name"],
                         filters=[("date", ">=", pd.Timestamp(start_signal))])
    names = px.drop_duplicates("sym").set_index("sym").name.to_dict()
    O = px.pivot_table(index="date", columns="sym", values="open")
    C = px.pivot_table(index="date", columns="sym", values="close")
    days = C.index[C.index > pd.Timestamp(start_signal)]
    if not len(days):
        return {"state": "waiting", "msg": f"First buy happens at the next trading day's open after {start_signal}."}
    rebal = set(days[::REBAL_DAYS])
    cache: dict = {}

    def snap_before(t):
        ds = [d for d in snaps if d < t]
        if not ds:
            return None
        d = max(ds)
        if d not in cache:
            cache[d] = pd.read_csv(snaps[d])
        return d, cache[d]

    first = snap_before(days[0])[1]
    models = sorted({c[: -len("_signal")] for c in first.columns if c.endswith("_signal")})
    out: dict = {"top10": {}, "all_buy": {}}
    for mode in out:
        for m in models:
            out[mode][m] = _run_model(m, mode, days, rebal, snap_before, O, C, names)
    last = C.index.max()
    # benchmarks from the same first open
    t0 = days[0]
    n = pd.read_parquet(LAKE / "silver" / "yahoo" / "NIFTY.parquet").set_index("date")
    nifty_r = [(str(t.date()), float(n.close[t] / n.open[t0])) for t in days if t in n.index and t0 in n.index]
    uni = first.sym.tolist()
    r = (C.loc[days, C.columns.intersection(uni)].div(O.loc[t0, C.columns.intersection(uni)]) - 1)
    r = r.loc[:, r.iloc[-1].abs() < 0.6]
    ew_r = [(str(t.date()), 1 + float(r.loc[t].mean())) for t in days]

    def bench_for(cap):
        cv = lambda xs: [{"date": d_, "value": round(cap * x, 2)} for d_, x in xs]  # noqa: E731
        return {"nifty": {"name": "NIFTY 50", "curve": cv(nifty_r), "return_pct": round((nifty_r[-1][1] - 1) * 100, 2) if nifty_r else None},
                "all": {"name": "All liquid stocks (equal weight)", "curve": cv(ew_r), "return_pct": round((ew_r[-1][1] - 1) * 100, 2)}}
    bench = bench_for(CAPITAL)
    bench_by_mode = {k: bench_for(v) for k, v in CAPITAL_BY_MODE.items()}
    return {"state": "running", "start_signal": start_signal, "first_buy": str(t0.date()), "last_mark": str(last.date()),
            "days": len(days), "next_rebalance": _next_rebalance(days[0], len(days)),
            "rebalance_every_days": REBAL_DAYS, "capital": CAPITAL, "capital_by_mode": CAPITAL_BY_MODE, "slots": SLOTS,
            "benchmarks_by_mode": bench_by_mode,
            "models": out["top10"], "portfolios": out, "benchmarks": bench, "rules": __doc__}


def main() -> int:
    r = simulate()
    if r.get("state") != "running":
        print(r.get("msg")); return 0
    tab = pd.DataFrame([{k: v for k, v in x.items() if k not in ("holdings", "trade_log", "curve")} for x in r["models"].values()])
    print(f"virtual portfolios from {r['start_signal']} signals, first buy {r['first_buy']}, marked {r['last_mark']} ({r['days']} days)")
    print(tab.sort_values("value", ascending=False).to_string(index=False))
    print({k: v["return_pct"] for k, v in r["benchmarks"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
