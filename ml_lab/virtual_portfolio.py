"""Virtual (simulated) portfolios: one per ML / DL model, Rs 1,00,000 each, no real money.

Rules (the same as the walk-forward backtest, applied to the daily signal snapshots saved by the refresh):
  * start: signals of START_SIGNAL (close), first buy at the next trading day's open
  * on every rebalance day (the start, then every REBAL_DAYS trading days), using the latest snapshot before that open:
      - sell holdings whose signal is SELL (or that left the liquid universe)
      - keep the rest (HOLD / BUY)
      - fill free slots (max 10) with the highest-ranked BUY stocks (P(BUY) - P(SELL)), equity / 10 each, whole shares
  * trades at the open; Groww delivery charges + 0.1-0.4 % slippage (same cost model as the backtest)
  * every portfolio is marked at each day's close
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
    out = {}
    for m in models:
        cash, hold, trades, charges = CAPITAL, {}, [], 0.0      # hold: sym -> {qty, cost, entry_date}
        curve = []
        for t in days:
            if t in rebal:
                sd, s = snap_before(t)
                if f"{m}_signal" in s.columns:
                    sig = dict(zip(s.sym, s[f"{m}_signal"]))
                    tv = dict(zip(s.sym, s.tv))
                    for sym in list(hold):                     # sell SELL / out-of-universe at the open
                        p = O.loc[t].get(sym, np.nan)
                        if sig.get(sym, "SELL") == "SELL" and np.isfinite(p):
                            h = hold.pop(sym)
                            val = h["qty"] * p
                            c = order_cost(val, "sell", slippage(tv.get(sym, 0)))
                            cash += val - c
                            charges += c
                            trades.append({"date": str(t.date()), "side": "SELL", "sym": sym, "qty": h["qty"], "price": round(float(p), 2),
                                           "pnl": round(val - c - h["cost"]), "signal_date": str(sd.date())})
                    eq = cash + sum(h["qty"] * O.loc[t].get(k, np.nan) for k, h in hold.items() if np.isfinite(O.loc[t].get(k, np.nan)))
                    s = s.assign(score=s[f"{m}_p_buy"] - s[f"{m}_p_sell"])
                    for _, r in s[(s[f"{m}_signal"] == "BUY") & ~s.sym.isin(list(hold))].sort_values("score", ascending=False).iterrows():
                        if len(hold) >= SLOTS:
                            break
                        p = O.loc[t].get(r.sym, np.nan)
                        if not np.isfinite(p) or p <= 0:
                            continue
                        sl = slippage(r.tv)
                        q = int(min(eq / SLOTS, cash) // (p * (1 + sl) * 1.003))
                        if q <= 0:
                            continue
                        val = q * p
                        c = order_cost(val, "buy", sl)
                        if val + c > cash:
                            continue
                        cash -= val + c
                        charges += c
                        hold[r.sym] = {"qty": q, "cost": val + c, "buy_price": round(float(p), 2), "entry_date": str(t.date()),
                                       "p_buy": round(float(r[f"{m}_p_buy"]), 3)}
                        trades.append({"date": str(t.date()), "side": "BUY", "sym": r.sym, "qty": q, "price": round(float(p), 2),
                                       "pnl": None, "signal_date": str(sd.date())})
            mv = 0.0
            for k, h in hold.items():
                p = C.loc[t].get(k, np.nan)
                if not np.isfinite(p):                          # no trade today: last known close
                    col = C[k].loc[:t].dropna()
                    p = col.iloc[-1] if len(col) else h["buy_price"]
                h["last"] = float(p)
                mv += h["qty"] * p
            curve.append({"date": str(t.date()), "value": round(cash + mv, 2)})
        last = C.index.max()
        sell_cost = sum(order_cost(h["qty"] * h["last"], "sell", 0.002) for h in hold.values())
        value = curve[-1]["value"]
        out[m] = {"model": m, "family": FAMILY.get(m, ""), "value": round(value), "return_pct": round((value / CAPITAL - 1) * 100, 2),
                  "pnl": round(value - CAPITAL), "charges_paid": round(charges), "net_if_sold": round(value - sell_cost - CAPITAL),
                  "cash": round(cash), "n_holdings": len(hold), "trades": len(trades),
                  "holdings": [{"sym": k, "name": names.get(k, k), "qty": h["qty"], "buy_price": h["buy_price"], "last": round(h["last"], 2),
                                "value": round(h["qty"] * h["last"]), "pnl": round(h["qty"] * h["last"] - h["cost"]),
                                "ret_pct": round((h["last"] / h["buy_price"] - 1) * 100, 2), "entry_date": h["entry_date"], "p_buy": h["p_buy"]}
                               for k, h in sorted(hold.items(), key=lambda kv: -kv[1]["qty"] * kv[1]["last"])],
                  "trade_log": trades, "curve": curve}
    # benchmarks from the same first open
    t0 = days[0]
    n = pd.read_parquet(LAKE / "silver" / "yahoo" / "NIFTY.parquet").set_index("date")
    nifty = [{"date": str(t.date()), "value": round(CAPITAL * float(n.close[t] / n.open[t0]), 2)} for t in days if t in n.index and t0 in n.index]
    uni = first.sym.tolist()
    r = (C.loc[days, C.columns.intersection(uni)].div(O.loc[t0, C.columns.intersection(uni)]) - 1)
    r = r.loc[:, r.iloc[-1].abs() < 0.6]
    ew = [{"date": str(t.date()), "value": round(CAPITAL * (1 + float(r.loc[t].mean())), 2)} for t in days]
    bench = {"nifty": {"name": "NIFTY 50", "curve": nifty, "return_pct": round((nifty[-1]["value"] / CAPITAL - 1) * 100, 2) if nifty else None},
             "all": {"name": "All liquid stocks (equal weight)", "curve": ew, "return_pct": round((ew[-1]["value"] / CAPITAL - 1) * 100, 2)}}
    return {"state": "running", "start_signal": start_signal, "first_buy": str(t0.date()), "last_mark": str(last.date()),
            "days": len(days), "next_rebalance": _next_rebalance(days[0], len(days)),
            "rebalance_every_days": REBAL_DAYS, "capital": CAPITAL, "slots": SLOTS,
            "models": out, "benchmarks": bench, "rules": __doc__}


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
