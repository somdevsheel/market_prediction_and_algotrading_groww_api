"""Consolidated markdown research report (populated only with actual backtest results)."""
from __future__ import annotations

from pathlib import Path


def _f(v, d=0):
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:,.{d}f}"
    return str(v)


def write_markdown(run_dir: Path, rep: dict, S: dict) -> None:
    st = rep["strategies"]
    order = ["ofi", "microprice", "volume_delta", "vwap_momentum", "vwap_mean_reversion", "hybrid",
             "baseline_vwap_cross", "baseline_momentum"]
    L = ["# Order-Flow Scalping Research Report", "",
         f"Data: NIFTY near-month futures, 1-minute OHLCV, {rep['metadata']['sessions']} sessions "
         f"({rep['metadata']['bars']:,} bars). Train ≤ {rep['metadata']['train_end']} ({rep['metadata']['train_sessions']} "
         f"sessions), out-of-sample after ({rep['metadata']['test_sessions']} sessions). 1 lot, base costs: "
         f"half-spread {S['scenarios']['base']['half_spread']} + slippage {S['scenarios']['base']['slippage']} pts per side "
         "plus full futures charges.", "",
         "**PROXY MODE:** no historical Level-2 order book or aggressor-tagged trades exist from Groww. OFI and volume "
         "delta use candle-based proxies; microprice is not testable. These are NOT genuine order-flow backtests.", "",
         "## Out-of-sample comparison (parameters selected on train only)", "",
         "| Strategy | Mode | Net P&L | Profit Factor | Sharpe | Max DD | Trades | Verdict |",
         "|---|---|---:|---:|---:|---:|---:|---|"]
    for k in order:
        r = st.get(k)
        if not r:
            continue
        if r.get("mode") == "NOT TESTABLE":
            L.append(f"| {r['label']} | not testable | — | — | — | — | — | {r['reason']} |")
            continue
        m = r["test"]
        L.append(f"| {r['label']} | {'proxy' if 'PROXY' in r['mode'] else 'OHLCV'} | {_f(m.get('net_pnl'))} | "
                 f"{_f(m.get('profit_factor'), 2)} | {_f(m.get('sharpe'), 2)} | {_f(m.get('max_dd'))} | {_f(m.get('trades'))} | "
                 f"{r.get('verdict', '')} |")
    L += ["", f"Parameter sets tested in total: **{rep['n_parameter_sets_tested']}** (multiple-testing correction "
          "applies Bonferroni over this count).", "", "## Full period, train and walk-forward", "",
          "| Strategy | Full net | Train net | Test net | WF out-of-sample net | Grid: train>0 / test>0 | Avg all-in cost/trade |",
          "|---|---:|---:|---:|---:|---|---:|"]
    for k in order:
        r = st.get(k)
        if not r or r.get("mode") == "NOT TESTABLE":
            continue
        sb = r.get("stability") or {}
        L.append(f"| {r['label']} | {_f(r['full'].get('net_pnl'))} | {_f(r['train'].get('net_pnl'))} | {_f(r['test'].get('net_pnl'))} | "
                 f"{_f(r['walk_forward']['oos'].get('net_pnl'))} | {sb.get('train_positive', '—')}/{sb.get('combos', '—')} · "
                 f"{sb.get('test_positive', '—')}/{sb.get('combos', '—')} | {_f((r.get('costs_test') or {}).get('per_trade_all_in'))} |")
    L += ["", "## Cost scenarios (default parameters, out-of-sample net)", "",
          "| Strategy | Optimistic | Base | Conservative | Limit orders (base) |", "|---|---:|---:|---:|---:|"]
    for k in order:
        r = st.get(k)
        if not r or r.get("mode") == "NOT TESTABLE":
            continue
        s = r["scenarios_default_params"]
        lim = r["limit_order_execution_default_params"]
        L.append(f"| {r['label']} | {_f(s['optimistic']['test'].get('net_pnl'))} | {_f(s['base']['test'].get('net_pnl'))} | "
                 f"{_f(s['conservative']['test'].get('net_pnl'))} | {_f(lim.get('net_pnl'))} (full, {_f(lim.get('nonfills'))} non-fills) |")
    L += ["", "## Execution & cost assumptions", "",
          "- Decision at a bar's close; market orders fill at the NEXT bar's open ± half-spread ± slippage (latency 1 bar).",
          "- Limit orders rest at the signal close and fill only if price trades through by ≥1 tick within 3 bars; otherwise non-fill.",
          "- Stops: stop-market at the stop (or gapped open) minus spread+slippage; if a bar touches stop and target, the stop is assumed first.",
          "- Charges: brokerage ₹20/order, STT 0.02% sell, NSE 0.00173%, SEBI, stamp 0.002% buy, IPFT, GST 18% (verify rates).",
          "- Risk: 1 lot fixed, ATR stops/targets, max hold 30 min, daily loss limit, 5-loss shutdown, square-off 15:20.",
          "", "Per-strategy files: trades.csv, daily_pnl.csv, monthly_returns.csv, equity.png, drawdown.png. "
          "Everything else (sensitivity grids, walk-forward folds, regime/hour breakdowns, cost breakdowns, fitted hybrid "
          "weights) is in summary.json; config_snapshot.yaml + metadata make the run reproducible."]
    (run_dir / "REPORT.md").write_text("\n".join(L) + "\n")
