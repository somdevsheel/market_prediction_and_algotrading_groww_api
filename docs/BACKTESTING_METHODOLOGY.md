# Backtesting methodology

* **Data**: real Groww 5-minute candles for NSE-NIFTY, the near-month NIFTY future (last expiry of the month),
  and the *actual option contract* chosen on each signal (including expired contracts).
* **Decision timing**: strategy sees only completed candles up to the decision time (candle start + 5 min).
  Futures candle of the same timestamp is used (it is also complete at that time).
* **Entry**: next 5-min option candle open × (1 + spread/2) + slippage ticks. No historical bid/ask exists →
  spread is an explicit scenario assumption: optimistic 0.2 %, base 0.5 %, conservative 1.0 % (+0/1/3 ticks).
* **Stop sizing**: IV implied from the option open with our BSM solver → delta → structural stop =
  |Δ| × underlying distance to invalidation, capped by the ₹ risk budget, floored at `min_stop_points`.
* **Exits** per candle: gap below stop → open; stop & target in the same candle → stop first (conservative);
  underlying 5-min close beyond invalidation → next open; ≥15:15 → open. Sells × (1 − spread/2) − slippage.
* **Risk**: the live `risk.pre_trade.evaluate` is called for each trade (capital, outlay, planned loss).
* **Costs**: brokerage ₹20/order, STT 0.1 % sell premium, NSE txn 0.03503 %, SEBI, stamp 0.003 % buy, IPFT, GST 18 %.
* **Limits**: 1 trade/day, 1 lot, no re-entry, intraday only. Lot size by date from `lot_size_schedule` (verify).
* **Validation**: chronological IS/OOS split (last 30 % of days OOS), Monte Carlo trade-order shuffle for
  drawdown distribution, bootstrap CI of expectancy, per-month/year/regime/exit-reason/expiry-day breakdowns.

## Active strategy backtest
`scripts/backtest_strategy.py` generates ORB signals with `strategies/orb/orb_core.py` (same code as live), then
the backtester applies the contract-level IV filter, the risk engine and real 1-minute option candles. The
experiment suite (A–I), walk-forward/sensitivity runner, the EMA/ORB-video studies and the order-flow
scalping engine were deleted on 2026-11-10 after failing (recoverable from git history (commit 8a0cab9 and earlier)).

## Metric definitions
net P&L = Σ(gross − charges); ROI = net / capital; win rate = share of trades with net > 0; expectancy = net / trades;
profit factor = Σwins / |Σlosses|; max drawdown on daily-closed equity from starting capital; Sharpe/Sortino on
daily returns over **all** sample days (no-trade days = 0) × √252; CAGR only if sample ≥ ½ year; Calmar = CAGR / |maxDD%|.

## Limitations
Fills are simulated, not historical fills. 5-minute granularity hides intra-candle path. No partial-fill model in
backtests (no historical depth; 1 lot assumed fully filled). Paper trading *does* walk live order-book depth and
can partially fill. Expiry-day contracts are excluded by default (`avoid_expiry_day`).
