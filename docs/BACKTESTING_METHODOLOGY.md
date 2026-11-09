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

## Experiments (config only, no code changes)
| id | strategy | definition |
|---|---|---|
| A | sr_breakout_v1 | prior-day H/L, opening range, confirmed swing levels |
| B | breakout_vwap_v1 | OR breakout + futures VWAP + volume + 15m EMA context |
| C | breakout_retest_v1 | OR breakout → retest holds → confirmation |
| D | vol_filtered_v1 | B + HV ratio band, GARCH forecast floor, VIX band, RV/IV ≥ 0.9, contract IV ≤ 1.25×forecast, event block |
| E | regime_filtered_v1 | B + HMM regime aligned with direction + event block |
| H | ml_confirmed_v1 | B + walk-forward ML probability ≥ 0.45 (probabilities are strictly out-of-sample) |
| I | combined_v1 | B + D + E + H filters |
D/E/H/I are evaluated by passing B's causal signals through the filters (identical to running the filtered
strategy because filters act only after a base signal).

## Robustness suite (`scripts/research_suite.py`)
* 3 cost scenarios per experiment with the real ₹20k constraint, plus a research-capital run (₹10 L) to
  measure per-trade edge without the funding constraint.
* Parameter grid (buffer × volume multiplier) and exit sensitivity (reward:risk 1.5/2/3).
* Walk-forward: 12-month train / 3-month test; on each fold the combo with the best training expectancy is
  chosen and only its *next* test window counts.
* Cost sensitivity (charges × 0–2), data-quality sensitivity (randomly drop 5 %/10 % of option candles).
* Year-by-year, regime (rule + HMM), days-to-expiry, CE/PE, exit-reason breakdowns.

## Metric definitions
net P&L = Σ(gross − charges); ROI = net / capital; win rate = share of trades with net > 0; expectancy = net / trades;
profit factor = Σwins / |Σlosses|; max drawdown on daily-closed equity from starting capital; Sharpe/Sortino on
daily returns over **all** sample days (no-trade days = 0) × √252; CAGR only if sample ≥ ½ year; Calmar = CAGR / |maxDD%|.

## Limitations
Fills are simulated, not historical fills. 5-minute granularity hides intra-candle path. No partial-fill model in
backtests (no historical depth; 1 lot assumed fully filled). Paper trading *does* walk live order-book depth and
can partially fill. Expiry-day contracts are excluded by default (`avoid_expiry_day`).
