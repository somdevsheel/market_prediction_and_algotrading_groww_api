# Known limitations

## Trading / execution
* **Not verified with real orders.** The Groww execution adapter is unit-tested against mocked responses that
  follow the documented fields, but no real order has been placed. MIS acceptance for index options, SL
  (stop-limit) acceptance, and `average_fill_price` semantics must be confirmed with a tiny supervised live test.
* The broker-side stop is a stop-limit. A fast gap can jump through the limit and leave it unfilled; the
  software stop then exits with marketable limits (up to 4 tries), and failing that the kill switch fires
  with RECONCILIATION_REQUIRED, so **you must act manually**.
* REST polling only (no WebSocket): decisions use 5-minute closes; position checks run every ~3 s. Stops are
  not tick-precise.
* Paper fills cross the live bid/ask plus 1 tick. Queue position and market impact are not modelled.
* If the worker process dies, only the broker-side stop protects the position. MIS positions are squared off
  by the broker near close (policy and charges are Groww's).
* No authentication on the dashboard API. Keep it on 127.0.0.1.

## Data
* NIFTY spot has no volume. VWAP and volume filters use the near-month future.
* No historical bid/ask, so backtest spreads are assumptions (scenario table).
* The holiday list in `config/instruments.yaml` is maintained by hand. Verify it against NSE yearly.
  The worker also refuses to trade if no fresh candle appears by 09:25.
* The lot-size schedule for backtests before Dec 2025 is unverified.
* The documented candle range limits conflict between two Groww doc pages; we use the newer
  (backtesting) page. The historical rate-limit category is undocumented; ~2 req/s triggered breaches.

## Research
* ML sample size is limited by one-signal-per-day strategies (hundreds, not thousands); treat ML metrics as
  weak evidence. ML never trades unless its strategy passes governance.
* HMM labels depend on refit windows; states are re-labelled by mean return each refit.
* Event dates before FY2025-26 were entered from memory (marked VERIFY in config/events.yaml).
* Historical lot sizes for 2021-07 → 2024-11 are partly from memory (marked VERIFY).
* EWMA (1-step) and GARCH (multi-step over the refit window) evaluations use different horizons.
* Spec deviations: no Zerodha (excluded), no React/Next.js (static dashboard), SQLite instead of PostgreSQL,
  local backups (S3 optional), Python 3.10, no Polars/APScheduler.
* Charges are configured rates. Verify them against the current Groww and NSE schedules.

## Capital
* With ₹20,000 capital and one 1-ITM lot at ~₹15–19k, a modest drawdown makes the next lot unaffordable,
  which the backtest shows as `sufficient_funds` rejections. The account had ₹600 on 2026-11-09.
