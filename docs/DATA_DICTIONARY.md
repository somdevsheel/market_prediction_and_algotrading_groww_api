# Data dictionary

## Candles (silver)
| column | type | notes |
|---|---|---|
| timestamp | datetime tz Asia/Kolkata | candle **start** time |
| open/high/low/close | float | |
| volume | float | 0/NULL for the NIFTY index (no traded volume) — use futures |
| open_interest | float | Groww returns OI for expired option contracts; NULL for current-day data in some responses |
| source | str | provider name (`groww`) |
| schema_version | int | 1 |

Paths: `data_lake/silver/<dataset>/symbol=<groww_symbol>/interval=<interval>/year=Y/month=MM/part.parquet`
Datasets: `nifty` (NSE-NIFTY, CASH), `futures` (NSE-NIFTY-<ddMonyy>-FUT), `options` (NSE-NIFTY-<ddMonyy>-<strike>-<CE|PE>).
Bronze keeps the raw response with `meta` (source, api_version, params, downloaded_at, sha256 checksum).
Manifests (`data_lake/manifests/<dataset>.jsonl`): one line per window with status `complete|partial|failed`
and the quality report. Quarantine holds rows failing OHLC / price / session checks with `quarantine_reason`.

## Symbol formats (observed from Groww, 2026-11-09)
* trading_symbol (orders/quotes): `NIFTY26N1725000CE` · groww_symbol (historical): `NSE-NIFTY-17Nov26-25000-CE`
* Index: `NSE-NIFTY` / `NIFTY`; VIX LTP key `NSE_INDIAVIX`. NIFTY lot size today: 65.

## Ledger (state/ledger.sqlite3)
* `signals` — every strategy signal (full JSON incl. reason codes and diagnostics)
* `risk_decisions` — every approval/rejection with all checks, planned and worst-case loss
* `orders` / `order_events` — projection + append-only transition log
* `trades` / `trade_events` — trade_id, signal_id, strategy_id, model_version, mode, contract, entry/exit
  time & price, expected entry/exit, qty, planned stop, target, planned_risk, realized_risk, charges,
  slippage, gross_pnl, net_pnl, exit_reason, regime, status (PENDING_ENTRY, OPEN, EXIT_PENDING, CLOSED,
  CANCELLED, REJECTED, RECONCILIATION_REQUIRED)
* `system_status` — heartbeat, market snapshot, position MTM, last decision; `errors`.

## Backtest outputs (data/backtests/<run_id>/)
`trades.csv`, `equity.csv`, `summary.json` (full config snapshot, metrics, IS/OOS, Monte Carlo, bootstrap,
skipped signals, data issues).

## Additional datasets
* silver `india_vix` (NSE-INDIAVIX, CASH) 5minute + 1day; silver `nifty` 1day.
* gold `volatility/daily_features.parquet` — one row per session + next-session row: hv5..hv60, hv_ratio,
  ret_prev, trend_prev, range_prev, ewma_fc, garch_fc, gjr_fc, vix_prev, vix_chg, atm_iv_prev, iv_pctile,
  rv_iv_ratio, hmm_p0..p2, hmm_label, hmm_state, events. All values known before that session's open.
* gold `volatility/model_eval.parquet` — OOS QLIKE per volatility model.
* gold `implied_volatility/atm_iv_daily.parquet` — date, expiry, strike, spot_1000, spot_eod, iv_ce_1000,
  iv_pe_1000, iv_atm_1000, iv_atm_close, days_to_expiry.
* gold `ml/dataset_<base>.parquet`, `ml/oos_predictions_<base>.parquet` (decision_ts, model, prob, label,
  train_end), `ml/metrics_<base>.parquet`; final model `state/models/ml_<base>_<model>.pkl` + `.json` metadata.
* gold `backtest_results/research_<ts>.parquet`; full report `data/backtests/research_<ts>/report.json`.
* `state/governance.json` — append-only status transitions with evidence and config hash.
* ledger table `alerts` (ts, level, code, message, delivered).

## Measured coverage (Groww, probe of 2026-11-09)
| year | NIFTY 1m | NIFTY 5m | expired near-ATM weekly option (5m) |
|---|---|---|---|
| 2020–2026 | 375 candles/day | 75/day | 75 candles with volume and OI, every year sampled |
Bulk ingest 2020-01-01 → 2026-11-09: NIFTY and INDIA VIX 5m and daily, 83 near-month futures contracts,
0 failed windows. Post-close flat index candles (15:30–15:55) appear on some days and are quarantined.
