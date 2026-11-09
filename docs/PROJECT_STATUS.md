# Project status — 2026-11-09 (phase 2: everything except Zerodha)

## Phase gates
| # | Phase | State | Evidence |
|---|---|---|---|
| 1 | Repo & infrastructure | ✅ | modular packages, YAML config, redacted JSON logs, Docker, 79 passing tests |
| 2 | Historical provider adapters | ✅ Groww · ⛔ Zerodha (excluded) | coverage probe 2020–2026 (1m/5m index, expired options with volume+OI); bulk ingest 0 failed windows; instrument master snapshot + checks (4,410 contracts, 0 mapping errors) |
| 3 | Data lake | ✅ | bronze/silver/gold/quarantine/manifests; idempotent; completeness report; gold features, IV history, ML, backtest results |
| 4 | Quant research | ✅ | HV/EWMA/GARCH/GJR/EGARCH + OOS evaluation, BSM/IV/Greeks (+ provider comparison), smile/skew/term structure/IV rank, RV vs IV, rule + HMM regimes, ML walk-forward (LR/RF/XGBoost/LightGBM) |
| 5 | Features & signals | ✅ | strategies A, B, C; filter layers D, E, H, I; reproducible signal ids; no-look-ahead tests |
| 6 | Contract selector | ✅ | 1-ITM, expiry-day avoidance, liquidity/spread/staleness checks, contract-level IV filter |
| 7 | Backtesting | ✅ | real option candles, 3 cost scenarios, IS/OOS, walk-forward, parameter/cost/data-quality sensitivity, Monte Carlo, bootstrap, year/regime/DTE/CE-PE breakdowns |
| 8 | Risk engine | ✅ | 12+ rejection scenarios tested; kill switch |
| 9 | Paper trading | ✅ code · ⏳ live sessions | depth-walking partial fills; paper-vs-backtest comparison; first real session Wed 2026-11-11 |
| 10 | Broker integration | ⚠ mocks only | no real order yet (needs funds + approval) |
| 11 | Dashboard | ✅ | React + TypeScript: Overview, Market, Strategy, P&L, Trades, Backtests (run + compare), Research, Health; light/dark |
| 12 | Monitoring & recovery | ✅ | alerts (ledger + webhook), Prometheus /metrics, freshness/connectivity/stuck-order checks, token auto-refresh, verified backups, scheduler, restart reconciliation |
| 13 | Live approval | 🔒 | governance requires OOS-validated + ≥20 profitable paper trades + config-hash match + same-day approval; funds ₹600 |

## Active strategy (after cleanup, 2026-11-09)
Only `orb_retest30_cheapiv_v1` is active: +₹44,810 on real options 2021–2026 (₹20k / ₹1,000 risk, conservative
costs), PF 1.23, max DD −₹22,802, lost in 2024 and 2025. Every other strategy tested (A, B, C, D/E/H/I, ORB video
setups, 9/21 EMA variants, ML meta-labeling) failed on real options and is archived in `archive/` with its results.

## Spec deviations
No Zerodha (excluded by the user) · SQLite instead of
PostgreSQL · local backups with optional S3 · Python 3.10 · no Polars/APScheduler.
