# Nifty Quant System

Research-driven, risk-controlled NIFTY 50 intraday options-buying platform on the Groww Trade API.
**Default mode is PAPER. Live trading is locked** until three things are true: `settings.mode: live`, the
strategy status is `APPROVED_FOR_LIVE`, and a same-day approval exists (`scripts/approve_live.py`).

## Daily runbook (paper)

```bash
# 1. Before 09:15 IST: refresh credentials (token expires ~06:00 daily) — edit .env, see .env.example
python3 scripts/check_connection.py        # token, funds, lot size, next session, live blockers

# 2. Start the dashboard (separate terminal) — React app served by FastAPI
(cd apps/web && npm install && npm run build)                       # once, and after UI changes
python3 -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8100   # http://127.0.0.1:8100

# 3. Start the worker (waits for the session, max 1 trade, squares off 15:15, exits)
python3 scripts/run_trader.py              # paper
```

Emergency: `python3 scripts/kill.py --on "reason"` (or the dashboard button). The running worker exits any
open position and blocks entries. Clear with `--off`.

## Automation
```bash
python3 apps/worker/scheduler.py     # 08:50 checks, 09:05 paper worker, 15:45 EOD ingest + features + backup
```

## Strategy
One validated strategy is active: **`orb_retest30_cheapiv_v1`** (30-min ORB retest, bought only when option IV ≤
GARCH forecast). Rules, evidence, risks and the live runbook: `docs/STRATEGY_AND_LIVE_RUNBOOK.md`.
Strategies that failed testing (incl. order-flow scalping, S/R, EMA, ORB variants) were deleted; recoverable from git history (commit 8a0cab9 and earlier).

## Research / maintenance
```bash
python3 scripts/backtest_strategy.py --capital 20000 --risk 1000   # reproduce the strategy backtest
python3 scripts/bulk_ingest.py --start 2026-11-01 --end <today> --intervals 1minute,5minute,1day
python3 scripts/build_gold.py iv --start 2026-11-01                # ATM IV history from real option candles
python3 scripts/build_gold.py features                             # GARCH/HV/VIX/HMM/events (+ next session)
python3 scripts/promote_strategy.py --list                         # governance status
python3 scripts/live_smoke_test.py                                 # zero-cost live broker check (market hours)
python3 scripts/backup.py                                          # verified backup
python3 -m pytest                                                  # test suite
```
The dashboard can also launch backtests (Backtests tab) and compare runs.

## Layout
`core/` config, clock, logging, secrets · `ingestion/` Groww provider, data lake, quality ·
`quant/` HV/EWMA/GARCH, BSM/IV/Greeks, indicators, regime · `strategies/` signal logic ·
`execution/` contract selection, order state machine, paper & Groww brokers, order manager ·
`risk/` pre-trade checks, kill switch · `ledger/` SQLite source of truth · `backtesting/` engine, costs,
metrics, robustness · `apps/` worker engine, API, dashboard · `docs/` architecture and status.

See `docs/PROJECT_STATUS.md` for what is done, what is not, and `docs/KNOWN_LIMITATIONS.md`.

## Dashboard development
`apps/web/` is a React 18 + TypeScript + Vite app (TradingView Lightweight Charts). For live-reload development run
the API on :8100 and `cd apps/web && npm run dev` (http://localhost:5173, proxies `/api` to :8100). `npm run build`
writes `apps/web/dist/`, which FastAPI serves at `/` (the old single-file dashboard remains only as a fallback).



To see it on the dashboard: the server on port 8010 isn't running now. Start it with:


python3 -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8010
Then open http://127.0.0.1:8010/#stocks.

Updating daily: after the market closes, run:


python3 -m ml_lab.data.all_stocks update
python3 -m ml_lab.stocks_bhs --current-only
python3 -m ml_lab.stock_price_forecast