# Nifty Quant System

Research-driven, risk-controlled NIFTY 50 intraday options-buying platform on the Groww Trade API.
**Default mode is PAPER. Live trading is locked** until three things are true: `settings.mode: live`, the
strategy status is `APPROVED_FOR_LIVE`, and a same-day approval exists (`scripts/approve_live.py`).

## Daily runbook (paper)

```bash
# 1. Before 09:15 IST: refresh credentials (token expires ~06:00 daily) — edit .env, see .env.example
python3 scripts/check_connection.py        # token, funds, lot size, next session, live blockers

# 2. Start the dashboard (separate terminal)
python3 -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000   # http://127.0.0.1:8000

# 3. Start the worker (waits for the session, max 1 trade, squares off 15:15, exits)
python3 scripts/run_trader.py              # paper
```

Emergency: `python3 scripts/kill.py --on "reason"` (or the dashboard button). The running worker exits any
open position and blocks entries. Clear with `--off`.

## Automation
```bash
python3 apps/worker/scheduler.py     # 08:50 checks, 09:05 paper worker, 15:45 EOD ingest + features + backup
```

## Research
```bash
python3 scripts/bulk_ingest.py --start 2020-01-01 --end 2026-11-09 --intervals 5minute,1day
python3 scripts/build_gold.py iv --start 2021-01-01        # ATM IV history from real option candles
python3 scripts/build_gold.py features                     # HV/EWMA/GARCH/VIX/IV/HMM/events (+ next session)
python3 scripts/research_suite.py --start 2021-01-01 --end 2026-11-06   # experiments A–I, ML, walk-forward
python3 scripts/promote_strategy.py --list                 # governance status
python3 scripts/backup.py                                  # verified backup
python3 -m pytest                                          # 79 tests
```
The dashboard can also launch backtests (Backtests tab) and compare runs.

## Layout
`core/` config, clock, logging, secrets · `ingestion/` Groww provider, data lake, quality ·
`quant/` HV/EWMA/GARCH, BSM/IV/Greeks, indicators, regime · `strategies/` signal logic ·
`execution/` contract selection, order state machine, paper & Groww brokers, order manager ·
`risk/` pre-trade checks, kill switch · `ledger/` SQLite source of truth · `backtesting/` engine, costs,
metrics, robustness · `apps/` worker engine, API, dashboard · `docs/` architecture and status.

See `docs/PROJECT_STATUS.md` for what is done, what is not, and `docs/KNOWN_LIMITATIONS.md`.
