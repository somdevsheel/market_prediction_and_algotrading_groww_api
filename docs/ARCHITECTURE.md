# Architecture

```
Groww REST (growwapi 1.5.0)
  ├─ ingestion/providers/groww.py   GrowwProvider: candles (window-split, rate-limited, retried), expiries,
  │                                  contracts, instrument CSV, quotes (bid/ask depth), option chain (+Greeks), LTP
  ├─ ingestion/historical/          DataLake (bronze json / silver parquet / gold / quarantine / manifests),
  │                                  quality checks, idempotent ensure_candles()
  ├─ ingestion/instruments,macro    expiry calendar & near-month futures map; event-risk calendar (RBI, budget, CPI)
  ├─ quant/                         volatility (HV, EWMA, GARCH/GJR/EGARCH), options (BSM, IV, Greeks, surface,
  │                                  term structure), features (indicators + gold daily features), regime (rules +
  │                                  Gaussian HMM), machine_learning (walk-forward pipeline: LR/RF/XGBoost/LightGBM)
  ├─ strategies/                    A sr_breakout, B breakout_vwap, C breakout_retest; FilteredStrategy layers for
  │                                  D (volatility), E (HMM regime), H (ML), I (combined); registry from YAML
  ├─ governance/                    status lifecycle with evidence, config hashing (no silent live replacement)
  ├─ monitoring/                    alerts (ledger + webhook), Prometheus metrics, paper-vs-backtest comparison
  ├─ execution/contract_selector    1-ITM CE/PE, expiry choice, quote validation, rejection reasons
  ├─ risk/                          pre_trade.evaluate (veto power), stop sizing, kill switch (file)
  ├─ execution/                     OrderState machine, OrderManager (idempotent refs, reconciliation),
  │                                  BrokerAdapter → PaperBroker | GrowwBroker
  ├─ ledger/store.py                SQLite: signals, risk_decisions, orders, order_events, trades, trade_events,
  │                                  system_status, errors   ← single source of truth
  ├─ apps/worker/engine.py          TradingEngine.step(): same code path for paper and live
  ├─ backtesting/                   event-driven Backtester on real option candles, costs, metrics, robustness,
  │                                  walk-forward selection, parameter/cost/data-quality sensitivity
  ├─ apps/worker/scheduler.py       daily jobs: pre-market check, start worker, EOD ingest + gold rebuild + backup
  └─ apps/api + apps/dashboard      FastAPI + static dashboard (8 screens, backtest launcher, comparisons)
```

Design rules
* Strategy → direction only. Contract selector → instrument. Risk engine → approve/veto. Order manager → orders.
* Paper vs live differ **only** in the `BrokerAdapter` passed to `TradingEngine`.
* Every decision (signal, rejection reason, risk checks, order transition) is persisted.
* Restart safety: `TradingEngine.recover()` rebuilds state from the ledger and reconciles with the broker before
  any new entry; mismatches activate the kill switch.

Deviations from the spec (deliberate, for the first iteration)
* Dashboard is a single static HTML page (no React/Next.js build step).
* SQLite for metadata (PostgreSQL later), local Parquet (S3 later), REST polling (no WebSocket yet).
* No Polars/APScheduler: pandas is sufficient at current data volumes; the scheduler is a small dependency-free
  loop with once-per-day job state in the ledger.
* No Zerodha adapter (excluded by the user); interfaces are ready for it.
