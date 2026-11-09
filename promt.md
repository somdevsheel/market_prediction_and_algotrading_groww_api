# Nifty Quant Algo Trading System
## Complete Engineering Roadmap and Master Implementation Specification

**Project type:** Personal quantitative research and automated Nifty options trading platform  
**Primary instrument:** Nifty 50 index options  
**Trading style:** Intraday options buying, CE and PE  
**Implementation environment:** Python, cloud infrastructure, web dashboard  
**Development agent:** Claude Code  
**Architecture:** Modular, broker-independent, research-driven and risk-controlled

---

# 1. Project Objective

Build a complete quantitative algorithmic trading platform capable of:

1. Collecting and maintaining historical market data.
2. Building a structured financial data lake.
3. Performing quantitative research.
4. Calculating volatility, implied volatility and option Greeks.
5. Identifying market regimes.
6. Generating and evaluating trading signals.
7. Backtesting multiple strategies and model combinations.
8. Performing realistic options execution simulations.
9. Running automated paper trading.
10. Executing live trades through a broker API after explicit approval.
11. Monitoring positions, orders, risk and performance.
12. Displaying all trading and research information through a web dashboard.

This is a research and execution engineering project, not a strategy that should assume profitability. Every strategy must demonstrate robust out-of-sample performance before live deployment.

## 1.1 Initial trading constraints

Implement these as configurable risk parameters, not hardcoded assumptions.

| Parameter | Initial configuration |
|---|---:|
| Initial capital | ₹20,000 |
| Instrument | Nifty 50 options |
| Trading direction | CE and PE buying |
| Maximum position | 1 lot |
| Maximum trades | 1 per trading day |
| Trading type | Intraday only |
| Overnight positions | Not allowed |
| Risk budget | ₹500–₹1,000 per trade |
| Re-entry | Disabled |
| Averaging | Disabled |
| Default mode | Research / Paper |
| Live trading | Locked until approval |

The ₹500–₹1,000 risk budget is a planned stop-loss budget, not a guaranteed maximum loss. Include gaps, slippage, execution failure and the possibility of losing the full option premium in risk calculations.

The system must check the current exchange lot size, contract specifications, available funds and actual option premium before allowing a trade.

---

# 2. Overall System Architecture

Build the system using independent services and clearly defined interfaces.

```text
                    MARKET DATA SOURCES
                ┌────────────────────────┐
                │ Groww Historical API   │
                │ Zerodha Kite API       │
                │ Exchange Data Sources │
                │ Other Licensed Sources│
                └────────────┬───────────┘
                             │
                             ▼
                 HISTORICAL DATA INGESTION
                             │
                             ▼
                     DATA VALIDATION
                             │
                             ▼
                       DATA LAKE
                ┌────────────────────────┐
                │ Bronze: Raw Data       │
                │ Silver: Clean Data     │
                │ Gold: Research Data    │
                └────────────┬───────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
       QUANT RESEARCH                 FEATURE ENGINE
              │                             │
       ┌──────┴─────────┐                   │
       │                │                   │
       ▼                ▼                   ▼
   VOLATILITY       OPTION MODELS      MARKET REGIMES
   HV / EWMA        BSM / Greeks       HMM / Classification
   GARCH            IV / Skew
       │                │                   │
       └────────────────┴───────────────────┘
                             │
                             ▼
                      SIGNAL ENGINE
                             │
                             ▼
                       RISK ENGINE
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
                BACKTEST         PAPER TRADING
                                      │
                                      ▼
                                LIVE EXECUTION
                                      │
                                      ▼
                               BROKER ADAPTER
                                      │
                                      ▼
                               TRADE LEDGER
                                      │
                                      ▼
                               WEB DASHBOARD
```

Each component must be independently testable. Do not put the entire application into one Python file.

---

# 3. Phase One: Project Foundation and Infrastructure

## 3.1 Technology stack

Use the following preferred technologies.

| Component | Technology |
|---|---|
| Programming language | Python 3.11+ |
| Data processing | Pandas, NumPy, Polars |
| Historical storage | Parquet |
| Analytical query engine | DuckDB |
| Metadata database | PostgreSQL |
| Local development database | SQLite |
| Object storage | AWS S3 |
| API backend | FastAPI |
| Frontend | React / Next.js |
| Charts | Lightweight Charts or Plotly |
| Background jobs | APScheduler initially; scalable queue later |
| ML | Scikit-learn, XGBoost |
| Statistical models | ARCH, Statsmodels, SciPy |
| Deep learning, if justified | PyTorch |
| Monitoring | Structured logging and Prometheus-compatible metrics |
| Deployment | Docker and Docker Compose |
| Testing | Pytest |
| Version control | Git |

Avoid introducing Kubernetes, Kafka, microservices or expensive infrastructure during the initial development stage unless measured workloads justify them.

## 3.2 Repository structure

Create a clean, modular project.

```text
nifty-quant-system/
│
├── apps/
│   ├── api/
│   ├── dashboard/
│   └── worker/
│
├── config/
│   ├── settings.yaml
│   ├── risk.yaml
│   ├── strategies.yaml
│   └── instruments.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── backtests/
│
├── data_lake/
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   ├── quarantine/
│   └── manifests/
│
├── ingestion/
│   ├── providers/
│   │   ├── base.py
│   │   ├── groww.py
│   │   ├── zerodha.py
│   │   └── other_provider.py
│   ├── instruments/
│   ├── historical/
│   ├── live/
│   └── schedulers/
│
├── quant/
│   ├── volatility/
│   ├── options/
│   ├── greeks/
│   ├── regime/
│   ├── statistics/
│   ├── machine_learning/
│   └── features/
│
├── strategies/
│   ├── base.py
│   ├── breakout/
│   ├── vwap/
│   ├── volatility/
│   └── combined/
│
├── backtesting/
│   ├── engine/
│   ├── portfolio/
│   ├── execution/
│   ├── costs/
│   ├── validation/
│   └── reports/
│
├── execution/
│   ├── broker_interface.py
│   ├── groww_adapter.py
│   ├── zerodha_adapter.py
│   ├── order_manager.py
│   ├── position_manager.py
│   └── reconciliation.py
│
├── risk/
│   ├── pre_trade.py
│   ├── position_limits.py
│   ├── daily_limits.py
│   └── kill_switch.py
│
├── paper_trading/
│
├── ledger/
│
├── dashboard/
│
├── monitoring/
│
├── tests/
│
├── scripts/
│
├── migrations/
│
├── docs/
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

Claude Code must create the structure incrementally and verify each module before moving to the next phase.

---

# 4. Phase Two: Historical Market Data Infrastructure

Historical data quality is one of the most important parts of this project.

Do not start by developing complex machine-learning strategies. First establish reliable, reproducible historical datasets.

## 4.1 Required datasets

### A. Nifty index data

Collect:

- Nifty 50 historical OHLC.
- 1-minute candles, where available.
- 5-minute candles.
- 15-minute candles.
- 30-minute candles.
- Hourly candles.
- Daily candles.

### B. Nifty futures data

Collect:

- Current futures contracts.
- Historical expired futures contracts.
- OHLC.
- Volume.
- Open interest.
- Contract expiry.
- Contract identifiers.
- Trading dates and timestamps.

Use futures volume for market participation confirmation. Nifty spot index data itself does not provide traded exchange volume.

### C. Nifty options data

Collect CE and PE contracts across available strikes and expiries.

Required fields:

- Trading date.
- Timestamp.
- Underlying price.
- Strike price.
- Option type.
- Expiry date.
- Contract identifier.
- Open.
- High.
- Low.
- Close.
- Volume.
- Open interest.
- Implied volatility, if supplied.
- Bid and ask, if supplied.
- Instrument lot size.

Historical options must be stored contract by contract.

### D. Market context

Collect where legally available:

- India VIX.
- Market holidays.
- RBI policy dates.
- CPI and inflation release dates.
- Major domestic and global macroeconomic events.
- Relevant index constituent and corporate-action metadata.

Use macroeconomic information as an event-risk filter, not as an automatic directional trading signal.

## 4.2 Data provider abstraction

Implement a common interface:

```python
class HistoricalDataProvider:

    def get_historical_candles(self, instrument, start, end, interval):
        raise NotImplementedError

    def get_instruments(self):
        raise NotImplementedError

    def get_option_contracts(self, underlying, expiry=None):
        raise NotImplementedError

    def get_option_chain(self, underlying, expiry):
        raise NotImplementedError
```

Implement independent Groww and Zerodha adapters only for endpoints they actually support.

Important requirements:

- Read the current official API documentation before implementing endpoints.
- Never invent endpoints or request parameters.
- Respect provider rate limits.
- Implement pagination and date-window splitting.
- Record missing periods and API failures.
- Do not silently substitute another instrument's data.
- Record provider provenance for every dataset.
- Keep historical ingestion separate from live execution.

### Historical coverage validation

Before committing to a provider, download sample data for:

- 2020.
- 2021.
- 2022.
- 2023.
- 2024.
- 2025.
- 2026, up to the latest available date.

Test expired option contracts, not just current contracts.

Verify whether actual 1-minute and 5-minute historical candles, volume and OI are available for each expired contract.

Do not claim that an entire year of data is available merely because an API accepts a date range.

## 4.3 Data ingestion pipeline

Implement:

1. Instrument master download.
2. Instrument and contract identification.
3. Historical date-range discovery.
4. Incremental download.
5. Rate-limit management.
6. Retry with exponential backoff.
7. Data validation.
8. Duplicate removal.
9. Missing-candle detection.
10. Parquet conversion.
11. Manifest creation.
12. Data completeness report.

Every ingestion job must be idempotent. Running it twice must not create duplicate records.

---

# 5. Phase Three: Data Lake

Implement a Bronze, Silver and Gold architecture.

## 5.1 Bronze layer

Store the original downloaded data with minimal modification.

```text
bronze/
    market/
        nifty/
        futures/
        options/
    instruments/
    macro/
    india_vix/
    corporate_actions/
```

Retain:

- Original provider response where permitted.
- Download timestamp.
- Source name.
- Instrument identifier.
- Date range.
- Request parameters.
- API version.
- Data checksum.

## 5.2 Silver layer

Store normalized and validated data.

```text
silver/
    market/
        nifty/
        futures/
        options/
    volatility/
    option_chain/
    instruments/
```

Standardize:

- Timestamp and timezone.
- Instrument identifiers.
- Expiry dates.
- Strike prices.
- Option types.
- Numeric data types.
- Missing values.
- Duplicate records.

Do not fabricate missing candles or volume.

## 5.3 Gold layer

Store research-ready datasets.

```text
gold/
    features/
    returns/
    volatility/
    technical_indicators/
    options_greeks/
    implied_volatility/
    market_regimes/
    strategy_signals/
    backtest_results/
```

Use partitioning by:

- Dataset.
- Trading year.
- Month.
- Instrument.
- Expiry where appropriate.

Use Parquet with sensible compression and schema versioning.

Use DuckDB for local analytical queries and S3 for durable object storage.

## 5.4 Data quality

Create automated reports covering:

- Missing dates.
- Missing intervals.
- Duplicate records.
- Invalid OHLC relationships.
- Zero or negative prices.
- Suspicious volume.
- Invalid expiry dates.
- Inconsistent lot sizes.
- Contract mapping errors.
- Timezone inconsistencies.

Quarantine invalid records rather than silently deleting them.

---

# 6. Phase Four: Quantitative Research Engine

Build reusable quantitative libraries. Each model must have its own inputs, outputs, tests and documentation.

## 6.1 Historical volatility

Calculate:

HV5, HV10, HV20, HV30 and HV60.

For log returns:

r(t) = ln(S(t) / S(t-1))

Annualized historical volatility:

HV = standard_deviation(r) × sqrt(252) × 100

For intraday analysis, calculate volatility using an appropriate number of trading periods rather than blindly applying the daily annualization formula.

Include:

- Rolling volatility.
- Realized volatility.
- Intraday volatility.
- Volatility percentile.
- Volatility expansion and contraction.

## 6.2 EWMA

Implement:

σ²(t) = λσ²(t-1) + (1-λ)r²(t-1)

Start with λ = 0.94 as a configurable benchmark.

Compare multiple decay factors using out-of-sample forecasting errors.

## 6.3 GARCH family

Implement:

- GARCH(1,1).
- EGARCH.
- GJR-GARCH.

Measure:

- Forecast error.
- Volatility clustering.
- Asymmetric volatility response.
- Model stability.
- Out-of-sample performance.

Do not assume the more complicated model is automatically superior.

## 6.4 Implied volatility engine

Implement:

- Black-Scholes-Merton pricing.
- Implied volatility solver.
- IV by strike.
- IV by expiry.
- ATM IV.
- IV skew.
- IV smile.
- IV term structure.
- IV percentile and rank.

Use appropriate Indian index option conventions and market inputs.

Handle zero bids, stale quotes, illiquid contracts and invalid prices.

## 6.5 Option Greeks engine

Calculate:

- Delta.
- Gamma.
- Theta.
- Vega.
- Rho.

Display Greeks for CE and PE contracts.

Compare model-derived Greeks with provider-supplied Greeks when available.

Do not use a theoretical option price as an executable market price.

## 6.6 Forecast realized volatility versus IV

Compare forecast realized volatility with implied volatility over the same remaining trading horizon.

Calculate:

- Forecast RV.
- Implied volatility.
- RV-IV difference.
- Volatility ratio.
- Forecast confidence.
- Forecast error.

Treat a positive RV-IV difference as a research signal, not a standalone trade recommendation.

## 6.7 Market regime detection

Implement a baseline rule-based regime classifier first.

Possible regimes:

- Trending bullish.
- Trending bearish.
- Range-bound.
- High volatility.
- Low volatility.
- Volatility expansion.
- Unstable / event-driven.

Then implement Hidden Markov Models.

HMM inputs can include:

- Returns.
- Realized volatility.
- ATR.
- ADX.
- VWAP deviation.
- Futures volume.
- India VIX.

Evaluate regime stability and out-of-sample usefulness.

## 6.8 Machine learning research

Implement a research pipeline for:

- Logistic Regression.
- Random Forest.
- XGBoost.
- LightGBM, if appropriate.

Possible predictions:

- Probability of breakout continuation.
- Probability of breakout failure.
- Probability of reaching a target before stop.
- Expected return over a defined horizon.
- Market regime classification.

Prevent:

- Look-ahead bias.
- Target leakage.
- Survivorship bias.
- Random shuffling of time-series data.
- Training on information unavailable at the decision timestamp.

Use chronological train, validation and test splits.

Use walk-forward validation.

Machine learning must remain optional. It must not control live orders until independently validated.

---

# 7. Phase Five: Technical Analysis and Signal Engine

Create an independent feature engine.

Required indicators:

- VWAP.
- EMA 9.
- EMA 20.
- EMA 50.
- EMA 200.
- RSI.
- MACD.
- ADX.
- ATR.
- Opening range.
- Support and resistance.
- Swing highs and lows.
- Volume expansion.
- Price momentum.

Use:

- 15-minute candles for market context.
- 5-minute candles for entry decisions.
- 1-minute candles only where the execution strategy genuinely requires them.

## 7.1 Initial strategy families

Develop each strategy independently.

### Strategy A: Support/resistance breakout

- Identify significant support and resistance.
- Detect confirmed breakout.
- Avoid signals from incomplete candles.
- Check breakout strength.
- Define invalidation level.

### Strategy B: Breakout + VWAP

- Bullish breakout above resistance.
- Price above VWAP.
- Positive market structure.
- Futures volume confirmation.

Reverse the logic for bearish trades.

### Strategy C: Breakout + retest

- Detect breakout.
- Wait for retest.
- Check whether the broken level holds.
- Require confirmation before entry.

### Strategy D: Volatility-filtered breakout

Combine directional breakout confirmation with:

- HV.
- EWMA forecast.
- GARCH forecast.
- ATM IV.
- IV skew.
- India VIX.

### Strategy E: Regime-filtered breakout

Allow strategy execution only in market regimes where its historical performance supports doing so.

## 7.2 Strategy signal format

Every strategy must return a structured signal:

```python
{
    "strategy_id": "breakout_vwap_v1",
    "timestamp": "...",
    "underlying": "NIFTY",
    "direction": "BULLISH",
    "signal_strength": 0.0,
    "entry_condition": "...",
    "invalidation_level": 0.0,
    "target_level": 0.0,
    "regime": "...",
    "model_version": "...",
    "reason_codes": []
}
```

Signals must be explainable and reproducible.

---

# 8. Phase Six: Options Contract Selection

The signal engine identifies direction using the underlying. A separate contract-selection engine chooses the option.

For bullish signals:

- Select a suitable CE contract.
- Prefer one strike ITM initially.

For bearish signals:

- Select a suitable PE contract.
- Prefer one strike ITM initially.

Validate:

- Expiry.
- Strike.
- Option type.
- Lot size.
- Available funds.
- Bid and ask.
- Spread.
- Volume.
- Open interest.
- Quote freshness.
- Contract validity.

Never select a contract only because it has the highest volume or open interest.

The contract-selection engine must return a rejection reason when no suitable contract exists.

---

# 9. Phase Seven: Backtesting Engine

This is a core subsystem. Do not implement a simplistic loop that buys an option at its closing price and sells it at a later closing price.

Build a realistic event-driven backtesting engine.

## 9.1 Required capabilities

- Historical candle replay.
- Historical option contract selection.
- Strategy execution.
- Portfolio accounting.
- Position tracking.
- Entry and exit simulation.
- Stop-loss and target handling.
- Expiry handling.
- Transaction charges.
- Bid/ask spread assumptions.
- Slippage.
- Partial fills where data permits.
- Market gaps.
- Trading-session restrictions.
- Daily trade limits.
- Capital constraints.

## 9.2 Execution assumptions

Run multiple scenarios:

1. Optimistic execution.
2. Base-case execution.
3. Conservative execution.

If historical bid/ask data is unavailable, explicitly model spread and slippage assumptions and report the limitations.

Do not describe simulated fills as actual historical fills.

## 9.3 Backtesting metrics

Calculate:

- Net P&L.
- Gross P&L.
- ROI.
- CAGR, where meaningful.
- Win rate.
- Loss rate.
- Average win.
- Average loss.
- Expectancy.
- Profit factor.
- Maximum drawdown.
- Sharpe ratio.
- Sortino ratio.
- Calmar ratio.
- Consecutive wins and losses.
- Average holding time.
- Daily P&L.
- Monthly P&L.
- Annual P&L.
- CE performance.
- PE performance.
- Strategy-wise performance.
- Regime-wise performance.
- Expiry-day performance.
- Transaction cost impact.

Use clearly documented definitions for every metric.

## 9.4 Strategy comparison

Create a framework to compare:

| Experiment | Configuration |
|---|---|
| A | Breakout |
| B | Breakout + VWAP |
| C | Breakout + VWAP + HV |
| D | Breakout + VWAP + EWMA |
| E | Breakout + VWAP + GARCH |
| F | Breakout + VWAP + IV filter |
| G | Breakout + VWAP + regime filter |
| H | Breakout + VWAP + ML confirmation |
| I | Combined model |

The experiment engine must allow parameters to be changed without rewriting strategy code.

## 9.5 Robustness testing

Implement:

- Walk-forward analysis.
- Out-of-sample testing.
- Parameter sensitivity.
- Bootstrap analysis.
- Monte Carlo trade-sequence analysis.
- Regime segmentation.
- Year-by-year performance.
- Expiry-wise performance.
- Cost sensitivity.
- Slippage sensitivity.
- Data-quality sensitivity.

Do not select the best strategy solely by historical net profit.

Store experiment configuration, source-data versions, model versions and results so every experiment can be reproduced.

---

# 10. Phase Eight: Risk Management Engine

The risk engine must be independent of strategy logic.

It must have the authority to reject any proposed trade.

## 10.1 Pre-trade checks

Before placing an order:

- Confirm trading mode.
- Confirm market session.
- Confirm current market data.
- Confirm instrument validity.
- Confirm quote freshness.
- Confirm sufficient funds.
- Confirm maximum one lot.
- Confirm daily trade count.
- Confirm no existing position.
- Calculate planned loss.
- Include charges and slippage estimates.
- Validate stop and exit conditions.

## 10.2 Daily restrictions

Initial limits:

- Maximum one completed trade per day.
- Maximum one lot.
- No averaging.
- No re-entry.
- No overnight position.
- No duplicate orders.
- No trading after daily loss limit.

## 10.3 Kill switch

Implement:

- Manual kill switch.
- Automated kill switch.
- Cancel pending orders.
- Block new entries.
- Reconcile open positions.
- Emergency exit workflow.
- Broker connectivity failure handling.

Do not assume a failed API response means an order was not placed. Reconcile order status before retrying to avoid duplicate positions.

---

# 11. Phase Nine: Paper Trading

Paper trading must use the same:

- Market data interface.
- Strategy engine.
- Contract selection.
- Risk checks.
- Order state machine.
- Position accounting.
- Trade ledger.
- Dashboard.

Only the execution adapter should change between paper and live modes.

Run paper trading for a meaningful period and across multiple market conditions.

Record:

- Signal timestamp.
- Expected entry.
- Simulated fill.
- Expected stop.
- Actual simulated exit.
- Slippage.
- Strategy version.
- Market regime.
- Reason for entry.
- Reason for exit.

The system must compare paper execution with historical backtesting assumptions.

---

# 12. Phase Ten: Live Execution

Live trading must remain disabled by default.

## 12.1 Broker architecture

Create separate adapters:

- Groww historical data adapter.
- Groww execution adapter, if supported and validated.
- Zerodha market-data adapter.
- Zerodha execution adapter.

Do not couple strategy logic to a particular broker.

Check current official documentation for authentication, rate limits, WebSocket behavior, order types, product types, contract identifiers and API restrictions.

## 12.2 Order lifecycle

Implement explicit order states:

```text
CREATED
   |
VALIDATED
   |
SUBMITTED
   |
ACKNOWLEDGED
   |
OPEN
   |
PARTIALLY_FILLED
   |
FILLED
   |
EXIT_PENDING
   |
CLOSED
```

Also support:

- REJECTED.
- CANCELLED.
- EXPIRED.
- UNKNOWN.
- RECONCILIATION_REQUIRED.

Use idempotency and reconciliation to prevent duplicate orders.

## 12.3 Live trading approval

Before enabling live execution, require:

1. Backtest validation.
2. Paper-trading validation.
3. Risk-engine tests.
4. Broker integration tests.
5. Static IP configuration, where required.
6. Current regulatory and exchange compliance review.
7. Manual live-mode approval.

Do not bypass broker or exchange requirements for automated trading.

---

# 13. Phase Eleven: Trade Ledger and Accounting

Maintain an immutable trade-event history.

Store:

- Trade ID.
- Signal ID.
- Strategy ID.
- Instrument.
- Contract.
- Entry time.
- Exit time.
- Entry price.
- Exit price.
- Quantity.
- Charges.
- Slippage.
- Gross P&L.
- Net P&L.
- Planned risk.
- Realized risk.
- Exit reason.
- Model version.
- Execution mode.

The ledger must distinguish:

- Open positions.
- Closed trades.
- Cancelled orders.
- Rejected orders.
- Failed execution attempts.

Never calculate dashboard P&L only from temporary frontend state.

Use the ledger and broker reconciliation as the source of truth.

---

# 14. Phase Twelve: Web Dashboard

Create a clean, professional quantitative trading dashboard.

Do not overload the home screen with unnecessary indicators.

## 14.1 Main dashboard

Display:

- Total capital.
- Available balance.
- Current P&L.
- Today's P&L.
- Total realized P&L.
- Total trades.
- Win rate.
- Profit factor.
- ROI.
- Maximum drawdown.
- Current trading mode.
- System health.

## 14.2 P&L analytics

Provide:

- Daily P&L.
- Weekly P&L.
- Monthly P&L.
- Yearly P&L.
- Cumulative equity curve.
- Drawdown curve.
- CE versus PE comparison.
- Strategy-wise P&L.

## 14.3 Live market screen

Display:

- Nifty spot.
- Nifty futures.
- India VIX.
- VWAP.
- EMA values.
- ATR.
- Market regime.
- Breakout levels.
- Current signal.
- Option contract.
- Bid/ask.
- IV.
- Delta, Gamma, Theta and Vega.

## 14.4 Quant research screen

Display:

- HV5, HV10, HV20, HV30, HV60.
- EWMA forecast.
- GARCH forecast.
- Realized versus implied volatility.
- IV smile.
- IV skew.
- Term structure.
- HMM regime.
- Model forecasts.
- Model evaluation statistics.

## 14.5 Backtesting screen

Allow the user to:

- Select a strategy.
- Select date range.
- Select timeframe.
- Select contract-selection rules.
- Configure risk parameters.
- Configure transaction costs.
- Run backtests.
- Compare experiments.
- View trades.
- Export results.
- Compare in-sample and out-of-sample performance.

## 14.6 Trade history screen

Show:

- Trade book.
- Order history.
- Entry and exit.
- CE/PE.
- Strategy.
- Net P&L.
- ROI.
- Charges.
- Execution status.

## 14.7 System health screen

Show:

- Data provider status.
- Last market-data timestamp.
- Historical ingestion status.
- WebSocket status.
- Broker authentication status.
- Order reconciliation status.
- Database status.
- Background worker status.
- Latest errors.
- Kill-switch status.

---

# 15. Phase Thirteen: Monitoring and Reliability

Implement:

- Structured logs.
- Error tracking.
- API latency metrics.
- Data freshness checks.
- Job failure alerts.
- Broker connection monitoring.
- Order status monitoring.
- Position reconciliation.
- Database backups.
- S3 backup verification.
- Secret management.
- Authentication refresh handling.

Never log API secrets, passwords, access tokens or other credentials.

The system must recover safely after restarts. It must reconstruct open positions and order states from persistent records and broker reconciliation before allowing new trades.

---

# 16. Phase Fourteen: Research Governance

Every strategy and model must have:

- Unique ID.
- Version.
- Description.
- Input dataset.
- Training period, where applicable.
- Validation period.
- Out-of-sample period.
- Parameter configuration.
- Performance report.
- Known limitations.
- Approval status.

Possible statuses:

```text
RESEARCH
BACKTESTED
OUT_OF_SAMPLE_VALIDATED
PAPER_TRADING
APPROVED_FOR_LIVE
RETIRED
```

Never silently replace a live strategy with a new model version.

Store model and strategy configurations so historical decisions can be reconstructed.

---

# 17. Development Roadmap and Milestones

Claude Code must implement the project in the following order.

| Phase | Deliverable | Completion gate |
|---|---|---|
| 1 | Repository and infrastructure | Application starts and tests pass |
| 2 | Historical provider adapters | Sample data downloaded and validated |
| 3 | Data lake | Bronze/Silver/Gold datasets created |
| 4 | Quant research library | Model unit tests pass |
| 5 | Feature and signal engine | Reproducible signals generated |
| 6 | Options contract selector | Correct contracts validated |
| 7 | Backtesting engine | Historical option trades simulated |
| 8 | Risk engine | All risk scenarios tested |
| 9 | Paper trading | End-to-end paper execution works |
| 10 | Broker integration | Orders and positions reconciled |
| 11 | Dashboard | All core screens functional |
| 12 | Monitoring and recovery | Restart and failure tests pass |
| 13 | Live approval | Explicit user authorization |

Do not skip a completion gate to move faster.

---

# 18. Mandatory Testing Requirements

Create automated tests for:

- Volatility calculations.
- EWMA.
- GARCH forecasting.
- Black-Scholes pricing.
- IV solver.
- Greeks.
- Indicator calculations.
- Contract selection.
- Expiry handling.
- Signal generation.
- Stop-loss calculations.
- Position sizing.
- Transaction costs.
- Backtest accounting.
- Order lifecycle.
- Duplicate-order prevention.
- Broker reconciliation.
- Kill switch.
- Database recovery.

Also create integration tests using recorded or synthetic market data.

Synthetic data must never be presented as real historical performance.

---

# 19. Acceptance Criteria

The project is not complete until all these conditions are met.

### Historical data

- Historical Nifty data is stored.
- Expired option contracts are discoverable where provider coverage permits.
- Data completeness is measured.
- Data provenance is recorded.
- Ingestion can resume after failure.

### Quant research

- HV, EWMA and GARCH work.
- IV and Greeks are calculated.
- HMM regime classification works.
- ML experiments use chronological validation.
- Model performance can be compared.

### Backtesting

- Actual historical option-contract prices are used.
- Trading costs are included.
- Stops and targets are simulated realistically.
- No look-ahead bias is introduced.
- Out-of-sample results are reported.
- Every experiment is reproducible.

### Execution

- Paper trading works end to end.
- Order states are persisted.
- Duplicate orders are prevented.
- Positions are reconciled.
- Live execution is disabled until approved.

### Dashboard

- P&L and ROI are calculated from the trade ledger.
- Backtests can be compared.
- Quant models are visible.
- System health is visible.
- Data can be exported.

---

# 20. Instructions to Claude Code

You are acting as the lead quantitative developer, software architect, data engineer and trading-systems engineer.

Follow these rules:

1. Do not build a superficial prototype and call it a complete algorithmic trading system.
2. Do not put the entire application in a single Python file.
3. Do not use hardcoded historical market prices.
4. Do not invent API endpoints.
5. Do not assume historical expired options data exists without testing it.
6. Do not use underlying-only backtests as proof of options strategy profitability.
7. Do not introduce look-ahead bias.
8. Do not ignore transaction costs, slippage or liquidity.
9. Do not allow the strategy engine to bypass risk controls.
10. Do not enable live orders automatically.
11. Do not create fake P&L or fake performance statistics.
12. Do not report a model as validated just because it runs successfully.
13. Do not use a complicated model without demonstrating its incremental value.
14. Document every module and public interface.
15. Write automated tests for each subsystem.
16. Keep broker adapters independent.
17. Preserve raw data and maintain dataset versioning.
18. Ask for explicit approval before irreversible operations or live trading.

## Required development workflow

For every phase:

1. Inspect the existing repository.
2. Read the official documentation for relevant external APIs.
3. Write an implementation plan.
4. Identify dependencies and interfaces.
5. Implement the module.
6. Run unit tests.
7. Run integration tests.
8. Check edge cases and failure conditions.
9. Update documentation.
10. Record what is complete and what remains incomplete.

Maintain these files:

- `docs/ARCHITECTURE.md`
- `docs/DATA_DICTIONARY.md`
- `docs/API_INTEGRATIONS.md`
- `docs/QUANT_MODELS.md`
- `docs/BACKTESTING_METHODOLOGY.md`
- `docs/RISK_MANAGEMENT.md`
- `docs/DEPLOYMENT.md`
- `docs/PROJECT_STATUS.md`
- `docs/KNOWN_LIMITATIONS.md`

## Final instruction

Begin with Phase One only.

First inspect the existing project directory and identify existing code that can be reused. Do not delete or overwrite existing work without approval.

Then prepare the architecture, repository structure, dependency plan and implementation checklist.

After completing Phase One, report:

- What was created.
- What was tested.
- What passed.
- What failed.
- What remains incomplete.
- What is required for Phase Two.

Proceed phase by phase. Do not claim the complete system is ready until every acceptance criterion has been demonstrated.

**The final objective is a reliable, reproducible quantitative research and trading platform, not simply an application that generates buy and sell signals.**