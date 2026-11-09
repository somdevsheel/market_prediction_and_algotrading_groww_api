# Claude Code: NIFTY Institutional-Style Order Flow Scalping Backtest

**Project:** Order Flow Scalping Research Engine  
**Version:** 1.0  
**Mode:** Historical Backtesting Only  
**Initial Instrument:** NIFTY 50 Futures  
**Primary Objective:** Test whether order-book imbalance and microstructure signals can generate statistically significant, risk-adjusted returns after transaction costs.

## 1. Objective

Build a modular algorithmic trading strategy combining:

1. Order Flow Imbalance (OFI)
2. Microprice Prediction
3. Volume Delta
4. VWAP Momentum
5. VWAP Mean Reversion

Test each signal independently and in combination. Do not assume any strategy is profitable.

## 2. Data Requirements

Support the following historical datasets:

- OHLCV candles: 1-second, 1-minute and 5-minute.
- Bid and ask prices.
- Bid and ask quantities.
- Level 2 order-book depth, preferably 5 or more levels.
- Trade-by-trade price and quantity.
- Aggressive buy/sell trade classification, where available.
- Timestamp with millisecond or better precision.

Important: If historical Level 2 or trade-level data is unavailable, implement a candle-based proxy mode. Clearly label its results as proxy backtests, not genuine order-flow backtests.

Validate missing timestamps, duplicate records, bad prices, market holidays and data gaps.

## 3. Mathematical Models

### A. Order Flow Imbalance

Calculate best-level order flow imbalance:

\[
OFI_t =
\Delta Q^{bid}_t-\Delta Q^{ask}_t
\]

Implement the event-based Cont-Kukanov-Stoikov OFI formulation using changes in best bid/ask prices and quantities.

Normalize:

\[
NOFI_t=\frac{OFI_t}{\text{Rolling Average Depth}+\epsilon}
\]

### B. Microprice

\[
MP_t =
\frac{Ask_t \times BidQty_t+
Bid_t \times AskQty_t}
{BidQty_t+AskQty_t}
\]

Calculate microprice deviation:

\[
MPD_t=MP_t-MidPrice_t
\]

### C. Volume Delta

\[
VD_t=BuyVolume_t-SellVolume_t
\]

Calculate cumulative volume delta over configurable rolling windows.

Use actual aggressor-side trade data where available. Do not assume candle direction identifies true trade aggressors.

### D. VWAP

\[
VWAP_t=
\frac{\sum_{i=1}^{t}Price_i Volume_i}
{\sum_{i=1}^{t}Volume_i}
\]

Calculate price deviation from VWAP and its normalized value using rolling volatility.

## 4. Strategies to Implement

Build five independent strategy modules.

**Strategy 1: OFI Scalping**
- Long when normalized OFI is strongly positive.
- Short when normalized OFI is strongly negative.
- Add configurable persistence and liquidity filters.

**Strategy 2: Microprice Prediction**
- Long when microprice indicates upward pressure.
- Short when microprice indicates downward pressure.
- Test prediction horizons of 1, 5, 10 and 30 seconds.

**Strategy 3: Volume Delta Scalping**
- Identify aggressive buying and selling pressure.
- Test cumulative delta and delta acceleration.
- Avoid signals during insufficient trading activity.

**Strategy 4: VWAP Momentum**
- Trade continuation when price and volume confirm a directional move.
- Include volatility and trend filters.

**Strategy 5: VWAP Mean Reversion**
- Identify statistically significant deviations from VWAP.
- Test reversion entries and VWAP-based exits.
- Disable during strong directional market regimes.

## 5. Hybrid Strategy

After individual testing, develop a combined strategy.

Example scoring framework:

\[
Score_t =
w_1 Z(OFI_t)+
w_2 Z(MPD_t)+
w_3 Z(VD_t)+
w_4 Z(VWAPSignal_t)
\]

Weights must be configurable and calibrated only on training data.

Initial hypothesis:
- Positive score: potential long.
- Negative score: potential short.
- Weak score: no trade.

Test fixed thresholds and regime-dependent thresholds.

Do not optimize weights on the final out-of-sample dataset.

## 6. Market Regime Detection

Classify market conditions into:

- Trending bullish
- Trending bearish
- Sideways
- High volatility
- Low volatility

Use ADX, realized volatility, VWAP slope and price structure as candidate features.

Compare strategy performance across each regime.

## 7. Backtesting Engine

Build an event-driven backtesting engine.

Required functionality:

- Historical event replay.
- Signal generation.
- Order submission simulation.
- Entry and exit execution.
- Position tracking.
- Stop-loss and target management.
- Slippage simulation.
- Brokerage and exchange charges.
- Position sizing.
- Daily loss limits.
- End-of-day position square-off.

Execution assumptions:
- Never execute at a price that was not available at the decision time.
- Prevent look-ahead bias.
- Account for bid-ask spread.
- Simulate limit-order non-fills.
- Account for latency and market impact.
- Do not assume every limit order gets filled.
- Use realistic marketable-limit or market-order execution models.

For order-book strategies, include queue-position and partial-fill modelling where the data supports it.

## 8. Risk Management

Implement configurable controls:

- Maximum risk per trade.
- Maximum daily loss.
- Maximum open positions.
- Maximum consecutive losing trades.
- Volatility-adjusted stop-loss.
- Maximum holding duration.
- Daily trading shutdown.
- No new entries outside configured market hours.

Do not increase position size after losses.

## 9. Performance Metrics

Calculate:

| Metric | Purpose |
|---|---|
| Net P&L | Actual simulated profitability |
| ROI | Return on starting capital |
| Win Rate | Winning trade percentage |
| Profit Factor | Gross profit / gross loss |
| Sharpe Ratio | Risk-adjusted returns |
| Sortino Ratio | Downside risk-adjusted returns |
| Maximum Drawdown | Worst peak-to-trough loss |
| Average Holding Time | Scalping efficiency |
| Expectancy | Expected net profit per trade |
| Slippage | Execution cost |
| MAE | Maximum adverse excursion |
| MFE | Maximum favourable excursion |
| Turnover | Trading activity |

All financial metrics must include transaction costs where applicable.

## 10. Statistical Validation

Do not accept a strategy based on profitability in a single backtest.

Implement:

1. Train/test separation.
2. Walk-forward validation.
3. Out-of-sample testing.
4. Parameter sensitivity analysis.
5. Bootstrap confidence intervals.
6. Performance by market regime.
7. Performance by time of day.
8. Trade-count sufficiency checks.
9. Comparison against simple VWAP and momentum baselines.

Report parameter stability and the impact of multiple strategy/parameter experiments.

## 11. Strategy Comparison

Create a consolidated comparison report:

| Strategy | Net P&L | Profit Factor | Sharpe | Max DD | Trades |
|---|---:|---:|---:|---:|---:|
| OFI | | | | | |
| Microprice | | | | | |
| Volume Delta | | | | | |
| VWAP Momentum | | | | | |
| VWAP Mean Reversion | | | | | |
| Hybrid | | | | | |

Populate only with actual backtest results.

## 12. Project Architecture

Create the following modular structure:

```text
orderflow_scalping/
│
├── config/
│   ├── settings.yaml
│   └── strategy_params.yaml
│
├── data/
│   ├── ingestion/
│   ├── validation/
│   ├── preprocessing/
│   └── storage/
│
├── indicators/
│   ├── ofi.py
│   ├── microprice.py
│   ├── volume_delta.py
│   └── vwap.py
│
├── strategies/
│   ├── ofi_strategy.py
│   ├── microprice_strategy.py
│   ├── volume_delta_strategy.py
│   ├── vwap_momentum.py
│   ├── vwap_mean_reversion.py
│   └── hybrid_strategy.py
│
├── backtesting/
│   ├── engine.py
│   ├── execution.py
│   ├── portfolio.py
│   ├── costs.py
│   └── risk_manager.py
│
├── validation/
│   ├── walk_forward.py
│   ├── statistical_tests.py
│   └── sensitivity.py
│
├── analytics/
│   ├── performance.py
│   ├── trade_analysis.py
│   └── report_generator.py
│
├── results/
├── tests/
└── main.py
```

## 13. Required Outputs

Generate the following after each backtest:

1. Complete trade-by-trade CSV.
2. Daily P&L report.
3. Equity curve.
4. Drawdown chart.
5. Monthly returns.
6. Strategy comparison report.
7. Entry and exit timestamps.
8. Execution and transaction cost breakdown.
9. Parameter configuration snapshot.
10. Backtest reproducibility metadata.

## 14. Implementation Instructions for Claude Code

Follow this implementation sequence:

**Phase 1:** Inspect existing project and historical datasets. Reuse existing infrastructure where possible.

**Phase 2:** Implement data ingestion, validation and feature calculation.

**Phase 3:** Implement five independent strategy modules.

**Phase 4:** Build event-driven backtesting and realistic execution simulation.

**Phase 5:** Implement risk management and performance analytics.

**Phase 6:** Run individual and hybrid strategy experiments.

**Phase 7:** Perform walk-forward and out-of-sample validation.

**Phase 8:** Generate a consolidated research report.

Do not overwrite existing project files or existing backtest results. Use a separate experiment directory and preserve all configurations.

### Final Acceptance Criteria

The project is complete when:

- All five strategies run independently.
- Hybrid strategy runs successfully.
- No look-ahead bias is detected in the tests.
- Transaction costs and execution assumptions are documented.
- Results can be reproduced from saved configurations.
- Out-of-sample performance is reported separately.
- Missing order-book data is never silently replaced with invented data.

**Most important instruction:** Do not attempt to manufacture profitability through parameter optimization. The objective is to discover whether a statistically robust trading edge exists after realistic execution costs, not to produce an attractive backtest.

**Execution mode: Historical research and paper simulation only. No live orders.**