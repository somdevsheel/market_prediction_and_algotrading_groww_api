# Archive: strategies that did not pass real-option backtesting

Everything here was tested on real NIFTY option prices (2020/21 – Nov 2026) with full charges and slippage and
**did not** produce a reliable profit. It is kept for reference and reproducibility; nothing in this folder is
imported by the live system, and these tests are not collected by `pytest`.

| Strategy / study | Code | Result on real options |
|---|---|---|
| A: support/resistance breakout | `strategies/breakout/sr_breakout.py` | lost |
| B: OR breakout + VWAP + volume | `strategies/breakout/breakout_vwap.py` | negative in all 12 experiment/cost combinations |
| C: breakout retest (5-min) | `strategies/breakout/breakout_retest.py` | lost |
| D/E/H/I: volatility, HMM-regime, ML, combined filters on B | `config/strategies_retired.yaml` | lost |
| ORB 60-min breakout, 30-min retest (no IV filter), 5-min reversal (video) | `config/strategies_retired.yaml`, `scripts/orb_research.py` | lost (60-min: profit only 2020-22) |
| 9/21 EMA crossover with windows, 3 modes (video prompts) | `strategies/ema_cross/`, `backtesting/ema_options.py`, `scripts/ema921_research.py` | lost, or profit only from 2020; ATM-sell needs ~Rs 1.2 L margin |
| 9/21 EMA, first cross held to close (YouTube Short) | (research script, results in `backtests/`) | −Rs 31k to −Rs 2.5 L |
| ML meta-labeling (LR/RF/XGBoost/LightGBM) on ORB retest | `scripts/research_suite.py` | AUC ≈ 0.50 (no predictive power) |

Backtest outputs: `archive/backtests/` (git-ignored). The full pre-cleanup project is the first git commit.
To restore a strategy: move its file back under `strategies/`, re-register its class in `strategies/registry.py`
and copy its entry from `config/strategies_retired.yaml` into `config/strategies.yaml`.
