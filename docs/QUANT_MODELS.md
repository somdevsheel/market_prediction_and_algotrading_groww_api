# Quant models

| model | module | inputs → outputs | tests |
|---|---|---|---|
| HV5..HV60 | quant/volatility/hv.py | closes → annualized % (sample std × √ppy × 100); intraday uses 252×375/interval | manual calc |
| Realized / intraday RV | hv.py | returns → % | – |
| Vol percentile / expansion | hv.py | series → percentile, expansion/contraction label | – |
| EWMA | quant/volatility/ewma.py | σ²(t)=λσ²(t−1)+(1−λ)r²(t−1), λ=0.94 default; OOS MSE/QLIKE comparison of λ | recursion, no look-ahead |
| GARCH(1,1), GJR, EGARCH | quant/volatility/garch.py | Gaussian MLE (SciPy SLSQP), returns×100; h-step forecasts; rolling OOS QLIKE | parameter recovery on simulated GARCH |
| BSM price / Greeks | quant/options/bsm.py | European, ACT/365 to 15:30 expiry; theta/day, vega & rho per 1 pt | Hull value, parity, finite differences |
| IV solver | bsm.py | Brent on [1e-4, 5]; None for no-arb violations / t≤0 | round-trip, invalid inputs |
| IV surface | quant/options/surface.py | chain → model vs provider IV, ATM IV, smile, 2% skew proxy, IV rank/percentile, RV−IV | live check: model ATM IV 16.75% vs Groww 16.49% (2026-11-09) |
| Indicators | quant/features/indicators.py | EMA, RSI, MACD, ATR, ADX, session VWAP, opening range, causal swing points, volume expansion | yes |
| Regime (rules) | quant/regime/rules.py | ADX, ATR ratio, EMA20/50, VIX → label | via strategy tests |

| Gold daily features | quant/features/daily_features.py | 5-min NIFTY + VIX + ATM IV → per-session HV, EWMA/GARCH/GJR forecasts (rolling refit), VIX, IV percentile, RV/IV, HMM probabilities, event tags; every value uses data < session (plus a next-session row) | no-look-ahead test (shock last day, earlier rows unchanged) |
| ATM IV history | ingestion/historical/iv_history.py | real option candles → ATM IV at 10:00 and EOD per day (BSM solver) | via surface tests |
| Gaussian HMM | quant/regime/hmm.py | NumPy Baum-Welch, diagonal covariance, states sorted by mean return; `filtered_probs` (causal) for trading, rolling refit every 63 sessions on 750-day window | regime recovery >95 %, causality test |
| Event calendar | ingestion/macro/events.py | config/events.yaml → RBI / budget / election / CPI(next session) tags; filter only | yes |
| ML pipeline | quant/machine_learning/pipeline.py | candidate signals → decision-time features → label (target before invalidation) → expanding walk-forward (quarterly refit), no shuffling; LR, RF, XGBoost, LightGBM; AUC/Brier/top-30 % hit rate | chronology & signal-recovery test |
| IV term structure | apps/api `/api/research/term_structure` | live chains for next expiries → ATM IV per expiry | – |
| Model evaluation | scripts/build_gold.py features | rolling OOS QLIKE for GARCH/GJR/EGARCH and EWMA λ (same units) → gold/volatility/model_eval | – |

Model outputs are research inputs. ML can only influence orders through a strategy whose config includes
`min_ml_prob` *and* that has passed governance; RV−IV is a filter input, never a standalone trigger.
