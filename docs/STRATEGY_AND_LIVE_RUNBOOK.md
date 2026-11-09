# ORB Retest + Cheap-IV Strategy: Strategy Guide and Live Trading Runbook

**Strategy ID:** `orb_retest30_cheapiv_v1` (version 1.0.0) · **Instrument:** NIFTY weekly options (buying only)
**Governance status:** BACKTESTED (not yet validated by paper or live trading) · **Document date:** 2026-11-09

> Read section 5 (Risks) before trading real money. A profitable backtest is not a promise of future profit.

---

## 1. Strategy rules

### 1.1 Opening range
- The **opening range** is the high (H) and low (L) of NIFTY from **09:15 to 09:45** (30 minutes, 1-minute candles).
- **Range filter:** skip the day if `(H − L) / day's open > 0.5 %`.

### 1.2 Entry: breakout, then retest, then re-break (1-minute closes)
1. **First breakout:** a 1-minute candle closes above H (bullish) or below L (bearish).
2. **Retest:** a later candle closes **back inside** the range.
3. **Entry signal:** a later candle closes beyond the **same** side again.
   - If price closes beyond the opposite side first, the setup is cancelled for the day.
4. The entry must come **within 60 minutes** of the first breakout.
5. Maximum **one trade per day**. No re-entry, no averaging.

### 1.3 Contract
- Bullish → buy a **CE**. Bearish → buy a **PE**.
- Strike: **one strike in the money**. Expiry: the nearest weekly expiry that is **not today**.
- Liquidity checks before entry:
  - spread ≤ 1.5 %
  - volume ≥ 50,000 and OI ≥ 5,000
  - premium between ₹20 and ₹600
  - quote ≤ 20 seconds old

### 1.4 Cheap-IV filter (the part that makes it work)
- Enter only if **option IV ≤ GARCH forecast volatility** (ratio ≤ 1.0).
- The IV comes from Groww's option chain. The GARCH forecast is computed each evening from data up to that day's close (`data_lake/gold/volatility/daily_features.parquet`).
- **Why it matters:** in backtests, the same setup lost heavily when the option was expensive (IV above the forecast). Buying cheap options means the expected move is not already priced in.

### 1.5 Exits (whichever comes first)
| Exit | Rule |
|---|---|
| Underlying stop | NIFTY **touches** the opposite side of the opening range (L for longs, H for shorts) |
| Premium stop | option falls to the level where the loss reaches the ₹1,000 risk budget (incl. charges and slippage) |
| Time exit | **15:15** square-off |
| Safety exits | kill switch, broker-side protective stop-limit order, daily loss limit |

There is **no fixed profit target**. Winners run until the stop or 15:15.

### 1.6 Position size and risk
- **1 lot** (65 units, from the instrument master).
- Planned risk per trade: **≤ ₹1,000** (premium stop + round-trip charges + slippage).
- Daily loss limit: **₹1,500**. Capital allocated: **₹20,000** (`config/risk.yaml`).

---

## 2. Backtest evidence

### 2.1 Setup
| Item | Value |
|---|---|
| Data | real Groww 1-minute NIFTY and **real 1-minute option candles** of the exact contract chosen each day |
| Period | 2021-01-01 to 2026-11-06 |
| Fills | next-minute open ± half-spread + slippage ticks (simulated, not actual historical fills) |
| Charges | brokerage ₹20/order, STT, exchange, SEBI, stamp, IPFT, GST |
| Reproduce | `python3 scripts/backtest_strategy.py --capital 20000 --risk 1000` (results in `data/backtests/`) |

### 2.2 Results (₹20,000 account, ₹1,000 risk cap, 15:15 exit)
| Metric | Base costs | Conservative costs |
|---|---|---|
| Trades | 277 (~47 / year) | 277 |
| Net P&L | **+₹56,277** | **+₹44,810** |
| Profit factor | 1.30 | 1.23 |
| Win rate | 29.6 % | 29.2 % |
| Max drawdown | −₹21,080 | −₹22,802 |
| Last 30 % of the period | +₹7,380 | +₹3,387 |
| Bootstrap P(average trade > 0) | 0.93 | 0.88 |

### 2.3 Year by year (conservative costs)
| 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (to 6 Nov) |
|---|---|---|---|---|---|
| +₹6,696 | +₹20,291 | +₹21,282 | **−₹5,806** | **−₹9,999** | +₹12,346 |

### 2.4 Trade profile (₹1,000-cap version)
- Average loss ≈ ₹990. Average win ≈ ₹3,100. Worst single trade ≈ −₹1,115.
- **Max consecutive losses: 11.**
- Lowest account value from the ₹20,000 start: ₹13,681 (March 2021). The largest drawdown happened later, from a higher peak.

### 2.5 Same setup without the cheap-IV filter (for comparison)
| Version | Net P&L (base / conservative) |
|---|---|
| 30-min retest, all trades | −₹37,278 / −₹74,773 |
| Trades with IV above forecast only | −₹1,20,674 |

---

### 2.6 Recent regime warning (added 2026-11-10)
| Period | Lot size | Avg stop (option pts) | Stopped out | Win rate | Net (conservative) |
|---|---|---|---|---|---|
| 2021-07 → 2024-03 | 50 | 18.7 | 56 % | 41 % | +₹45,012 |
| 2024-11 → 2025-12 | 75 | 12.4 | 75 % | 22 % | −₹12,740 |
| 2026 | 65 | 14.3 | 75 % | 20 % | +₹15,434 (one +₹15,972 trade) |

**2024–2026 total at ₹1,000 risk: −₹3,459** (−₹19,432 without its best trade). The ₹1,000 cap is too tight for
current lot sizes. With a ₹3,000 stop, 2024–2026 was +₹12,075, which needs about ₹60k capital to keep risk near 5%
per trade. **Not recommended for live trading at ₹20k / ₹1,000.**

## 3. How we got here (research log)

| Strategy tested | Real-option result 2020/21–2026 | Verdict |
|---|---|---|
| B: OR breakout + VWAP + volume | negative in all 12 experiment/cost combinations (6-month test) | ❌ |
| Volatility filters (HV, GARCH, VIX) on B | still negative | ❌ |
| 60-min ORB breakout (video setup) | +₹36k base / −₹9k conservative; all profit in 2020–22, lost every year 2023–26 | ❌ |
| 30-min ORB retest (video setup) | −₹37k / −₹75k | ❌ |
| 5-min ORB reversal (video setup) | −₹1.15 L / −₹1.44 L | ❌ |
| **30-min ORB retest + cheap-IV filter** | **+₹56k / +₹45k** | ⚠ promising, unproven live |

- **Video replication.** The ORB video's spot-point results replicated closely for breakouts (60-min: 7,693 vs 8,346 points; PF 1.14 vs 1.16). Retest and reversal replicated less closely.
- **Out-of-sample check of the video's filters.** Choosing filters on 2020–2023 and testing on 2024–2026 showed:
  - the range-size filter for the 60-min breakout does not generalise
  - the retest timing filter does

Reports of the rejected studies: `archive/backtests/` (see `archive/README.md`).

---

## 4. Live trading runbook

### 4.1 One-time prerequisites
- [ ] **Groww F&O balance ≥ ₹25,000.** On 2026-11-09 it was **₹600**; every live order would be rejected.
- [ ] Credentials in `.env`. The access token expires daily at ~06:00 IST. Prefer TOTP (`GROWW_API_KEY` + `GROWW_TOTP_SECRET`) so it renews automatically.
- [ ] Optional alerts: set `ALERT_WEBHOOK_URL` (Slack, Discord or Teams incoming webhook) in `.env`.
- [ ] Optional: `python3 apps/worker/scheduler.py` to automate the evening data refresh, features and backup.

### 4.2 First live day only: zero-cost broker test (after 09:20)
```bash
python3 scripts/live_smoke_test.py      # type: PLACE TEST ORDER
```
It places a real limit BUY far below the market (it should not fill), checks the status, cancels it and confirms the cancellation.

- **It must print `PASS`.** If it shows REJECTED or anything else, **do not go live**. Inspect the order in the Groww app and share `state/live_smoke_test.json`.
- If the order fills unexpectedly, the script offers to sell it back immediately.

### 4.3 Enable live trading (each step is recorded and is your decision)
```bash
# 1. owner override of the paper-trading requirement (type: OVERRIDE orb_retest30_cheapiv_v1)
python3 scripts/promote_strategy.py --strategy orb_retest30_cheapiv_v1 --to APPROVED_FOR_LIVE \
    --force-reason "owner chose live without paper validation"
# 2. config/settings.yaml:   mode: paper   ->   mode: live
```
- Approval is tied to the strategy's **config hash**. Changing any parameter or filter voids it automatically.

### 4.4 Every trading day
| Time (IST) | Action |
|---|---|
| before 09:00 | Refresh the token (unless TOTP). Run `python3 scripts/check_connection.py`: balance ≥ ₹20k, no `[BLOCK]`, correct next session |
| 09:00 | `python3 scripts/approve_live.py`: 7 gate questions + typed phrase; **valid for today only** |
| 09:05 | Start the dashboard `python3 -m uvicorn apps.api.main:app --port 8100` and the bot `python3 scripts/run_trader.py --mode live` |
| 09:15–15:15 | Watch the dashboard (Overview, Market, Health) and keep the Groww app open on your phone |
| 15:15 | The bot squares off any open position and exits |
| after 15:30 | Run `python3 scripts/build_gold.py features` to refresh tomorrow's GARCH forecast (the scheduler does this at 15:45) |

### 4.5 Emergency procedures
| Situation | Action |
|---|---|
| Something looks wrong | `python3 scripts/kill.py --on "reason"` or the dashboard **Activate kill switch** button. The bot exits any position and blocks entries |
| Bot frozen or crashed with a position | **Close the position manually in the Groww app.** Then check `state/ledger.sqlite3` and Health on the dashboard before restarting |
| Alert `TRADING_BLOCKED` / `RECONCILIATION_REQUIRED` | The bot found a mismatch between its records and Groww. Check positions and orders in the Groww app, resolve manually, then `python3 scripts/kill.py --off` |
| Token expired mid-day | Paste a new token into `.env` and restart the bot. It recovers open positions from the ledger and Groww before trading |
| Network outage with an open position | The broker-side stop-limit order protects the position. Verify it in the Groww app |

---

### 4.6 Fully automated live trading (no daily commands)
One-time setup:
1. **TOTP auto-login.** On Groww → Cloud API Keys, create a **TOTP** key. Put both values in `.env`:
   `GROWW_API_KEY=<TOTP token>` and `GROWW_TOTP_SECRET=<base32 secret from the QR>`. The system then logs itself in
   every morning, and `check_connection.py` stops warning about daily tokens.
2. **Funds** ≥ `risk.min_live_funds` (₹15,000 default; raise it if you widen the stop, see §2.6).
3. **Smoke test** passed once during market hours: `python3 scripts/live_smoke_test.py`.
4. **Governance + mode:** `promote_strategy.py … --to APPROVED_FOR_LIVE` (after final risk settings), and set
   `mode: live` in `config/settings.yaml`.
5. **Standing approval:** `python3 scripts/approve_live.py --days 10` (max 20 trading days, voided by any strategy change).
6. **Start automation:** install the services in `deploy/README.md`, or run `python3 apps/worker/scheduler.py --mode live`.

Every trading morning the scheduler then:
- **08:50:** runs the connection check.
- **09:05:** runs the live gate (approval, kill switch, smoke test, login, funds, reconciliation) and starts the
  bot **LIVE**. If any check fails, it starts **PAPER** and sends a `LIVE_GATE_FAILED` alert.
- **15:45:** refreshes data, IV and features, then makes a verified backup.

Stop automation at any time with `python3 scripts/approve_live.py --revoke` (next morning runs paper),
`python3 scripts/kill.py --on "stop"` (immediate), or `systemctl --user stop nifty-scheduler`.

## 5. Risks and limitations (read before trading)

1. **Selection bias.** This strategy was found after testing several setups and filters and keeping the best one. Expect live results to be weaker than the backtest.
2. **Losing years and streaks.** It lost money in 2024 and 2025 and had up to 11 losses in a row. A ₹10–11k drawdown (about half the capital) is normal behaviour, not a malfunction.
3. **Drawdown larger than capital.** The worst backtest drawdown (~₹23k) exceeds ₹20k. It came after earlier profits; starting at a bad time could exhaust the account.
4. **The ₹1,000 risk is a plan, not a guarantee.** Gaps, fast markets, an unfilled stop-limit order or an outage can lose more, up to the full premium (~₹12–19k).
5. **Simulated fills.** No historical bid/ask exists, so the backtest assumes spreads. Real fills may be worse.
6. **First real orders.** The live execution path is tested against documented Groww responses, not yet against real fills. The smoke test (4.2) reduces, but does not remove, this risk.
7. **Rare signals.** The strategy trades about once a week. A few months of live results are still too few trades to judge it.
8. **Regulation.** Check current SEBI and NSE rules for retail algorithmic trading and Groww's API terms before running automated orders.

---

## 6. Where things live
| Item | Location |
|---|---|
| Strategy config | `config/strategies.yaml` → `orb_retest30_cheapiv_v1` |
| Risk limits | `config/risk.yaml` |
| Session times, mode | `config/settings.yaml` |
| Strategy code | `strategies/orb/orb_core.py`, `strategies/orb/orb_strategy.py`, `strategies/filters.py` |
| Trading engine | `apps/worker/engine.py` |
| Ledger (source of truth) | `state/ledger.sqlite3` |
| Governance history | `state/governance.json` |
| Backtest outputs | `data/backtests/` |
| Other docs | `docs/RISK_MANAGEMENT.md`, `docs/BACKTESTING_METHODOLOGY.md`, `docs/KNOWN_LIMITATIONS.md` |
