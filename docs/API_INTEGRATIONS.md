# API integrations

## Groww Trade API (SDK `growwapi` 1.5.0; docs read 2026-11-09)
Used endpoints (all verified against the installed SDK source and live read-only calls):
| purpose | SDK method | notes |
|---|---|---|
| candles | `get_historical_candles(exchange, segment, groww_symbol, start, end, candle_interval)` | max range/request: 1–5 min 30 d, 10–30 min 90 d, ≥1 h 180 d; data from 2020 |
| expiries | `get_expiries(exchange, underlying, year, month)` | |
| contracts | `get_contracts(exchange, underlying, expiry)` | returns groww symbols incl. expired |
| quote | `get_quote(trading_symbol, exchange, segment)` | `depth.buy/sell[0]` = best bid/ask, `last_trade_time` epoch s |
| LTP | `get_ltp(("NSE_NIFTY","NSE_INDIAVIX"), "CASH")` | |
| option chain | `get_option_chain(exchange, underlying, expiry_date)` | per-strike ltp, OI, volume, Greeks incl. IV |
| funds | `get_available_margin_details()` | `fno_margin_details.option_buy_balance_available` |
| orders | `place_order`, `cancel_order`, `get_order_status_by_reference`, `get_order_detail` | `order_reference_id` 8–20 alnum, ≤2 hyphens |
| positions | `get_positions_for_user(segment)` | net qty field `quantity` |

Rate limits (documented): orders 10/s 250/min; live data 10/s 300/min; non-trading 20/s 500/min;
token generation 5/s 30/min and 150/day. Historical category is undocumented → we use 4/s 200/min.
Configured slightly below the limits in `config/settings.yaml`.

Order status enum (docs): NEW, ACKED, TRIGGER_PENDING, APPROVED, REJECTED, FAILED, EXECUTED, DELIVERY_AWAITED,
CANCELLED, CANCELLATION_REQUESTED, MODIFICATION_REQUESTED, COMPLETED (examples also show OPEN). We derive
fills from `filled_quantity`; any unknown status → UNKNOWN → reconciliation.

Auth: daily access token (expires ~06:00 IST), or API key + TOTP (pyotp) or API key + secret (needs daily
approval on Groww). Token is never logged (`core/log.py` redaction).

Not yet verified (requires a funded live test): whether `MIS` is accepted for NFO options on Groww, whether
SL (stop-limit) sell orders on index options are accepted, exact `average_fill_price` semantics.

## Zerodha
Not implemented. The `HistoricalDataProvider`/`LiveDataProvider`/`BrokerAdapter` interfaces are ready for it.
