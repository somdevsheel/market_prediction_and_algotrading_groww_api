# Risk management

Configured in `config/risk.yaml` (capital ₹20,000; 1 lot; 1 trade/day; risk budget ₹500–1,000; daily loss
limit ₹1,500; no overnight/re-entry/averaging).

Pre-trade checks (`risk/pre_trade.py`, all evaluated and logged): mode valid · live approved (live only) ·
kill switch off · entry window · no reconciliation pending · quote fresh (≤20 s) · lots ≤ max · daily trade
count · no open position / pending order · daily loss limit · stop < entry < target · **planned loss** =
(limit − stop)×qty + round-trip charges + slippage ≤ ₹1,000 · premium outlay ≤ limit and ≤ capital ·
funds available. Worst-case loss (full premium + charges) is computed and stored for every trade.

The planned loss is **not** a maximum: gaps, a stop-limit that does not fill, or exit failure can lose more,
up to the full premium (~₹16–19k for one 1-ITM lot at today's prices).

Execution safety
* Deterministic `order_reference_id` per (trade, role, attempt) → no duplicate intents.
* `place_order` is never retried; ambiguous failures → UNKNOWN → status by reference → RECONCILIATION_REQUIRED.
* Broker-side protective SL (stop-limit) after entry; before any exit the SL is cancelled and its status
  confirmed (prevents double-sell / accidental short). Software stop/target/invalidation/time exits run too.
* Exit retries up to 4 times with widening limit; if still unfilled → kill switch + RECONCILIATION_REQUIRED.
* Kill switch (file `state/KILL_SWITCH`): CLI, dashboard, or automatic; worker exits positions and blocks entries.
* Startup reconciliation: previous-day open trades, broker/ledger quantity mismatches or unknown broker
  positions activate the kill switch.

Monitoring & alerts (monitoring/alerts.py): TRADING_BLOCKED (critical), BROKER_CONNECTIVITY (≥3 consecutive API
failures), DATA_STALE (no new 5-min candle for 12 min in session), ORDER_STUCK (working order >120 s),
JOB_FAILED_* (scheduler). Alerts go to the ledger (dashboard → Health) and to `ALERT_WEBHOOK_URL` if set.
Token expiry: on an authentication error the provider re-logs in via TOTP / key+secret (if configured) and
retries once.

Governance: live requires the strategy's *effective* status APPROVED_FOR_LIVE (state/governance.json), which
is void if params/filters/version change after approval (config hash), plus same-day manual approval.
