"""Intraday trading engine shared by paper and live modes.

Only the ``BrokerAdapter`` differs between modes; market data, strategy,
contract selection, risk checks, order state machine and ledger are identical.

``TradingEngine.step()`` is a single, idempotent iteration that can be driven by
the real-time loop (``scripts/run_trader.py``) or by tests with a frozen clock.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime, time, timedelta
from typing import Optional

import pandas as pd

from backtesting.costs.charges import option_charges
from core.clock import IST, Clock, MarketCalendar, parse_hhmm
from core.config import active_strategy_config, resolve_path
from execution.broker_interface import BrokerAdapter, OrderRequest
from execution.contract_selector import choose_expiry, select_contract
from execution.order_manager import OrderManager, make_order_ref
from execution.order_state import OrderState as S
from ledger.store import Ledger
from quant.features.indicators import atr, ema, opening_range, session_vwap
from quant.regime.rules import RegimeConfig, classify
from risk import pre_trade
from risk.kill_switch import KillSwitch
from quant.features.daily_features import features_for
from strategies.base import MarketContext, Signal
from strategies.filters import FilteredStrategy, contract_iv_ok
from strategies.registry import build_strategy

log = logging.getLogger("engine")


def completed(df: pd.DataFrame, now: datetime, minutes: int, lag_s: int) -> pd.DataFrame:
    """Drop candles that have not closed yet (start + interval + lag > now)."""
    if df is None or df.empty:
        return df
    end = df["timestamp"] + pd.Timedelta(minutes=minutes) + pd.Timedelta(seconds=lag_s)
    return df[end <= pd.Timestamp(now)].reset_index(drop=True)


def round_tick(x: float, tick: float = 0.05) -> float:
    return round(round(x / tick) * tick, 2)


def live_approved(cfg: dict, today: date, strategy_status: str) -> tuple[bool, str]:
    """Live trading needs: mode=live, strategy APPROVED_FOR_LIVE, and an owner approval covering today
    (single day, or a standing approval with ``valid_until``; revoke any time with scripts/approve_live.py --revoke)."""
    if cfg["settings"].get("mode") != "live":
        return False, "settings.mode is not live"
    if strategy_status != "APPROVED_FOR_LIVE":
        return False, f"strategy status is {strategy_status}, not APPROVED_FOR_LIVE"
    p = resolve_path(cfg, "live_approval_file")
    if not p.exists():
        return False, "no live approval file (run scripts/approve_live.py)"
    try:
        a = json.loads(p.read_text())
    except Exception:
        return False, "approval file unreadable"
    start, until = a.get("date"), a.get("valid_until") or a.get("date")
    if not start or not (start <= today.isoformat() <= until):
        return False, f"approval covers {start}..{until}, not {today}"
    from governance.registry import config_hash

    sid = cfg["strategies"]["active_strategy"]
    if a.get("config_hash") and a["config_hash"] != config_hash(cfg["strategies"]["strategies"][sid]):
        return False, "strategy config changed after the live approval was given"
    return True, f"approved until {until}"


class TradingEngine:
    def __init__(self, cfg: dict, provider, broker: BrokerAdapter, ledger: Ledger, clock: Clock,
                 kill: KillSwitch, sleep=None):
        import time as _t

        self.cfg = cfg
        self.settings, self.risk_cfg, self.inst = cfg["settings"], cfg["risk"], cfg["instruments"]
        self.mode = broker.mode
        self.provider, self.broker, self.ledger, self.clock, self.kill = provider, broker, ledger, clock, kill
        self.sleep = sleep or _t.sleep
        self.cal = MarketCalendar(self.settings, self.inst)
        self.om = OrderManager(broker, ledger, self.settings, sleep=self.sleep)
        sid, scfg = active_strategy_config(cfg)
        self.strategy = build_strategy(sid, scfg)
        self.scfg = scfg
        self.filters = scfg.get("filters") or {}
        self.risk_cfg = {**self.risk_cfg, **(scfg.get("risk_overrides") or {})}
        base = getattr(self.strategy, "base", self.strategy)
        self.iv = int(getattr(base, "interval_minutes", 5))  # strategy candle interval in minutes
        self.history_days = int(getattr(base, "history_days", 4 if self.iv == 1 else 10))
        if scfg.get("contract_overrides"):
            self.inst = {**self.inst, "contract_selection": {**self.inst["contract_selection"], **scfg["contract_overrides"]}}
        if scfg.get("square_off"):
            from core.clock import parse_hhmm as _p

            self.cal.square_off = _p(scfg["square_off"])
        self.iv_str = f"{self.iv}minute"
        from governance.registry import effective_status

        self.strategy_status = effective_status(cfg, sid)
        self.features_table = self._load_features()
        self._attach_ml_scorer()
        from monitoring.alerts import AlertManager

        self.alerts = AlertManager(ledger)
        self._last_health = None
        self.lag = int(self.settings["session"].get("candle_close_lag_seconds", 5))
        self.ex = self.settings["execution"]
        self.tick = float(self.inst.get("tick_size", 0.05))
        self.last_decision_candle: Optional[pd.Timestamp] = None
        self.trade: Optional[dict] = None
        self.blocked_reason: Optional[str] = None
        self.done_for_day = False
        self._expiries: Optional[list[date]] = None
        self._fut_symbol: Optional[str] = None
        self._last_invalidation_check: Optional[pd.Timestamp] = None
        self._acted: set[str] = set()

    def _load_features(self):
        p = resolve_path(self.cfg, "data_lake") / "gold" / "volatility" / "daily_features.parquet"
        if not p.exists():
            return None
        try:
            return pd.read_parquet(p)
        except Exception as e:  # noqa: BLE001
            log.warning("could not read daily features: %s", e)
            return None

    def _attach_ml_scorer(self) -> None:
        if not (isinstance(self.strategy, FilteredStrategy) and "min_ml_prob" in self.filters):
            return
        from quant.machine_learning import pipeline as ml

        base_id = self.scfg.get("ml_base", self.strategy.strategy_id)
        mp = resolve_path(self.cfg, "ledger_db").parent / "models" / f"ml_{base_id}_{self.scfg.get('ml_model', 'logistic')}.pkl"
        if not mp.exists():
            log.warning("ML model %s missing: ML-filtered signals will be rejected", mp)
            return
        bundle = ml.load_model(mp)

        def scorer(sig, feats):
            ts = pd.Timestamp(sig.timestamp)
            return ml.predict_one(bundle, ml.sample_row(ts.date(), ts, sig, feats))

        self.strategy.ml_scorer = scorer

    # ================================================================ startup
    def startup(self) -> None:
        today = self.clock.today()
        if self.mode == "live":
            ok, why = live_approved(self.cfg, today, self.strategy_status)
            if not ok:
                raise RuntimeError(f"LIVE TRADING LOCKED: {why}")
        self.recover(today)
        self.ledger.set_status("worker", {"mode": self.mode, "started_at": self.clock.now().isoformat(),
                                          "strategy": self.strategy.strategy_id, "strategy_status": self.strategy_status})

    def recover(self, today: date) -> None:
        """Rebuild state from the ledger and reconcile with the broker before allowing new trades."""
        for o in self.ledger.working_orders():
            if o["mode"] != self.mode:
                continue
            if self.mode == "paper":
                self.ledger.update_order(o["order_id"], S.CANCELLED.value, {"reason": "paper session restarted"})
            else:
                self.om.reconcile(o["order_id"])
        broker_pos = self.broker.get_net_positions() if self.mode == "live" else {}
        known = set()
        for t in self.ledger.open_trades():
            if t["mode"] != self.mode:
                continue
            known.add(t["trading_symbol"])
            if t["trading_day"] != today.isoformat():
                self.ledger.update_trade(t["trade_id"], "RECONCILIATION_REQUIRED", status="RECONCILIATION_REQUIRED")
                self._block(f"open trade {t['trade_id']} from {t['trading_day']} needs manual reconciliation")
                continue
            if t["status"] == "PENDING_ENTRY":
                o = self.ledger.get_order(make_order_ref(t["trade_id"], "ENTRY"))
                if o and o["filled_qty"]:
                    self.ledger.update_trade(t["trade_id"], "RECOVERED_ENTRY", status="OPEN", qty=o["filled_qty"],
                                             entry_price=o["avg_price"])
                else:
                    self.ledger.update_trade(t["trade_id"], "ENTRY_NOT_FILLED", status="CANCELLED")
                    continue
                t = self.ledger.get_trade(t["trade_id"])
            if self.mode == "live":
                bq = broker_pos.get(t["trading_symbol"], 0)
                if bq != t["qty"]:
                    self.ledger.update_trade(t["trade_id"], "RECONCILIATION_REQUIRED", status="RECONCILIATION_REQUIRED",
                                             details={"broker_qty": bq, "ledger_qty": t["qty"]})
                    self._block(f"broker qty {bq} != ledger qty {t['qty']} for {t['trading_symbol']}")
                    continue
            self.trade = self.ledger.get_trade(t["trade_id"])
            log.info("recovered open trade %s %s", t["trade_id"], t["trading_symbol"])
        if self.mode == "live":
            orphans = {k: v for k, v in broker_pos.items() if k not in known}
            if orphans:
                self._block(f"broker has positions not in ledger: {orphans}")
        if self.mode == "paper" and hasattr(self.broker, "restore"):
            realized = sum((t["net_pnl"] or 0) for t in self.ledger.trades("paper") if t["status"] == "CLOSED")
            self.broker.restore([self.trade] if self.trade else [], realized, self.cfg["settings"]["paper"]["starting_capital"])
        ds = self.ledger.day_stats(today, self.mode)
        if ds["reconciliation_required"]:
            self._block("a trade today requires reconciliation")

    def _block(self, reason: str) -> None:
        self.blocked_reason = reason
        self.kill.activate(reason, source="system")
        self.ledger.record_error("engine", reason)
        log.error("TRADING BLOCKED: %s", reason)
        if hasattr(self, "alerts"):
            self.alerts.alert("CRITICAL", "TRADING_BLOCKED", reason)

    def health_checks(self, now: datetime) -> None:
        """Data freshness, broker connectivity, stuck orders, metrics snapshot (at most every 30 s)."""
        if self._last_health and (now - self._last_health).total_seconds() < 30:
            return
        self._last_health = now
        from monitoring.metrics import METRICS

        self.ledger.set_status("metrics", METRICS.snapshot())
        if getattr(self.provider, "consecutive_failures", 0) >= 3:
            self.alerts.alert("ERROR", "BROKER_CONNECTIVITY",
                              f"{self.provider.consecutive_failures} consecutive Groww API failures")
        t = now.time()
        if self.cal.is_trading_day(now.date()) and time(9, 30) <= t < self.cal.close and self.last_decision_candle is not None:
            age = (pd.Timestamp(now) - (self.last_decision_candle + pd.Timedelta(minutes=self.iv))).total_seconds()
            if age > 12 * 60 and not self.done_for_day:
                self.alerts.alert("ERROR", "DATA_STALE", f"no new completed 5m candle for {age / 60:.0f} min")
        for o in self.ledger.working_orders():
            if o["mode"] != self.mode:
                continue
            age = (now - datetime.fromisoformat(o["updated_at"])).total_seconds()
            if age > 120:
                self.alerts.alert("ERROR", f"ORDER_STUCK_{o['order_id']}", f"order {o['order_id']} {o['role']} in "
                                  f"{o['state']} for {age:.0f}s")

    # ============================================================ market data
    def _futures_symbol(self, today: date) -> Optional[str]:
        if self._fut_symbol is None:
            try:
                f = self.provider.futures_contracts(self.inst["underlying"])
                f = f[f["expiry_date"] >= today]
                self._fut_symbol = f.iloc[0]["groww_symbol"] if len(f) else None
            except Exception as e:  # noqa: BLE001
                log.warning("futures lookup failed: %s", e)
        return self._fut_symbol

    def market_context(self, now: datetime) -> MarketContext:
        today = now.date()
        start = datetime.combine(today - timedelta(days=10), time(9, 15), IST)
        if self.iv == 1:  # 1-minute strategies: limited history (strategy may ask for more)
            start = datetime.combine(today - timedelta(days=self.history_days), time(9, 15), IST)
        idx = self.provider.get_historical_candles(self.inst["index_groww_symbol"], start, now, self.iv_str, "CASH")
        idx = completed(idx, now, self.iv, self.lag)
        fut = None
        fs = self._futures_symbol(today)
        if fs:
            fut = self.provider.get_historical_candles(fs, datetime.combine(today, time(9, 15), IST), now, self.iv_str, "FNO")
            fut = completed(fut, now, self.iv, self.lag)
        vix = None
        try:
            vix = self.provider.get_ltp([f"NSE_{self.inst['vix_trading_symbol']}"], "CASH").get(
                f"NSE_{self.inst['vix_trading_symbol']}")
        except Exception as e:  # noqa: BLE001
            log.warning("vix fetch failed: %s", e)
        p = self.strategy.params
        regime = classify(idx, vix, RegimeConfig(vix_unstable=p.get("vix_unstable_level", 25))) if len(idx) else None
        feats = features_for(self.features_table, today)
        if self.filters and not feats:
            log.warning("no gold features for %s: run scripts/build_gold.py features (filters will reject)", today)
        return MarketContext(now=now, index_5m=idx, futures_5m=fut, vix=vix, regime=regime, features=feats)

    def _publish_market(self, ctx: MarketContext) -> None:
        idx = ctx.index_5m
        snap: dict = {"ts": ctx.now.isoformat(), "vix": ctx.vix, "regime": ctx.regime,
                      "strategy_reasons": self.strategy.last_reasons}
        if idx is not None and len(idx):
            snap["spot"] = float(idx["close"].iloc[-1])
            snap["last_candle"] = idx["timestamp"].iloc[-1].isoformat()
            for s in (9, 20, 50, 200):
                v = ema(idx["close"], s).iloc[-1]
                snap[f"ema{s}"] = None if pd.isna(v) else round(float(v), 2)
            a = atr(idx).iloc[-1]
            snap["atr14"] = None if pd.isna(a) else round(float(a), 2)
            day = idx[idx["timestamp"].dt.date == ctx.now.date()]
            rng = opening_range(day, int(self.strategy.params.get("opening_range_minutes", 15))) if len(day) else None
            if rng:
                snap["or_high"], snap["or_low"] = rng
        f = ctx.futures_5m
        if f is not None and len(f):
            snap["futures"] = float(f["close"].iloc[-1])
            snap["futures_vwap"] = round(float(session_vwap(f).iloc[-1]), 2)
        self.ledger.set_status("market", snap)

    # ================================================================== loop
    def step(self) -> None:
        now = self.clock.now()
        today = now.date()
        self.ledger.set_status("heartbeat", {"ts": now.isoformat(), "mode": self.mode,
                                             "position": self.trade["trading_symbol"] if self.trade else None,
                                             "blocked": self.blocked_reason, "kill": self.kill.info()})
        if not self.cal.is_trading_day(today):
            self.done_for_day = True
            return
        self.health_checks(now)
        t = now.time()
        if t < self.cal.open:
            return
        if self.trade:
            self.manage_position(now)
            return
        if t >= self.cal.stop_trading or self.cal.past_square_off(now):
            self.done_for_day = True
            return
        if self.kill.active():
            return
        if not self.cal.in_entry_window(now):
            if t >= self.cal.entry_end:
                self.done_for_day = True
            return
        ds = self.ledger.day_stats(today, self.mode)
        if ds["trades_today"] >= self.risk_cfg["max_trades_per_day"]:
            self.done_for_day = True
            return
        expected_last = (pd.Timestamp(now) - pd.Timedelta(minutes=self.iv, seconds=self.lag)).floor(f"{self.iv}min")
        if self.last_decision_candle is not None and expected_last <= self.last_decision_candle:
            return  # no new completed candle yet -> no API calls
        ctx = self.market_context(now)
        idx = ctx.index_5m
        if idx is None or idx.empty or idx["timestamp"].iloc[-1].date() != today:
            abort = parse_hhmm(self.settings["session"].get("no_data_abort_time", "09:25"))
            if self.last_decision_candle is not None:
                # we already saw today's data: this is a transient data failure, not a holiday -> retry next step
                self.ledger.record_error("engine", "index data unavailable this step; will retry")
                return
            if t >= abort:
                log.error("no fresh index candles today by %s -> treating as non-trading day", abort)
                self.ledger.record_error("engine", "no fresh index data; possible holiday or data outage")
                self.done_for_day = True
            return
        last_ts = idx["timestamp"].iloc[-1]
        if self.last_decision_candle is not None and last_ts <= self.last_decision_candle:
            return
        self.last_decision_candle = last_ts
        sig = self.strategy.evaluate(ctx)
        self._publish_market(ctx)
        log.info("candle %s evaluated: %s", last_ts.strftime("%H:%M"), self.strategy.last_reasons)
        self._act_on(sig, now)

    def _act_on(self, sig, now: datetime) -> None:
        """Act on a signal at most once (prevents re-entry on the same signal after a stop-out)."""
        if not sig or sig.signal_id in self._acted:
            return
        self._acted.add(sig.signal_id)
        self.ledger.record_signal(sig.to_dict(), self.mode, now.date())
        self.try_enter(sig, now)

    # ================================================================= entry
    def _expiry_list(self, today: date) -> list[date]:
        if self._expiries is None:
            ex: list[date] = []
            for y, m in {(today.year, today.month), ((today.replace(day=28) + timedelta(days=7)).year,
                                                     (today.replace(day=28) + timedelta(days=7)).month)}:
                ex += self.provider.get_expiries(self.inst["underlying"], y, m)
            self._expiries = sorted(set(ex))
        return self._expiries

    def _instrument_lookup(self, trading_symbol: str) -> Optional[dict]:
        df = self.provider.get_instruments()
        r = df[(df["exchange"] == "NSE") & (df["trading_symbol"] == trading_symbol)]
        return r.iloc[0].to_dict() if len(r) else None

    def try_enter(self, sig: Signal, now: datetime) -> None:
        today = now.date()
        rec = lambda ok, payload: self.ledger.record_risk_decision(sig.signal_id, ok, payload, today)  # noqa: E731
        try:
            expiries = self._expiry_list(today)
            expiry = choose_expiry(expiries, today, self.inst["contract_selection"].get("avoid_expiry_day", True))
            if expiry is None:
                rec(False, {"stage": "contract_selection", "reason": "no_valid_expiry"})
                return
            chain = self.provider.get_option_chain(self.inst["underlying"], expiry)
            spot = chain.underlying_ltp or sig.reference_price
            sel = select_contract(sig.direction, spot, now, expiries, chain, self._instrument_lookup,
                                  lambda s: self.provider.get_quote(s, "FNO"), self.inst,
                                  self.risk_cfg["max_quote_age_seconds"])
        except Exception as e:  # noqa: BLE001
            log.exception("contract selection failed")
            self.ledger.record_error("contract_selection", str(e))
            rec(False, {"stage": "contract_selection", "reason": f"error:{e}"})
            return
        if not sel.ok:
            log.info("contract rejected: %s", sel.reason)
            rec(False, {"stage": "contract_selection", "reason": sel.reason, "checks": sel.checks})
            return
        c, q = sel.contract, sel.quote
        ok_iv, iv_note = contract_iv_ok((sel.chain_row or {}).get("iv"), features_for(self.features_table, today),
                                        self.filters)
        if not ok_iv:
            rec(False, {"stage": "contract_iv_filter", "reason": iv_note, "checks": sel.checks})
            return
        lots = int(self.risk_cfg["max_lots"])
        qty = c.lot_size * lots
        limit = round_tick(q.ask + self.ex["limit_buffer_ticks"] * c.tick_size, c.tick_size)
        # size the stop from the worst acceptable entry (the limit), so planned loss holds for any fill <= limit
        stop_pts, how = pre_trade.compute_stop_points(limit, (sel.chain_row or {}).get("delta"),
                                                      sig.reference_price - sig.invalidation_level, qty,
                                                      self.risk_cfg, self.inst["charges"], c.tick_size)
        stop = round_tick(limit - stop_pts, c.tick_size)
        target = round_tick(limit + self.risk_cfg["reward_risk"] * stop_pts, c.tick_size)
        plan = pre_trade.TradePlan(mode=self.mode, trading_symbol=c.trading_symbol, option_type=c.option_type,
                                   lot_size=c.lot_size, lots=lots, entry_price=limit, stop_price=stop,
                                   target_price=target, quote_time=q.received_at or now, tick_size=c.tick_size)
        ds = self.ledger.day_stats(today, self.mode)
        funds = self.broker.get_available_funds()
        approved_live = live_approved(self.cfg, today, self.strategy_status)[0] if self.mode == "live" else False
        dec = pre_trade.evaluate(plan, now, pre_trade.DayState(ds["trades_today"], ds["realized_pnl_today"],
                                                               ds["open_positions"], len([o for o in self.ledger.working_orders() if o["mode"] == self.mode]),
                                                               ds["reconciliation_required"] or bool(self.blocked_reason)),
                                 funds, self.kill.active(), self.cal.in_entry_window(now), self.risk_cfg,
                                 self.inst["charges"], approved_live)
        payload = {"stage": "risk", "contract": c.to_dict(), "quote": {"bid": q.bid, "ask": q.ask, "ltp": q.ltp},
                   "stop_points": stop_pts, "stop_method": how, "stop": stop, "target": target, "limit": limit,
                   "planned_loss": dec.planned_loss, "worst_case_loss": dec.worst_case_loss,
                   "est_charges": dec.est_charges, "failures": dec.failures, "checks": dec.checks}
        rec(dec.approved, payload)
        self.ledger.set_status("last_decision", {"signal": sig.to_dict(), **payload, "approved": dec.approved})
        if not dec.approved:
            log.warning("RISK REJECTED %s: %s", c.trading_symbol, dec.failures)
            return
        trade_id = f"{today:%Y%m%d}-{uuid.uuid4().hex[:8]}"
        self.ledger.insert_trade({
            "trade_id": trade_id, "signal_id": sig.signal_id, "strategy_id": sig.strategy_id,
            "model_version": sig.model_version, "mode": self.mode, "trading_day": today.isoformat(),
            "trading_symbol": c.trading_symbol, "option_type": c.option_type, "strike": c.strike,
            "expiry": c.expiry.isoformat(), "qty": qty, "status": "PENDING_ENTRY", "expected_entry": q.ask,
            "planned_stop": stop, "target": target, "planned_risk": dec.planned_loss, "regime": sig.regime,
            "details": {"signal": sig.to_dict(), "stop_points": stop_pts, "stop_method": how,
                        "invalidation_level": sig.invalidation_level, "direction": sig.direction,
                        "lot_size": c.lot_size, "tick": c.tick_size, "worst_case_loss": dec.worst_case_loss,
                        "entry_quote": {"bid": q.bid, "ask": q.ask}, "chain_iv": (sel.chain_row or {}).get("iv"),
                        "entry_reason": sig.reason_codes}})
        req = OrderRequest(order_ref=make_order_ref(trade_id, "ENTRY"), trading_symbol=c.trading_symbol, side="BUY",
                           qty=qty, order_type="LIMIT", price=limit, product=self.ex["product"],
                           validity=self.ex["validity"])
        o = self.om.submit(req, trade_id=trade_id, signal_id=sig.signal_id, role="ENTRY", mode=self.mode,
                           trading_day=today)
        o = self.om.wait_terminal(o["order_id"], self.ex["fill_timeout_seconds"])
        if S(o["state"]) not in (S.FILLED,):
            o = self.om.cancel(o["order_id"])
        st = S(o["state"])
        if st == S.RECONCILIATION_REQUIRED or st not in (S.FILLED, S.CANCELLED, S.REJECTED, S.EXPIRED):
            self.ledger.update_trade(trade_id, "RECONCILIATION_REQUIRED", status="RECONCILIATION_REQUIRED")
            self._block(f"entry order {o['order_id']} in state {st.value}")
            return
        filled = int(o["filled_qty"] or 0)
        if filled <= 0:
            self.ledger.update_trade(trade_id, "ENTRY_NOT_FILLED", status="CANCELLED" if st != S.REJECTED else "REJECTED",
                                     exit_reason=f"entry_{st.value.lower()}")
            log.warning("entry not filled (%s)", st.value)
            return
        fill = float(o["avg_price"])
        stop = round_tick(fill - stop_pts, c.tick_size)
        target = round_tick(fill + self.risk_cfg["reward_risk"] * stop_pts, c.tick_size)
        self.ledger.update_trade(trade_id, "ENTRY_FILLED", status="OPEN", qty=filled, entry_price=fill,
                                 entry_time=self.clock.now().isoformat(), planned_stop=stop, target=target)
        self.trade = self.ledger.get_trade(trade_id)
        log.info("ENTERED %s %s x%d @ %.2f stop %.2f target %.2f [%s]", self.mode.upper(), c.trading_symbol, filled,
                 fill, stop, target, "simulated" if self.mode == "paper" else "live")
        if self.ex.get("broker_side_stop", True):
            self._place_protective_stop(stop)

    def _place_protective_stop(self, stop: float) -> None:
        t = self.trade
        tick = json.loads(t["details"]).get("tick", self.tick)
        limit = max(tick, round_tick(stop - self.ex["stop_limit_buffer_ticks"] * tick, tick))
        req = OrderRequest(order_ref=make_order_ref(t["trade_id"], "STOP"), trading_symbol=t["trading_symbol"],
                           side="SELL", qty=int(t["qty"]), order_type="SL", price=limit, trigger_price=stop,
                           product=self.ex["product"], validity=self.ex["validity"])
        o = self.om.submit(req, trade_id=t["trade_id"], signal_id=t["signal_id"], role="STOP", mode=self.mode,
                           trading_day=self.clock.today())
        if S(o["state"]) in (S.REJECTED, S.RECONCILIATION_REQUIRED):
            log.error("protective stop not accepted (%s); software stop remains active", o["state"])
            self.ledger.record_error("engine", f"protective stop {o['order_id']} {o['state']}")

    # ============================================================== position
    def _stop_order(self) -> Optional[dict]:
        if not self.trade:
            return None
        return self.ledger.get_order(make_order_ref(self.trade["trade_id"], "STOP"))

    def manage_position(self, now: datetime) -> None:
        t = self.trade
        so = self._stop_order()
        if so and S(so["state"]) not in (S.FILLED, S.CANCELLED, S.REJECTED, S.EXPIRED, S.CLOSED):
            so = self.om.refresh(so["order_id"])
        if so and S(so["state"]) == S.FILLED:
            self._close_trade(float(so["avg_price"]), "stop_loss_broker", t["planned_stop"])
            return
        if so and S(so["state"]) == S.PARTIALLY_FILLED:
            log.warning("protective stop partially filled; forcing exit of remainder")
            self.exit_position("stop_loss_partial", now)
            return
        if self.kill.active():
            self.exit_position("kill_switch", now)
            return
        if self.cal.past_square_off(now):
            self.exit_position("time_exit", now)
            return
        try:
            q = self.provider.get_quote(t["trading_symbol"], "FNO")
        except Exception as e:  # noqa: BLE001
            self.ledger.record_error("engine", f"quote failed while in position: {e}")
            return
        mtm = ((q.bid or q.ltp) - t["entry_price"]) * t["qty"]
        if not getattr(self, "_greeks_at", None) or (now - self._greeks_at).total_seconds() >= 60:
            self._greeks_at = now
            try:
                self._greeks = (self.provider.client.get_greeks("NSE", self.inst["underlying"], t["trading_symbol"],
                                                                t["expiry"]) or {}).get("greeks")
            except Exception:  # noqa: BLE001
                pass
        self.ledger.set_status("position", {"trade_id": t["trade_id"], "symbol": t["trading_symbol"], "qty": t["qty"],
                                            "entry": t["entry_price"], "bid": q.bid, "ask": q.ask, "ltp": q.ltp,
                                            "stop": t["planned_stop"], "target": t["target"], "mtm": round(mtm, 2),
                                            "greeks": getattr(self, "_greeks", None), "ts": now.isoformat()})
        ref = q.bid or q.ltp
        if ref is not None and ref <= t["planned_stop"]:
            self.exit_position("stop_loss", now)
            return
        if ref is not None and ref >= t["target"]:
            self.exit_position("target", now)
            return
        self._check_invalidation(now)

    def _strategy_exit(self, now: datetime) -> bool:
        """Strategies with exit_signal (e.g. EMA cross): exit on the opposite signal, then allow a same-bar reversal."""
        base = getattr(self.strategy, "base", self.strategy)
        if not hasattr(base, "exit_signal"):
            return False
        ctx = self.market_context(now)
        d = json.loads(self.trade["details"])["direction"]
        if base.exit_signal(ctx, d):
            self.exit_position("opposite_cross", now)
            if self.trade is None and not self.kill.active():
                self.done_for_day = False
                ds = self.ledger.day_stats(now.date(), self.mode)
                if ds["trades_today"] < self.risk_cfg["max_trades_per_day"] and self.cal.in_entry_window(now):
                    self._act_on(self.strategy.evaluate(ctx), now)  # same-bar reversal (strategy checks its window)
            return True
        return False

    def _check_invalidation(self, now: datetime) -> None:
        cutoff = pd.Timestamp(now) - pd.Timedelta(minutes=self.iv, seconds=self.lag)
        bucket = cutoff.floor(f"{self.iv}min")
        if self._last_invalidation_check is not None and bucket <= self._last_invalidation_check:
            return
        self._last_invalidation_check = bucket
        if self._strategy_exit(now):
            return
        try:
            idx = self.provider.get_historical_candles(self.inst["index_groww_symbol"],
                                                       datetime.combine(now.date(), time(9, 15), IST), now, self.iv_str,
                                                       "CASH")
        except Exception as e:  # noqa: BLE001
            log.warning("invalidation check failed: %s", e)
            return
        idx = completed(idx, now, self.iv, self.lag)
        if idx is None or idx.empty:
            return
        d = json.loads(self.trade["details"])
        inv = float(d["invalidation_level"])
        if ((d.get("signal") or {}).get("diagnostics") or {}).get("stop_mode") == "touch":  # ORB: stop on touch
            lo, hi = float(idx["low"].iloc[-1]), float(idx["high"].iloc[-1])
            if (d["direction"] == "BULLISH" and lo <= inv) or (d["direction"] == "BEARISH" and hi >= inv):
                self.exit_position("underlying_stop", now)
            return
        close = float(idx["close"].iloc[-1])
        if (d["direction"] == "BULLISH" and close < inv) or (d["direction"] == "BEARISH" and close > inv):
            self.exit_position("underlying_invalidation", now)

    def exit_position(self, reason: str, now: datetime) -> None:
        t = self.trade
        log.info("EXIT requested (%s) for %s", reason, t["trading_symbol"])
        self.ledger.update_trade(t["trade_id"], "EXIT_REQUESTED", status="EXIT_PENDING", exit_reason=reason)
        so = self._stop_order()
        if so and S(so["state"]) not in (S.FILLED, S.CANCELLED, S.REJECTED, S.EXPIRED):
            so = self.om.cancel(so["order_id"])
        so = self._stop_order()
        stop_filled = int(so["filled_qty"] or 0) if so else 0
        if so and stop_filled >= t["qty"]:
            self._close_trade(float(so["avg_price"]), "stop_loss_broker", t["planned_stop"])
            return
        if so and S(so["state"]) not in (S.FILLED, S.CANCELLED, S.REJECTED, S.EXPIRED):
            self.ledger.update_trade(t["trade_id"], "RECONCILIATION_REQUIRED", status="RECONCILIATION_REQUIRED")
            self._block(f"could not cancel protective stop {so['order_id']} ({so['state']})")
            return
        remaining = int(t["qty"]) - stop_filled
        fills: list[tuple[int, float]] = [(stop_filled, float(so["avg_price"]))] if stop_filled else []
        expected_exit = None
        for attempt in range(4):
            try:
                q = self.provider.get_quote(t["trading_symbol"], "FNO")
            except Exception as e:  # noqa: BLE001
                self.ledger.record_error("engine", f"exit quote failed: {e}")
                self.sleep(1)
                continue
            expected_exit = expected_exit or q.bid or q.ltp
            buf = self.ex["limit_buffer_ticks"] * (attempt + 1) * self.tick
            px = max(self.tick, round_tick((q.bid or q.ltp) - buf, self.tick))
            req = OrderRequest(order_ref=make_order_ref(t["trade_id"], "EXIT", attempt), trading_symbol=t["trading_symbol"],
                               side="SELL", qty=remaining, order_type="LIMIT", price=px, product=self.ex["product"],
                               validity=self.ex["validity"])
            o = self.om.submit(req, trade_id=t["trade_id"], signal_id=t["signal_id"], role="EXIT", mode=self.mode,
                               trading_day=now.date())
            o = self.om.wait_terminal(o["order_id"], self.ex["fill_timeout_seconds"] / 2)
            if S(o["state"]) != S.FILLED:
                o = self.om.cancel(o["order_id"])
            f = int(o["filled_qty"] or 0)
            if f:
                fills.append((f, float(o["avg_price"])))
                remaining -= f
            if S(o["state"]) == S.RECONCILIATION_REQUIRED:
                break
            if remaining <= 0:
                break
        if remaining > 0:
            self.ledger.update_trade(t["trade_id"], "EXIT_FAILED", status="RECONCILIATION_REQUIRED",
                                     details={**json.loads(t["details"]), "unfilled_qty": remaining})
            self._block(f"EXIT FAILED for {t['trading_symbol']}: {remaining} qty still open — act manually NOW")
            return
        avg = sum(q_ * p for q_, p in fills) / sum(q_ for q_, _ in fills)
        self._close_trade(avg, reason, expected_exit)

    def _close_trade(self, exit_price: float, reason: str, expected_exit: Optional[float]) -> None:
        t = self.ledger.get_trade(self.trade["trade_id"])
        qty, entry = int(t["qty"]), float(t["entry_price"])
        orders = [o for o in self.ledger.query("SELECT * FROM orders WHERE trade_id=?", (t["trade_id"],))
                  if (o["filled_qty"] or 0) > 0]
        n_buy = sum(1 for o in orders if o["side"] == "BUY") or 1
        n_sell = sum(1 for o in orders if o["side"] == "SELL") or 1
        ch = option_charges(entry * qty, exit_price * qty, self.inst["charges"], n_buy, n_sell)
        gross = (exit_price - entry) * qty
        slip = ((entry - (t["expected_entry"] or entry)) + ((expected_exit or exit_price) - exit_price)) * qty
        net = gross - ch.total
        self.ledger.update_trade(t["trade_id"], "CLOSED", status="CLOSED", exit_price=round(exit_price, 2),
                                 exit_time=self.clock.now().isoformat(), expected_exit=expected_exit,
                                 exit_reason=reason, gross_pnl=round(gross, 2), charges=ch.total,
                                 slippage=round(slip, 2), net_pnl=round(net, 2),
                                 realized_risk=round(max(0.0, -net), 2))
        log.info("CLOSED %s @ %.2f reason=%s gross=%.2f charges=%.2f net=%.2f (%s)", t["trading_symbol"], exit_price,
                 reason, gross, ch.total, net, self.mode)
        self.ledger.set_status("position", None)
        self.trade = None
        if self.ledger.day_stats(self.clock.today(), self.mode)["trades_today"] >= self.risk_cfg["max_trades_per_day"]:
            self.done_for_day = True
