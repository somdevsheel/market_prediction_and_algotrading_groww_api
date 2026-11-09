"""Event-driven intraday options backtester using *actual historical option-contract candles*.

Per trading day:
1. Replay completed 5-minute NIFTY candles; at each candle close the strategy sees only data up to that close
   (a ~2-week trailing window, plus the same-timestamp futures candle) and the gold daily features known
   before the open (HV/EWMA/GARCH/VIX/IV/HMM/events) and, if configured, a walk-forward ML probability.
2. On a signal, pick the 1-ITM contract of the nearest valid expiry, load that contract's real 5-minute
   candles, and fill at the *next* candle's open plus modelled half-spread and slippage (scenario-dependent).
   No bid/ask history exists, so spread is an explicit assumption and fills are simulated.
3. Stop / target / underlying-invalidation / time exits are evaluated candle by candle. If stop and target
   are both inside one candle, the stop is assumed first (conservative). Gaps through the stop fill at the open.
4. The same risk engine (``risk.pre_trade``) and charges model as live are used.

Sensitivity knobs: ``charges_multiplier``, ``drop_option_candle_frac`` (data-quality), scenarios (spread/slippage).
Partial fills are not modelled: historical depth is unavailable (1 lot assumed fully filled; documented).
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backtesting.costs.charges import option_charges
from backtesting.reports.metrics import compute_metrics, equity_curve
from backtesting.validation.robustness import robustness_report
from core.clock import IST, MarketCalendar, parse_hhmm
from execution.contract_selector import choose_expiry, itm_strike
from ingestion.historical.datalake import DataLake
from ingestion.historical.downloader import ensure_candles
from ingestion.instruments.calendar import ExpiryCalendar
from ingestion.providers.groww import groww_option_symbol
from quant.features.daily_features import features_for
from quant.options.bsm import greeks, implied_vol, year_fraction
from quant.regime.rules import RegimeConfig, classify
from risk import pre_trade
from strategies.base import MarketContext
from strategies.filters import contract_iv_ok
from strategies.registry import build_strategy

log = logging.getLogger(__name__)
HIST_DAYS = 16


class LazyRegime(dict):
    """Regime dict computed on first access (most bars never need it)."""

    def __init__(self, fn):
        super().__init__()
        self._fn, self._done = fn, False

    def _load(self):
        if not self._done:
            self.update(self._fn())
            self._done = True

    def get(self, k, default=None):
        self._load()
        return super().get(k, default)

    def __getitem__(self, k):
        self._load()
        return super().__getitem__(k)

    def items(self):
        self._load()
        return super().items()


def lot_size_on(d: date, schedule: list[dict]) -> int:
    lot = schedule[0]["lot_size"]
    for s in sorted(schedule, key=lambda x: x["from"]):
        if d >= date.fromisoformat(s["from"]):
            lot = s["lot_size"]
    return int(lot)


@dataclass
class BacktestResult:
    run_id: str
    config: dict
    trades: pd.DataFrame
    metrics: dict
    robustness: dict
    skipped: list = field(default_factory=list)
    data_issues: list = field(default_factory=list)
    equity: pd.DataFrame = field(default_factory=pd.DataFrame)

    def save(self, root: Path) -> Path:
        d = root / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        self.trades.to_csv(d / "trades.csv", index=False)
        self.equity.to_csv(d / "equity.csv", index=False)
        (d / "summary.json").write_text(json.dumps({
            "run_id": self.run_id, "config": self.config, "metrics": self.metrics, "robustness": self.robustness,
            "skipped": self.skipped[:500], "data_issues": self.data_issues[:500],
            "disclaimer": "Simulated fills on historical candles with modelled spread/slippage; not actual fills."},
            indent=2, default=str))
        return d


class Backtester:
    def __init__(self, cfg: dict, provider, lake: DataLake, scenario: str = "base",
                 strategy_overrides: Optional[dict] = None, experiment: str = "default",
                 strategy_id: Optional[str] = None, filter_overrides: Optional[dict] = None,
                 features: Optional[pd.DataFrame] = None, ml_probs: Optional[dict] = None,
                 charges_multiplier: float = 1.0, drop_option_candle_frac: float = 0.0, seed: int = 0,
                 calendar_preload: Optional[dict] = None, offline: bool = False, interval: Optional[int] = None):
        self.cfg, self.provider, self.lake = cfg, provider, lake
        self.settings, self.risk_cfg, self.inst = cfg["settings"], cfg["risk"], cfg["instruments"]
        self.scn_name = scenario
        self.scn = self.settings["backtest"]["scenarios"][scenario]
        sid = strategy_id or cfg["strategies"]["active_strategy"]
        self.scfg = cfg["strategies"]["strategies"][sid]
        self.overrides = strategy_overrides or {}
        self.strategy = build_strategy(sid, self.scfg, self.overrides, filter_overrides)
        if hasattr(getattr(self.strategy, "base", self.strategy), "exit_signal"):
            raise ValueError(f"{sid} uses signal-driven exits/reversals (EMA crossover); this generic backtester "
                             "cannot simulate them (see archive/scripts/ema921_research.py).")
        self.filters = {**(self.scfg.get("filters") or {}), **(filter_overrides or {})}
        self.risk_cfg = {**self.risk_cfg, **(self.scfg.get("risk_overrides") or {})}
        self.interval = interval or getattr(self.strategy, "interval_minutes", 5)
        self.iv_str = f"{self.interval}minute"
        self.experiment = experiment
        self.cal = MarketCalendar(self.settings, self.inst)
        self.tick = float(self.inst["tick_size"])
        self.r = float(self.settings["backtest"]["risk_free_rate"])
        self.features = features
        self.ml_probs = ml_probs or {}
        self.charges_mult = charges_multiplier
        self.drop_frac = drop_option_candle_frac
        self.rng = np.random.default_rng(seed)
        self.issues: list = []
        self.offline = offline
        self.calendar = ExpiryCalendar(None if offline else provider, self.inst["underlying"],
                                       lake.root / "manifests" / "expiries_cache.json", calendar_preload)
        self._opt_cache: dict[tuple[str, date], pd.DataFrame] = {}

    # ------------------------------------------------------------- data
    def _candles(self, ds: str, sym: str, seg: str, s: date, e: date) -> pd.DataFrame:
        if self.offline:  # read the lake only (no API) - used by parallel signal workers
            return self.lake.read_silver(ds, sym, self.iv_str, datetime.combine(s, time.min, IST),
                                         datetime.combine(e, time.max, IST))
        return ensure_candles(self.provider, self.lake, ds, sym, seg, self.iv_str, s, e)

    def load_underlying(self, start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
        idx = self._candles("nifty", self.inst["index_groww_symbol"], "CASH", start - timedelta(days=HIST_DAYS + 7), end)
        need_fut = self.strategy.params.get("use_vwap_filter") or self.strategy.params.get("use_volume_filter")
        if not need_fut:
            return idx, pd.DataFrame()
        by_sym: dict[str, list[date]] = {}
        d = start
        while d <= end:
            if d.weekday() < 5:
                s = self.calendar.futures_symbol_for(d)
                if s:
                    by_sym.setdefault(s, []).append(d)
            d += timedelta(days=1)
        futs = []
        for s, ds in by_sym.items():
            f = self._candles("futures", s, "FNO", min(ds), max(ds))
            if f.empty:
                self.issues.append({"kind": "futures_missing", "symbol": s})
                continue
            futs.append(f[f["timestamp"].dt.date.isin(ds)].assign(contract=s))
        fut = pd.concat(futs, ignore_index=True).sort_values("timestamp") if futs else pd.DataFrame()
        return idx, fut

    def option_day(self, symbol: str, d: date) -> pd.DataFrame:
        k = (symbol, d)
        if k not in self._opt_cache:
            df = ensure_candles(self.provider, self.lake, "options", symbol, "FNO", self.iv_str, d, d)
            if self.drop_frac > 0 and len(df):
                df = df[self.rng.random(len(df)) >= self.drop_frac].reset_index(drop=True)
            self._opt_cache[k] = df
        return self._opt_cache[k]

    # ------------------------------------------------------------ fills
    def buy_fill(self, px: float) -> float:
        return round(px * (1 + self.scn["spread_pct"] / 200) + self.scn["slippage_ticks"] * self.tick, 2)

    def sell_fill(self, px: float) -> float:
        return max(self.tick, round(px * (1 - self.scn["spread_pct"] / 200) - self.scn["slippage_ticks"] * self.tick, 2))

    # --------------------------------------------------------- signals
    def iter_signals(self, idx: pd.DataFrame, fut: pd.DataFrame, days: list[date], first_only: bool = True):
        """Yield (day, decision_ts, signal, features, day_idx) — the causal replay used by run() and the ML pipeline."""
        sess = self.settings["session"]
        entry_start, entry_end = parse_hhmm(sess["entry_start"]), parse_hhmm(sess["entry_end"])
        dates = idx["timestamp"].dt.date
        for d in days:
            hist = idx[(dates <= d) & (dates >= d - timedelta(days=HIST_DAYS))].reset_index(drop=True)
            day_idx = hist[hist["timestamp"].dt.date == d]
            fut_day = fut[fut["timestamp"].dt.date == d] if len(fut) else None
            feats = features_for(self.features, d)
            vix = feats.get("vix_prev")
            for _, bar in day_idx.iterrows():
                decision = bar["timestamp"] + pd.Timedelta(minutes=self.interval)
                if not (entry_start <= decision.time() < entry_end):
                    continue
                visible = hist[hist["timestamp"] <= bar["timestamp"]]
                fvis = fut_day[fut_day["timestamp"] <= bar["timestamp"]] if fut_day is not None else None
                regime = LazyRegime(lambda v=visible: classify(v.tail(300), vix, RegimeConfig(
                    vix_unstable=self.strategy.params.get("vix_unstable_level", 25))))
                f = dict(feats)
                if self.ml_probs:
                    f["ml_prob"] = self.ml_probs.get(decision.isoformat())
                ctx = MarketContext(now=decision.to_pydatetime(), index_5m=visible, futures_5m=fvis, regime=regime,
                                    vix=vix, features=f)
                sig = self.strategy.evaluate(ctx)
                if sig:
                    yield d, decision, sig, f, day_idx
                    if first_only:
                        break

    # -------------------------------------------------------------- run
    def signals(self, start: date, end: date) -> tuple[list, list]:
        """All candidate signals (every signal per day, in order) + the list of sessions. Cacheable/picklable."""
        idx, fut = self.load_underlying(start, end)
        if idx.empty:
            raise RuntimeError("no index data for the requested range")
        days = sorted(d for d in idx["timestamp"].dt.date.unique() if start <= d <= end and self.cal.is_trading_day(d))
        out = []
        for d, decision, sig, feats, day_idx in self.iter_signals(idx, fut, days, first_only=False):
            out.append((d, decision, sig, feats, day_idx[["timestamp", "high", "low", "close"]].copy()))
        return out, days

    def run(self, start: date, end: date, precomputed: tuple[list, list] | None = None) -> BacktestResult:
        sigs, days = precomputed or self.signals(start, end)
        trades, skipped = [], []
        capital = float(self.risk_cfg["capital"])
        realized = 0.0
        sq = parse_hhmm(self.settings["session"]["square_off"])
        traded_days = set()
        for d, decision, sig, feats, day_idx in sigs:
            if d in traded_days:
                continue
            res = self.simulate_trade(sig, d, decision, day_idx, capital + realized, sq, feats)
            if res.get("skipped"):
                skipped.append(res)
                continue
            traded_days.add(d)
            trades.append(res)
            realized += res["net_pnl"]
        return self._result(trades, skipped, days, start, end, capital)

    def _result(self, trades, skipped, days, start, end, capital) -> BacktestResult:
        tdf = pd.DataFrame(trades)
        config = {"experiment": self.experiment, "scenario": self.scn_name, "scenario_params": self.scn,
                  "start": str(start), "end": str(end), "strategy": self.strategy.strategy_id,
                  "strategy_version": self.strategy.version, "params": self.strategy.params, "filters": self.filters,
                  "risk": self.risk_cfg, "charges": self.inst["charges"], "charges_multiplier": self.charges_mult,
                  "drop_option_candle_frac": self.drop_frac,
                  "contract_selection": self.inst["contract_selection"],
                  "data_source": getattr(self.provider, "name", "lake"),
                  "features_rows": 0 if self.features is None else len(self.features),
                  "ml_predictions": len(self.ml_probs), "generated_at": datetime.now(IST).isoformat()}
        metrics = compute_metrics(tdf, capital, len(days))
        rob = robustness_report(tdf, days, capital, self.settings["backtest"]["out_of_sample_fraction"]) if len(tdf) else {}
        run_id = f"{datetime.now(IST):%Y%m%d-%H%M%S}-{self.experiment}-{self.scn_name}-{uuid.uuid4().hex[:4]}"
        return BacktestResult(run_id, config, tdf, metrics, rob, skipped,
                              self.issues + self.calendar.errors + (getattr(self.provider, "failures", None) or []),
                              equity_curve(tdf, capital) if len(tdf) else pd.DataFrame())

    def simulate_trade(self, sig, d: date, decision: pd.Timestamp, day_idx: pd.DataFrame, equity: float,
                       square_off: time, feats: dict | None = None) -> dict:
        opt = "CE" if sig.direction == "BULLISH" else "PE"
        expiry = choose_expiry(self.calendar.around(d), d, self.inst["contract_selection"].get("avoid_expiry_day", True))
        base = {"day": str(d), "signal_time": decision.isoformat(), "direction": sig.direction}
        if expiry is None:
            return {**base, "skipped": True, "reason": "no_valid_expiry"}
        strike = itm_strike(sig.reference_price, opt, self.inst["strike_step"],
                            self.inst["contract_selection"].get("itm_steps", 1))
        sym = groww_option_symbol(self.inst["underlying"], expiry, strike, opt)
        oc = self.option_day(sym, d)
        if oc.empty:
            return {**base, "skipped": True, "reason": "no_option_data", "symbol": sym}
        after = oc[oc["timestamp"] >= decision].reset_index(drop=True)
        if after.empty or after["timestamp"].iloc[0] != decision:
            return {**base, "skipped": True, "reason": "no_option_candle_at_entry", "symbol": sym}
        e_bar = after.iloc[0]
        cs = self.inst["contract_selection"]
        if not (e_bar["volume"] or 0) > 0:
            return {**base, "skipped": True, "reason": "zero_volume_entry_candle", "symbol": sym}
        entry = self.buy_fill(float(e_bar["open"]))
        if not (cs["min_premium"] <= entry <= cs["max_premium"]):
            return {**base, "skipped": True, "reason": f"premium_out_of_bounds:{entry}", "symbol": sym}
        lot = lot_size_on(d, self.inst["lot_size_schedule"])
        qty = lot * int(self.risk_cfg["max_lots"])
        t = year_fraction(decision.to_pydatetime(), expiry)
        iv = implied_vol(float(e_bar["open"]), sig.reference_price, strike, t, self.r, opt)
        ok_iv, iv_note = contract_iv_ok(iv * 100 if iv else None, feats or {}, self.filters)
        if not ok_iv:
            return {**base, "skipped": True, "reason": f"contract_iv_filter:{iv_note}", "symbol": sym}
        delta = greeks(sig.reference_price, strike, t, self.r, iv, opt).delta if iv else None
        pts, how = pre_trade.compute_stop_points(entry, delta, sig.reference_price - sig.invalidation_level, qty,
                                                 self.risk_cfg, self.inst["charges"], self.tick)
        stop = round(entry - pts, 2)
        target = round(entry + self.risk_cfg["reward_risk"] * pts, 2)
        plan = pre_trade.TradePlan("paper", sym, opt, lot, int(self.risk_cfg["max_lots"]), entry, stop, target,
                                   decision.to_pydatetime(), self.tick)
        dec = pre_trade.evaluate(plan, decision.to_pydatetime(), pre_trade.DayState(0, 0.0, 0, 0, False), equity,
                                 False, True, self.risk_cfg, self.inst["charges"], False)
        if not dec.approved:
            return {**base, "skipped": True, "reason": "risk_rejected", "failures": dec.failures, "symbol": sym}
        inv = sig.invalidation_level
        exit_px = exit_reason = exit_time = None
        pending_inv_exit = False
        idx_close = dict(zip(day_idx["timestamp"], day_idx["close"]))
        touch = (sig.diagnostics or {}).get("stop_mode") == "touch" and "low" in day_idx
        idx_low = dict(zip(day_idx["timestamp"], day_idx["low"])) if touch else {}
        idx_high = dict(zip(day_idx["timestamp"], day_idx["high"])) if touch else {}
        for i, b in after.iterrows():
            ts = b["timestamp"]
            if ts.time() >= square_off or pending_inv_exit:
                exit_px, exit_reason, exit_time = self.sell_fill(float(b["open"])), (
                    "underlying_invalidation" if pending_inv_exit else "time_exit"), ts
                break
            if i > 0 and b["open"] <= stop:
                exit_px, exit_reason, exit_time = self.sell_fill(float(b["open"])), "stop_gap", ts
                break
            if b["low"] <= stop:
                exit_px, exit_reason, exit_time = self.sell_fill(stop), "stop_loss", ts
                break
            if b["high"] >= target:
                exit_px, exit_reason, exit_time = self.sell_fill(target), "target", ts
                break
            if touch:  # underlying stop on touch (ORB): exit at this option candle's close (stop hit within it)
                lo, hi = idx_low.get(ts), idx_high.get(ts)
                if (sig.direction == "BULLISH" and lo is not None and lo <= inv) or \
                        (sig.direction == "BEARISH" and hi is not None and hi >= inv):
                    exit_px, exit_reason, exit_time = self.sell_fill(float(b["close"])), "underlying_stop", ts
                    break
                continue
            c = idx_close.get(ts)
            if c is not None and ((sig.direction == "BULLISH" and c < inv) or (sig.direction == "BEARISH" and c > inv)):
                pending_inv_exit = True
        if exit_px is None:
            last = after.iloc[-1]
            exit_px, exit_reason, exit_time = self.sell_fill(float(last["close"])), "end_of_data", \
                last["timestamp"] + pd.Timedelta(minutes=self.interval)
            self.issues.append({"kind": "option_data_ended_before_square_off", "symbol": sym, "day": str(d)})
        ch = option_charges(entry * qty, exit_px * qty, self.inst["charges"]).total * self.charges_mult
        gross = (exit_px - entry) * qty
        feats = feats or {}
        return {"trading_day": str(d), "trade_id": f"BT-{d:%Y%m%d}", "signal_id": sig.signal_id,
                "strategy_id": sig.strategy_id, "model_version": sig.model_version, "direction": sig.direction,
                "option_type": opt, "trading_symbol": sym, "strike": strike, "expiry": str(expiry),
                "expiry_day": d == expiry, "days_to_expiry": (expiry - d).days, "lot_size": lot, "qty": qty,
                "entry_time": decision.isoformat(), "entry_price": entry, "raw_entry_open": float(e_bar["open"]),
                "stop": stop, "target": target, "stop_method": how,
                "iv_at_entry": round(iv * 100, 2) if iv else None, "delta_at_entry": round(delta, 3) if delta else None,
                "exit_time": pd.Timestamp(exit_time).isoformat(), "exit_price": exit_px, "exit_reason": exit_reason,
                "gross_pnl": round(gross, 2), "charges": round(ch, 2), "net_pnl": round(gross - ch, 2),
                "planned_risk": dec.planned_loss, "regime": sig.regime, "hmm_label": feats.get("hmm_label"),
                "events": feats.get("events"), "ml_prob": feats.get("ml_prob"),
                "reason_codes": ",".join(sig.reason_codes)}
