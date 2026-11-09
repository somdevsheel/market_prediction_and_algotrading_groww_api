"""Composable signal filters (Strategies D, E, H, I are a base strategy + filters, configured in YAML).

Day-level filters read ``ctx.features`` (gold daily features known before the open, plus ``ml_prob``).
A missing feature FAILS the filter (never silently passes) unless ``allow_missing: true``.
The contract-level IV check (``contract_iv_max_over_rv``) is applied after contract selection by the
engine/backtester via ``contract_iv_ok``.
"""
from __future__ import annotations

from typing import Optional

from strategies.base import MarketContext, Signal, Strategy


class FilteredStrategy(Strategy):
    def __init__(self, base: Strategy, filters: dict, strategy_id: str, version: str, ml_scorer=None):
        super().__init__(base.params)
        self.base, self.filters = base, filters
        self.strategy_id, self.version = strategy_id, version
        self.ml_scorer = ml_scorer  # live: callable(signal, features) -> probability

    def _need(self, f: dict, key: str, R: list) -> Optional[float]:
        v = f.get(key)
        if v is None:
            if not self.filters.get("allow_missing", False):
                R.append(f"missing_feature:{key}")
            return None
        return v

    def check(self, ctx: MarketContext, direction: str) -> tuple[bool, list[str], list[str]]:
        F, f = self.filters, ctx.features or {}
        R: list[str] = []
        ok_codes: list[str] = []
        if F.get("block_events"):
            ev = [e for e in str(f.get("events") or "").split(",") if e]
            blocked = [e for e in ev if e in F["block_events"]]
            if blocked:
                R.append(f"event_risk:{','.join(blocked)}")
        if "hv_ratio_range" in F:
            v = self._need(f, "hv_ratio", R)
            if v is not None and not (F["hv_ratio_range"][0] <= v <= F["hv_ratio_range"][1]):
                R.append(f"hv_ratio_out:{v:.2f}")
            elif v is not None:
                ok_codes.append("hv_ok")
        for key, name in (("min_ewma_fc", "ewma_fc"), ("min_garch_fc", "garch_fc"), ("min_hv20", "hv20")):
            if key in F:
                v = self._need(f, name, R)
                if v is not None and v < F[key]:
                    R.append(f"{name}_low:{v:.1f}")
                elif v is not None:
                    ok_codes.append(f"{name}_ok")
        if "vix_range" in F:
            v = self._need(f, "vix_prev", R)
            if v is not None and not (F["vix_range"][0] <= v <= F["vix_range"][1]):
                R.append(f"vix_out:{v:.1f}")
        if "min_rv_iv_ratio" in F:
            v = self._need(f, "rv_iv_ratio", R)
            if v is not None and v < F["min_rv_iv_ratio"]:
                R.append(f"options_expensive:rv/iv={v:.2f}")
            elif v is not None:
                ok_codes.append("options_cheap_vs_forecast")
        if "max_iv_pctile" in F:
            v = self._need(f, "iv_pctile", R)
            if v is not None and v > F["max_iv_pctile"]:
                R.append(f"iv_pctile_high:{v:.0f}")
        if "allowed_regimes" in F:
            reg = (ctx.regime or {}).get("regime")
            if reg not in F["allowed_regimes"]:
                R.append(f"regime_not_allowed:{reg}")
        if "allowed_hmm" in F:
            h = self._need(f, "hmm_label", R)
            if h is not None:
                allowed = F["allowed_hmm"].get(direction, []) if isinstance(F["allowed_hmm"], dict) else F["allowed_hmm"]
                if h not in allowed:
                    R.append(f"hmm_not_allowed:{h}")
                else:
                    ok_codes.append(f"hmm:{h}")
        if "min_ml_prob" in F:
            v = self._need(f, "ml_prob", R)
            if v is not None and v < F["min_ml_prob"]:
                R.append(f"ml_prob_low:{v:.2f}")
            elif v is not None:
                ok_codes.append(f"ml_prob:{v:.2f}")
        return (not R), R, ok_codes

    def evaluate(self, ctx: MarketContext) -> Optional[Signal]:
        sig = self.base.evaluate(ctx)
        self.last_reasons = list(self.base.last_reasons)
        if sig is None:
            return None
        if self.ml_scorer is not None and "min_ml_prob" in self.filters:
            try:
                ctx.features = {**(ctx.features or {}), "ml_prob": self.ml_scorer(sig, ctx.features or {})}
            except Exception as e:  # noqa: BLE001
                self.last_reasons = ["base_signal", f"ml_scorer_error:{e}"]
                return None
        ok, R, codes = self.check(ctx, sig.direction)
        if not ok:
            self.last_reasons = ["base_signal"] + R
            return None
        sig.strategy_id, sig.model_version = self.strategy_id, self.version
        sig.reason_codes = sig.reason_codes + codes
        sig.diagnostics = {**sig.diagnostics, "filter_features": {k: ctx.features.get(k) for k in (
            "hv_ratio", "ewma_fc", "garch_fc", "vix_prev", "rv_iv_ratio", "iv_pctile", "hmm_label", "ml_prob", "events")}}
        return sig


def contract_iv_ok(option_iv_pct: Optional[float], features: dict, filters: dict) -> tuple[bool, str]:
    """Reject buying an option whose IV is too rich versus the volatility forecast."""
    lim = (filters or {}).get("contract_iv_max_over_rv")
    if not lim:
        return True, ""
    fc = (features or {}).get(filters.get("rv_forecast_feature", "garch_fc"))
    if option_iv_pct is None or fc is None:
        return False, "contract_iv_check_missing_data"
    ratio = option_iv_pct / fc
    return (ratio <= lim), f"contract_iv/rv={ratio:.2f}"
