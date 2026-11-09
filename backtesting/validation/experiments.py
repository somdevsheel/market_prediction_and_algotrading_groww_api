"""Experiment definitions (parameter overrides; no strategy code changes).

Only experiments whose filters are implemented are runnable. C-F (vol / IV
filters), H (ML) and I (combined) are placeholders until those filters exist.
"""
EXPERIMENTS = {
    "A_breakout": {"use_vwap_filter": False, "use_volume_filter": False, "use_context_filter": False},
    "B_breakout_vwap": {"use_vwap_filter": True, "use_volume_filter": True, "use_context_filter": False},
    "B2_breakout_vwap_context": {"use_vwap_filter": True, "use_volume_filter": True, "use_context_filter": True},
    "G_regime_filter": {"use_vwap_filter": True, "use_volume_filter": True, "use_context_filter": True,
                        "block_regimes": ["unstable", "range_bound", "volatility_expansion"]},
}
NOT_IMPLEMENTED = ["C_hv_filter", "D_ewma_filter", "E_garch_filter", "F_iv_filter", "H_ml_confirmation", "I_combined"]
