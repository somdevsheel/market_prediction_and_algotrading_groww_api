"""Tests for strategies A/C, filters, daily features (no look-ahead), ML walk-forward, walk-forward selection,
governance, alerts, metrics, paper depth fills, backup verification, token refresh, paper-vs-backtest."""
import copy
import json
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.clock import IST
from core.config import load_config
from tests import synthetic as syn

CFG = load_config()
D = date(2026, 11, 11)


def _ctx(idx, fut, upto, feats=None, regime=None):
    from strategies.base import MarketContext

    vis = idx.iloc[: upto + 1]
    now = vis["timestamp"].iloc[-1] + pd.Timedelta(minutes=5)
    return MarketContext(now=now.to_pydatetime(), index_5m=vis, futures_5m=fut[fut.timestamp <= vis.timestamp.iloc[-1]],
                         regime=regime or {"regime": "trending_bullish"}, features=feats or {})


def test_filters_block_and_pass():
    from strategies.filters import FilteredStrategy, contract_iv_ok

    base = SimpleNamespace(params={}, last_reasons=[])
    f = FilteredStrategy(base, {"min_garch_fc": 12, "block_events": ["rbi_policy"], "min_ml_prob": 0.5,
                                "allowed_hmm": {"BULLISH": ["hmm_bullish"]}}, "x", "1")
    ok, R, _ = f.check(SimpleNamespace(features={"garch_fc": 15, "events": "", "ml_prob": 0.6, "hmm_label": "hmm_bullish"},
                                       regime={}), "BULLISH")
    assert ok, R
    ok, R, _ = f.check(SimpleNamespace(features={"garch_fc": 10, "events": "rbi_policy", "ml_prob": 0.6,
                                                 "hmm_label": "hmm_bearish"}, regime={}), "BULLISH")
    assert not ok and len(R) == 3
    ok, R, _ = f.check(SimpleNamespace(features={}, regime={}), "BULLISH")  # missing features fail closed
    assert not ok and all(r.startswith("missing_feature") for r in R)
    assert contract_iv_ok(20.0, {"garch_fc": 15.0}, {"contract_iv_max_over_rv": 1.25})[0] is False
    assert contract_iv_ok(17.0, {"garch_fc": 15.0}, {"contract_iv_max_over_rv": 1.25})[0] is True


def test_daily_features_no_lookahead():
    from quant.features.daily_features import build_daily_features

    days = pd.bdate_range("2024-01-01", periods=330).date
    h = syn.history(list(days), seed=4)
    a = build_daily_features(h, hmm_states=2)
    h2 = h.copy()
    last_day = days[-1]
    h2.loc[h2.timestamp.dt.date == last_day, ["open", "high", "low", "close"]] *= 1.1  # shock the final day
    b = build_daily_features(h2, hmm_states=2)
    cols = ["hv20", "ewma_fc", "garch_fc", "hmm_p0"]
    pd.testing.assert_frame_equal(a[cols].iloc[:-1], b[cols].iloc[:-1])  # earlier rows unchanged
    pd.testing.assert_frame_equal(a[cols].iloc[[-1]], b[cols].iloc[[-1]])  # final day's features use data < day
    n = build_daily_features(h, hmm_states=2, next_session=days[-1] + timedelta(days=3))
    assert n["date"].iloc[-1].date() == days[-1] + timedelta(days=3) and pd.notna(n["hv20"].iloc[-1])
    assert pd.notna(n["ewma_fc"].iloc[-1]) and pd.notna(n["garch_fc"].iloc[-1])


def test_ml_walk_forward_is_chronological():
    from quant.machine_learning import pipeline as ml

    rng = np.random.default_rng(0)
    n = 600
    dates = pd.bdate_range("2022-01-03", periods=n)
    X = rng.normal(size=(n, len(ml.FEATURES)))
    y = (X[:, 2] + rng.normal(0, 1, n) > 0).astype(int)  # feature 'extension' is informative
    ds = pd.DataFrame(X, columns=ml.FEATURES)
    ds["date"] = dates.date.astype(str)
    ds["decision_ts"] = [f"{d}T10:00:00+05:30" for d in ds["date"]]
    ds["label"] = y
    P, M = ml.walk_forward(ds, min_train=150, refit_months=3)
    assert set(M.model) >= {"logistic", "random_forest"}
    assert (pd.to_datetime(P.train_end) < pd.to_datetime(P.date)).all()  # train strictly before test
    assert M[M.model == "logistic"].auc.iloc[0] > 0.65


def test_walk_forward_selection_uses_train_only():
    from backtesting.validation.walk_forward import walk_forward

    days = pd.bdate_range("2023-01-02", "2024-12-31").date
    good_then_bad = pd.DataFrame({"trading_day": [str(d) for d in days],
                                  "net_pnl": [100 if d.year == 2023 else -100 for d in days],
                                  "gross_pnl": 0.0, "charges": 0.0, "option_type": "CE"})
    flat = good_then_bad.assign(net_pnl=10.0)
    r = walk_forward({"gtb": good_then_bad, "flat": flat}, list(days), 12, 3, 20000, min_trades=5)
    assert r["folds"][0]["chosen"] == "gtb" and r["folds"][0]["test_net"] < 0  # honest OOS: overfit shows up


def test_governance_flow(tmp_path):
    from governance import registry as g

    cfg = copy.deepcopy(CFG)
    cfg["settings"]["paths"]["ledger_db"] = str(tmp_path / "l.sqlite3")
    sid = "orb_retest30_cheapiv_v1"
    with pytest.raises(ValueError):
        g.transition(cfg, sid, "APPROVED_FOR_LIVE", {})
    g.transition(cfg, sid, "BACKTESTED", {"backtest_run_id": "x"})
    with pytest.raises(ValueError):
        g.transition(cfg, sid, "OUT_OF_SAMPLE_VALIDATED", {"oos": {"trades": 80, "net_pnl_conservative": -5}})
    g.transition(cfg, sid, "OUT_OF_SAMPLE_VALIDATED", {"oos": {"trades": 80, "net_pnl_conservative": 500}})
    g.transition(cfg, sid, "PAPER_TRADING", {})
    g.transition(cfg, sid, "APPROVED_FOR_LIVE", {"paper": {"trades": 25, "net_pnl": 100}})
    assert g.effective_status(cfg, sid) == "APPROVED_FOR_LIVE"
    cfg["strategies"]["strategies"][sid]["params"]["or_minutes"] = 45  # silent change
    assert g.effective_status(cfg, sid) == "PAPER_TRADING"


def test_alerts_dedupe_and_redaction():
    from ledger.store import Ledger
    from monitoring.alerts import AlertManager

    sent = []
    a = AlertManager(Ledger(":memory:"), webhook_url="http://x", post=lambda url, json, timeout: sent.append(json))
    assert a.alert("ERROR", "X", "token=abc123 failed")
    assert not a.alert("ERROR", "X", "again")
    assert len(sent) == 1 and "abc123" not in sent[0]["text"] and len(a.recent()) == 1


def test_metrics_render():
    from monitoring.metrics import Metrics, render_prometheus

    m = Metrics()
    m.inc("groww_live_errors")
    m.observe("groww_live", 0.2)
    txt = render_prometheus(m.snapshot(), {"open_positions": 1})
    assert "nq_groww_live_errors_total 1.0" in txt and 'nq_groww_live_latency_seconds{quantile="0.5"} 0.2' in txt
    assert "nq_open_positions 1.0" in txt


def test_paper_depth_partial_fill():
    from core.models import Quote
    from execution.broker_interface import OrderRequest
    from execution.order_state import OrderState as S
    from execution.paper_adapter import PaperBroker

    now = datetime(2026, 11, 11, 10, tzinfo=IST)
    q = Quote("SYM", 100.2, 100.0, 100.2, raw={"depth": {"sell": [{"price": 100.2, "quantity": 30},
                                                                   {"price": 100.3, "quantity": 20},
                                                                   {"price": 101.0, "quantity": 500}],
                                                          "buy": [{"price": 100.0, "quantity": 500}]}},
              last_trade_time=now, received_at=now)
    pb = PaperBroker(lambda s: q, 50000, 0)
    st = pb.place_order(OrderRequest("NQE0PPPPPPPPPP", "SYM", "BUY", 65, "LIMIT", 100.5))
    assert st.state == S.PARTIALLY_FILLED and st.filled_qty == 50
    assert st.avg_price == pytest.approx((30 * 100.2 + 20 * 100.3) / 50)
    pb.cancel_order("NQE0PPPPPPPPPP", None)
    assert pb.get_order_status("NQE0PPPPPPPPPP").state == S.CANCELLED and pb.get_net_positions() == {"SYM": 50}


def test_token_refresh_on_auth_error(monkeypatch):
    from ingestion.providers import groww as gmod

    class GrowwAPIAuthenticationException(Exception):
        pass

    calls = {"n": 0}

    class C:
        token = "old"

        def get_ltp(self, *a):
            calls["n"] += 1
            if self.token == "old":
                raise GrowwAPIAuthenticationException()
            return {"ok": 1}

    p = gmod.GrowwProvider(CFG["settings"], client=C(), sleep=lambda s: None)
    monkeypatch.setattr(p, "refresh_token", lambda: (setattr(p.client, "token", "new") or True))
    assert p.get_ltp(["NSE_NIFTY"], "CASH") == {"ok": 1} and calls["n"] == 2


def test_backup_verify(tmp_path):
    import importlib.util
    import sqlite3

    spec = importlib.util.spec_from_file_location("backup", "scripts/backup.py")
    import sys

    sys.path.insert(0, "scripts")
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    d = tmp_path / "bk"
    d.mkdir()
    con = sqlite3.connect(d / "ledger.sqlite3")
    con.execute("CREATE TABLE trades (x)")
    con.execute("INSERT INTO trades VALUES (1)")
    con.commit(), con.close()
    (d / "MANIFEST.json").write_text(json.dumps({"row_counts": b.counts(d / "ledger.sqlite3"),
                                                 "checksums": {"ledger.sqlite3": b.sha(d / "ledger.sqlite3")}}))
    assert b.verify(d)["ok"]
    with open(d / "ledger.sqlite3", "ab") as f:
        f.write(b"tamper")
    assert not b.verify(d)["checksums_ok"]


def test_paper_vs_backtest_report():
    from monitoring.paper_vs_backtest import compare

    t = [{"status": "CLOSED", "entry_price": 101.0, "exit_price": 110.0, "expected_exit": 110.2, "charges": 60,
          "net_pnl": 500, "exit_reason": "target", "details": json.dumps({"entry_quote": {"bid": 100.0, "ask": 100.8}})}]
    r = compare(t, CFG["settings"]["backtest"]["scenarios"])
    assert r["avg_entry_cost_vs_mid_pts"] == pytest.approx(0.6) and r["closest_scenario"] in CFG["settings"]["backtest"]["scenarios"]
