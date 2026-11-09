import copy
import json
from datetime import date

from core.config import load_config
from governance.registry import config_hash
from ledger.store import Ledger
from monitoring.live_gate import live_gate
from risk.kill_switch import KillSwitch


def setup(tmp_path, approve=True, until="2026-11-20", smoke="PASS: place -> status -> cancel works"):
    cfg = copy.deepcopy(load_config())
    cfg["settings"]["mode"] = "live"
    cfg["settings"]["paths"].update(ledger_db=str(tmp_path / "l.sqlite3"), live_approval_file=str(tmp_path / "appr.json"),
                                    kill_switch_file=str(tmp_path / "K"))
    sid = cfg["strategies"]["active_strategy"]
    (tmp_path / "governance.json").write_text(json.dumps({"strategies": {sid: [{"status": "APPROVED_FOR_LIVE",
                                                          "config_hash": config_hash(cfg["strategies"]["strategies"][sid])}]}}))
    if approve:
        (tmp_path / "appr.json").write_text(json.dumps({"date": "2026-11-11", "valid_until": until,
                                                        "config_hash": config_hash(cfg["strategies"]["strategies"][sid])}))
    (tmp_path / "live_smoke_test.json").write_text(json.dumps({"steps": [{"step": "result", "verdict": smoke}]}))
    return cfg


def gate(cfg, tmp_path, d=date(2026, 11, 13), funds=30000.0, token_ok=True):
    def tok():
        if not token_ok:
            raise RuntimeError("expired")
        return "t"
    return live_gate(cfg, d, token_fn=tok, funds_fn=lambda: funds, ledger=Ledger(tmp_path / "l.sqlite3"),
                     kill=KillSwitch(tmp_path / "K"))


def test_gate_passes_with_standing_approval(tmp_path):
    ok, p = gate(setup(tmp_path), tmp_path)
    assert ok, p


def test_gate_blocks_each_problem(tmp_path):
    cfg = setup(tmp_path)
    assert not gate(cfg, tmp_path, d=date(2026, 11, 23))[0]                 # approval expired
    assert any("funds" in x for x in gate(cfg, tmp_path, funds=600)[1])     # ₹600 balance
    assert any("login" in x for x in gate(cfg, tmp_path, token_ok=False)[1])
    KillSwitch(tmp_path / "K").activate("t")
    assert any("kill switch" in x for x in gate(cfg, tmp_path)[1])


def test_gate_blocks_failed_smoke_and_config_change(tmp_path):
    cfg = setup(tmp_path, smoke="ORDER REJECTED")
    assert any("smoke" in x for x in gate(cfg, tmp_path)[1])
    cfg = setup(tmp_path)
    cfg["strategies"]["strategies"][cfg["strategies"]["active_strategy"]]["params"]["or_minutes"] = 45
    assert not gate(cfg, tmp_path)[0]
