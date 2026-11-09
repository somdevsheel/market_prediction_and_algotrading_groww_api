"""Research governance: strategy status lifecycle with evidence, config hashing, and live-replacement guard.

Source of truth for status = ``state/governance.json`` (append-only transition history); if a strategy has no
history there, the YAML ``status`` is used. Approval for live stores the strategy *config hash*: if the
params/filters/version change afterwards, the strategy is no longer approved (no silent replacement).
"""
from __future__ import annotations

import getpass
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.clock import IST
from core.config import resolve_path

ORDER = ["RESEARCH", "BACKTESTED", "OUT_OF_SAMPLE_VALIDATED", "PAPER_TRADING", "APPROVED_FOR_LIVE"]
REQUIREMENTS = {
    "OUT_OF_SAMPLE_VALIDATED": {"min_trades": 50},
    "APPROVED_FOR_LIVE": {"min_paper_trades": 20},
}


def config_hash(scfg: dict) -> str:
    keep = {k: scfg.get(k) for k in ("class", "version", "params", "filters", "ml_model", "ml_base")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _path(cfg: dict) -> Path:
    return resolve_path(cfg, "ledger_db").parent / "governance.json"


def load(cfg: dict) -> dict:
    p = _path(cfg)
    return json.loads(p.read_text()) if p.exists() else {"strategies": {}}


def effective_status(cfg: dict, sid: str) -> str:
    scfg = cfg["strategies"]["strategies"][sid]
    hist = load(cfg)["strategies"].get(sid, [])
    if not hist:
        return scfg.get("status", "RESEARCH")
    last = hist[-1]
    if last["status"] == "APPROVED_FOR_LIVE" and last.get("config_hash") != config_hash(scfg):
        return "PAPER_TRADING"  # config changed after approval -> approval void
    return last["status"]


def check_evidence(cfg: dict, sid: str, new_status: str, evidence: dict) -> list[str]:
    """Return a list of unmet requirements (empty = OK)."""
    problems = []
    cur = effective_status(cfg, sid)
    if new_status == "RETIRED":
        return []
    if new_status not in ORDER:
        return [f"unknown status {new_status}"]
    if ORDER.index(new_status) > ORDER.index(cur) + 1:
        problems.append(f"cannot jump from {cur} to {new_status}")
    if new_status == "BACKTESTED" and not evidence.get("backtest_run_id"):
        problems.append("backtest_run_id required")
    if new_status == "OUT_OF_SAMPLE_VALIDATED":
        oos = evidence.get("oos") or {}
        if (oos.get("trades") or 0) < REQUIREMENTS[new_status]["min_trades"]:
            problems.append(f"need >= {REQUIREMENTS[new_status]['min_trades']} OOS trades")
        if not (oos.get("net_pnl_conservative") or 0) > 0:
            problems.append("OOS net P&L under conservative costs must be > 0")
    if new_status == "APPROVED_FOR_LIVE":
        pt = evidence.get("paper") or {}
        if (pt.get("trades") or 0) < REQUIREMENTS[new_status]["min_paper_trades"]:
            problems.append(f"need >= {REQUIREMENTS[new_status]['min_paper_trades']} closed paper trades")
        if not (pt.get("net_pnl") or 0) > 0:
            problems.append("paper net P&L must be > 0")
    return problems


def transition(cfg: dict, sid: str, new_status: str, evidence: dict, note: str = "",
               force_reason: Optional[str] = None) -> dict:
    problems = check_evidence(cfg, sid, new_status, evidence)
    if problems and not force_reason:
        raise ValueError("; ".join(problems))
    g = load(cfg)
    scfg = cfg["strategies"]["strategies"][sid]
    entry = {"status": new_status, "at": datetime.now(IST).isoformat(), "by": getpass.getuser(),
             "version": scfg.get("version"), "config_hash": config_hash(scfg), "evidence": evidence, "note": note,
             "unmet_requirements_overridden": problems if force_reason else [], "force_reason": force_reason}
    g["strategies"].setdefault(sid, []).append(entry)
    p = _path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(g, indent=2, default=str))
    return entry


def registry_table(cfg: dict) -> list[dict]:
    g = load(cfg)["strategies"]
    rows = []
    for sid, scfg in cfg["strategies"]["strategies"].items():
        hist = g.get(sid, [])
        rows.append({"strategy_id": sid, "version": scfg.get("version"), "class": scfg.get("class"),
                     "description": scfg.get("description"), "status": effective_status(cfg, sid),
                     "config_hash": config_hash(scfg), "filters": scfg.get("filters"), "params": scfg.get("params"),
                     "history": hist, "active": sid == cfg["strategies"]["active_strategy"]})
    return rows
