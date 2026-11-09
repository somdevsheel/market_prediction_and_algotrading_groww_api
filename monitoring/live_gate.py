"""Pre-market gate for unattended LIVE trading. Every check must pass or the day runs in paper mode."""
from __future__ import annotations

import json
from datetime import date
from typing import Callable

from core.config import resolve_path


def smoke_test_passed(cfg: dict) -> tuple[bool, str]:
    p = resolve_path(cfg, "ledger_db").parent / "live_smoke_test.json"
    if not p.exists():
        return False, "live smoke test never run (python3 scripts/live_smoke_test.py during market hours)"
    rep = json.loads(p.read_text())
    verdict = next((s.get("verdict", "") for s in rep.get("steps", []) if s.get("step") == "result"), "")
    return (verdict.startswith("PASS"), verdict or "smoke test has no result")


def live_gate(cfg: dict, today: date, *, token_fn: Callable[[], str], funds_fn: Callable[[], float],
              ledger, kill) -> tuple[bool, list[str]]:
    from apps.worker.engine import live_approved
    from governance.registry import effective_status

    problems: list[str] = []
    sid = cfg["strategies"]["active_strategy"]
    ok, why = live_approved(cfg, today, effective_status(cfg, sid))
    if not ok:
        problems.append(f"approval: {why}")
    if kill.active():
        problems.append(f"kill switch active: {(kill.info() or {}).get('reason')}")
    ok, why = smoke_test_passed(cfg)
    if not ok:
        problems.append(f"smoke test: {why}")
    try:
        token_fn()
    except Exception as e:  # noqa: BLE001
        problems.append(f"login: {type(e).__name__}: {e}"[:200])
    else:
        try:
            need = float(cfg["risk"].get("min_live_funds", 15000))
            have = float(funds_fn())
            if have < need:
                problems.append(f"funds: option-buy balance ₹{have:,.0f} < required ₹{need:,.0f}")
        except Exception as e:  # noqa: BLE001
            problems.append(f"funds check failed: {e}"[:200])
    if ledger.query("SELECT COUNT(*) n FROM trades WHERE status='RECONCILIATION_REQUIRED'")[0]["n"]:
        problems.append("a trade requires manual reconciliation")
    return (not problems), problems
