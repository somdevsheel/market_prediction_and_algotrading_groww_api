"""Alerts: persisted to the ledger, logged, and optionally POSTed to a webhook (``ALERT_WEBHOOK_URL`` in env/.env,
Slack/Discord/Teams-compatible ``{"text": ...}``). Messages are redacted; the same code is not re-sent within
``dedupe_minutes``."""
from __future__ import annotations

import logging
import time
from datetime import datetime

from core.clock import IST
from core.log import redact
from core.secrets import get_secret

log = logging.getLogger("alerts")

SCHEMA = """CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, level TEXT, code TEXT,
message TEXT, delivered INTEGER)"""


class AlertManager:
    def __init__(self, ledger, dedupe_minutes: float = 10, webhook_url: str | None = None, post=None):
        self.ledger = ledger
        ledger.conn.execute(SCHEMA)
        self.dedupe = dedupe_minutes * 60
        self.webhook = webhook_url if webhook_url is not None else get_secret("ALERT_WEBHOOK_URL")
        self._last: dict[str, float] = {}
        self._post = post

    def alert(self, level: str, code: str, message: str) -> bool:
        now = time.time()
        if now - self._last.get(code, 0) < self.dedupe:
            return False
        self._last[code] = now
        msg = redact(message)
        delivered = 0
        if self.webhook:
            try:
                import requests

                (self._post or requests.post)(self.webhook, json={"text": f"[{level}] {code}: {msg}"}, timeout=5)
                delivered = 1
            except Exception as e:  # noqa: BLE001
                log.warning("alert webhook failed: %s", type(e).__name__)
        self.ledger.conn.execute("INSERT INTO alerts (ts, level, code, message, delivered) VALUES (?,?,?,?,?)",
                                 (datetime.now(IST).isoformat(), level, code, msg[:2000], delivered))
        log.log(logging.ERROR if level in ("CRITICAL", "ERROR") else logging.WARNING, "ALERT %s %s: %s", level, code, msg)
        return True

    def recent(self, n: int = 50) -> list[dict]:
        return self.ledger.query("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (n,))
