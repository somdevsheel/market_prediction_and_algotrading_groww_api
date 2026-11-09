"""Kill switch backed by a file so it survives restarts and can be flipped from
the dashboard, CLI (``scripts/kill.py``) or by the system itself."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from core.clock import IST


class KillSwitch:
    def __init__(self, path: Path | str):
        self.path = Path(path)

    def active(self) -> bool:
        return self.path.exists()

    def activate(self, reason: str, source: str = "manual") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"reason": reason, "source": source, "at": datetime.now(IST).isoformat()}))

    def deactivate(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def info(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return {"reason": "unreadable kill switch file"}
