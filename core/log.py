"""Structured JSON logging with secret redaction.

Every record passes through ``RedactingFilter`` which masks JWT-like tokens and
any value of known secret keys, so credentials never reach log files.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

_JWT = re.compile(r"eyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}")
_KV = re.compile(r"(?i)(token|secret|password|totp|api_key|authorization)(['\"]?\s*[:=]\s*['\"]?)([^\s'\",}]+)")


def redact(text: str) -> str:
    text = _JWT.sub("<redacted-jwt>", text)
    return _KV.sub(lambda m: f"{m.group(1)}{m.group(2)}<redacted>", text)


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            msg = f"{record.msg} {record.args}"
        record.msg, record.args = redact(msg), None
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {
            "ts": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            d.update(extra)
        if record.exc_info:
            d["exc"] = redact(self.formatException(record.exc_info))
        return redact(json.dumps(d, default=str))


_configured = False


def setup_logging(log_dir: Path | str | None = None, level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger()
    root.setLevel(level)
    f = RedactingFilter()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S"))
    sh.addFilter(f)
    root.addHandler(sh)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{datetime.now():%Y-%m-%d}.jsonl")
        fh.setFormatter(JsonFormatter())
        fh.addFilter(f)
        root.addHandler(fh)
    _configured = True


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, msg + (" " + json.dumps(fields, default=str) if fields else ""), extra={"extra_fields": fields})
