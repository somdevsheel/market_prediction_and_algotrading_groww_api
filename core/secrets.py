"""Credential loading for the Groww API.

Order of precedence:
1. ``GROWW_ACCESS_TOKEN`` (env or .env)
2. ``GROWW_API_KEY`` + ``GROWW_TOTP_SECRET``  -> TOTP flow (pyotp)
3. ``GROWW_API_KEY`` + ``GROWW_API_SECRET``   -> key/secret flow (needs daily approval in Groww)

The .env parser accepts ``KEY=VALUE`` lines and the legacy ``Access Token : <jwt>``
format. Tokens are never logged.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path

from core.config import PROJECT_ROOT

_ALIASES = {
    "access token": "GROWW_ACCESS_TOKEN",
    "access_token": "GROWW_ACCESS_TOKEN",
    "api key": "GROWW_API_KEY",
    "api secret": "GROWW_API_SECRET",
    "totp secret": "GROWW_TOTP_SECRET",
}


def parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_ ]*?)\s*[=:]\s*(.+)$", line)
        if not m:
            continue
        key, val = m.group(1).strip(), m.group(2).strip().strip('"').strip("'")
        key = _ALIASES.get(key.lower(), key.upper().replace(" ", "_"))
        if val:
            out[key] = val
    return out


def get_secret(name: str, env_file: Path | None = None) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    return parse_env_file(env_file or PROJECT_ROOT / ".env").get(name)


def jwt_expiry(token: str) -> float | None:
    """Return the ``exp`` claim (epoch seconds) of a JWT without verifying it."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return float(json.loads(base64.urlsafe_b64decode(payload))["exp"])
    except Exception:
        return None


class TokenUnavailable(RuntimeError):
    pass


def obtain_access_token(env_file: Path | None = None, force_generate: bool = False) -> str:
    tok = None if force_generate else get_secret("GROWW_ACCESS_TOKEN", env_file)
    if tok:
        exp = jwt_expiry(tok)
        if exp is None or exp > time.time() + 60:
            return tok
    api_key = get_secret("GROWW_API_KEY", env_file)
    totp_secret = get_secret("GROWW_TOTP_SECRET", env_file)
    api_secret = get_secret("GROWW_API_SECRET", env_file)
    if api_key and (totp_secret or api_secret):
        from growwapi import GrowwAPI

        if totp_secret:
            import pyotp

            return GrowwAPI.get_access_token(api_key=api_key, totp=pyotp.TOTP(totp_secret).now())
        return GrowwAPI.get_access_token(api_key=api_key, secret=api_secret)
    if tok:
        raise TokenUnavailable("GROWW_ACCESS_TOKEN has expired. Paste a fresh token into .env or configure TOTP.")
    raise TokenUnavailable("No Groww credentials found. See .env.example.")
