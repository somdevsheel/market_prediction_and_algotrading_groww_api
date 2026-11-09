import logging

from core.log import RedactingFilter, redact
from core.secrets import jwt_expiry, parse_env_file


def test_redacts_jwt_and_keys():
    s = redact("Authorization: Bearer eyJabcdefghij.eyJklmnopqrstu.sigsigsigsig token=abc123 api_key=XYZ")
    assert "eyJabcdefghij" not in s and "abc123" not in s and "XYZ" not in s


def test_filter_keeps_numeric_formatting():
    r = logging.LogRecord("x", logging.INFO, "f", 1, "rows=%d px=%.2f tok=%s", (5, 1.5, "eyJaaaaaaaaaa.bbbbbbbbbb.cccccccccc"), None)
    RedactingFilter().filter(r)
    assert r.getMessage() == "rows=5 px=1.50 tok=<redacted-jwt>"


def test_env_parser_accepts_legacy_format(tmp_path):
    p = tmp_path / ".env"
    p.write_text("Access Token : eyJhbGciOiJFUzI1NiJ9.eyJleHAiOjE3OTQyNzA2MDB9.sig\n# c\nGROWW_API_KEY=k\n")
    env = parse_env_file(p)
    assert env["GROWW_ACCESS_TOKEN"].startswith("eyJ") and env["GROWW_API_KEY"] == "k"
    assert jwt_expiry(env["GROWW_ACCESS_TOKEN"]) == 1794270600
