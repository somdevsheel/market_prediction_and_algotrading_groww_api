import json

from fastapi.testclient import TestClient


def test_api_endpoints(tmp_path, monkeypatch):
    import apps.api.main as m
    from ledger.store import Ledger
    from risk.kill_switch import KillSwitch

    monkeypatch.setattr(m, "LEDGER", Ledger(tmp_path / "l.sqlite3"))
    monkeypatch.setattr(m, "KILL", KillSwitch(tmp_path / "KILL"))
    m.LEDGER.insert_trade({"trade_id": "T1", "mode": "paper", "trading_day": "2026-11-11", "trading_symbol": "X",
                           "option_type": "CE", "strategy_id": "s", "qty": 65, "status": "CLOSED", "entry_price": 200.0,
                           "exit_price": 210.0, "entry_time": "2026-11-11T09:50:00+05:30",
                           "exit_time": "2026-11-11T10:20:00+05:30", "gross_pnl": 650.0, "charges": 60.0,
                           "net_pnl": 590.0, "details": {}})
    c = TestClient(m.app)
    assert c.get("/").status_code == 200
    s = c.get("/api/summary?mode=paper").json()
    assert s["total_realized_pnl"] == 590.0 and s["total_trades"] == 1 and s["win_rate_pct"] == 100.0
    p = c.get("/api/pnl?mode=paper").json()
    assert p["equity"][0]["equity"] == 20590.0 and p["by_option_type"]["CE"]["trades"] == 1
    assert c.get("/api/trades?mode=paper").json()[0]["roi_pct"] == round(590 / 13000 * 100, 2)
    assert "T1" in c.get("/api/export/trades.csv?mode=paper").text
    assert c.get("/api/health").status_code == 200
    assert c.get("/api/market").status_code == 200
    assert c.get("/api/backtests").status_code == 200
    c.post("/api/kill", json={"reason": "t"})
    assert m.KILL.active()
    assert c.post("/api/kill/off", json={"confirm": "no"}).status_code == 400
    c.post("/api/kill/off", json={"confirm": "RESUME"})
    assert not m.KILL.active()
