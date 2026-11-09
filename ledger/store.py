"""SQLite trade ledger — the single source of truth for orders, trades and P&L.

Event tables (``order_events``, ``trade_events``, ``risk_decisions``,
``signals``) are append-only. ``orders`` and ``trades`` are projections that
are updated as events arrive. The dashboard reads only from here.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from core.clock import IST

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
  signal_id TEXT PRIMARY KEY, ts TEXT, trading_day TEXT, strategy_id TEXT, model_version TEXT,
  direction TEXT, mode TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS risk_decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, trading_day TEXT, signal_id TEXT, approved INTEGER, payload TEXT);
CREATE TABLE IF NOT EXISTS orders (
  order_id TEXT PRIMARY KEY, trade_id TEXT, signal_id TEXT, role TEXT, mode TEXT, trading_symbol TEXT,
  side TEXT, qty INTEGER, order_type TEXT, price REAL, trigger_price REAL, state TEXT,
  broker_order_id TEXT, filled_qty INTEGER DEFAULT 0, avg_price REAL, created_at TEXT, updated_at TEXT,
  trading_day TEXT, last_error TEXT);
CREATE TABLE IF NOT EXISTS order_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT, ts TEXT, from_state TEXT, to_state TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS trades (
  trade_id TEXT PRIMARY KEY, signal_id TEXT, strategy_id TEXT, model_version TEXT, mode TEXT,
  trading_day TEXT, trading_symbol TEXT, option_type TEXT, strike REAL, expiry TEXT, qty INTEGER,
  status TEXT, entry_time TEXT, entry_price REAL, expected_entry REAL, exit_time TEXT, exit_price REAL,
  expected_exit REAL, planned_stop REAL, target REAL, planned_risk REAL, realized_risk REAL,
  charges REAL, slippage REAL, gross_pnl REAL, net_pnl REAL, exit_reason TEXT, regime TEXT, details TEXT);
CREATE TABLE IF NOT EXISTS trade_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, trade_id TEXT, ts TEXT, event TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS system_status (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS errors (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, source TEXT, message TEXT);
"""


def _now() -> str:
    return datetime.now(IST).isoformat()


def _j(x: Any) -> str:
    return json.dumps(x, default=str)


class Ledger:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, args)

    def query(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    # ------------------------------------------------------------ signals/risk
    def record_signal(self, sig: dict, mode: str, trading_day: date) -> None:
        self._exec("INSERT OR IGNORE INTO signals VALUES (?,?,?,?,?,?,?,?)",
                   (sig["signal_id"], sig["timestamp"], trading_day.isoformat(), sig["strategy_id"],
                    sig["model_version"], sig["direction"], mode, _j(sig)))

    def record_risk_decision(self, signal_id: str, approved: bool, payload: dict, trading_day: date) -> None:
        self._exec("INSERT INTO risk_decisions (ts, trading_day, signal_id, approved, payload) VALUES (?,?,?,?,?)",
                   (_now(), trading_day.isoformat(), signal_id, int(approved), _j(payload)))

    # ------------------------------------------------------------------ orders
    def insert_order(self, o: dict) -> None:
        now = _now()
        self._exec("""INSERT INTO orders (order_id, trade_id, signal_id, role, mode, trading_symbol, side, qty,
                      order_type, price, trigger_price, state, created_at, updated_at, trading_day)
                      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                   (o["order_id"], o.get("trade_id"), o.get("signal_id"), o["role"], o["mode"], o["trading_symbol"],
                    o["side"], o["qty"], o["order_type"], o.get("price"), o.get("trigger_price"), o["state"], now, now,
                    o["trading_day"]))
        self._exec("INSERT INTO order_events (order_id, ts, from_state, to_state, detail) VALUES (?,?,?,?,?)",
                   (o["order_id"], now, None, o["state"], _j({"created": o})))

    def get_order(self, order_id: str) -> Optional[dict]:
        r = self.query("SELECT * FROM orders WHERE order_id=?", (order_id,))
        return r[0] if r else None

    def update_order(self, order_id: str, new_state: str, detail: dict | None = None, **fields) -> None:
        with self._lock:
            cur = self.get_order(order_id)
            sets = ", ".join(f"{k}=?" for k in fields)
            args = tuple(fields.values())
            self._exec(f"UPDATE orders SET state=?, updated_at=?{', ' + sets if sets else ''} WHERE order_id=?",
                       (new_state, _now(), *args, order_id))
            self._exec("INSERT INTO order_events (order_id, ts, from_state, to_state, detail) VALUES (?,?,?,?,?)",
                       (order_id, _now(), cur["state"] if cur else None, new_state, _j(detail or fields)))

    def orders_for_day(self, d: date) -> list[dict]:
        return self.query("SELECT * FROM orders WHERE trading_day=? ORDER BY created_at", (d.isoformat(),))

    def working_orders(self) -> list[dict]:
        return self.query("SELECT * FROM orders WHERE state IN ('CREATED','VALIDATED','SUBMITTED','ACKNOWLEDGED',"
                          "'OPEN','PARTIALLY_FILLED','UNKNOWN','RECONCILIATION_REQUIRED')")

    # ------------------------------------------------------------------ trades
    def insert_trade(self, t: dict) -> None:
        cols = list(t.keys())
        self._exec(f"INSERT INTO trades ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                   tuple(_j(v) if isinstance(v, (dict, list)) else v for v in t.values()))
        self.trade_event(t["trade_id"], "CREATED", {"status": t.get("status")})

    def update_trade(self, trade_id: str, event: str, **fields) -> None:
        if fields:
            sets = ", ".join(f"{k}=?" for k in fields)
            self._exec(f"UPDATE trades SET {sets} WHERE trade_id=?",
                       tuple(_j(v) if isinstance(v, (dict, list)) else v for v in fields.values()) + (trade_id,))
        self.trade_event(trade_id, event, fields)

    def trade_event(self, trade_id: str, event: str, detail: dict) -> None:
        self._exec("INSERT INTO trade_events (trade_id, ts, event, detail) VALUES (?,?,?,?)",
                   (trade_id, _now(), event, _j(detail)))

    def get_trade(self, trade_id: str) -> Optional[dict]:
        r = self.query("SELECT * FROM trades WHERE trade_id=?", (trade_id,))
        return r[0] if r else None

    def open_trades(self) -> list[dict]:
        return self.query("SELECT * FROM trades WHERE status IN ('PENDING_ENTRY','OPEN','EXIT_PENDING',"
                          "'RECONCILIATION_REQUIRED') ORDER BY entry_time")

    def trades(self, mode: Optional[str] = None) -> list[dict]:
        if mode:
            return self.query("SELECT * FROM trades WHERE mode=? ORDER BY trading_day, entry_time", (mode,))
        return self.query("SELECT * FROM trades ORDER BY trading_day, entry_time")

    def day_stats(self, d: date, mode: str) -> dict:
        rows = self.query("SELECT status, net_pnl FROM trades WHERE trading_day=? AND mode=?", (d.isoformat(), mode))
        entered = [r for r in rows if r["status"] in ("OPEN", "EXIT_PENDING", "CLOSED", "RECONCILIATION_REQUIRED",
                                                      "PENDING_ENTRY")]
        return {"trades_today": len(entered),
                "realized_pnl_today": sum((r["net_pnl"] or 0) for r in rows if r["status"] == "CLOSED"),
                "open_positions": sum(1 for r in rows if r["status"] in ("OPEN", "EXIT_PENDING")),
                "reconciliation_required": any(r["status"] == "RECONCILIATION_REQUIRED" for r in rows)}

    # ------------------------------------------------------------------ status
    def set_status(self, key: str, value: Any) -> None:
        self._exec("INSERT INTO system_status VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,"
                   " updated_at=excluded.updated_at", (key, _j(value), _now()))

    def get_status(self) -> dict:
        return {r["key"]: {"value": json.loads(r["value"]), "updated_at": r["updated_at"]}
                for r in self.query("SELECT * FROM system_status")}

    def record_error(self, source: str, message: str) -> None:
        self._exec("INSERT INTO errors (ts, source, message) VALUES (?,?,?)", (_now(), source, message[:2000]))

    def recent_errors(self, n: int = 20) -> list[dict]:
        return self.query("SELECT * FROM errors ORDER BY id DESC LIMIT ?", (n,))
