"""Dashboard API (FastAPI). All P&L comes from the ledger, never from frontend state.

Run:  python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import io
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from backtesting.reports.metrics import compute_metrics, equity_curve
from core.clock import IST
from core.config import PROJECT_ROOT, load_config, resolve_path
from core.secrets import get_secret, jwt_expiry
from ledger.store import Ledger
from risk.kill_switch import KillSwitch

CFG = load_config()
app = FastAPI(title="Nifty Quant System", version="0.1.0")
LEDGER = Ledger(resolve_path(CFG, "ledger_db"))
KILL = KillSwitch(resolve_path(CFG, "kill_switch_file"))
STATIC = PROJECT_ROOT / "apps" / "dashboard" / "index.html"          # legacy fallback
WEB_DIST = PROJECT_ROOT / "apps" / "web" / "dist"                       # React build (cd apps/web && npm run build)
if (WEB_DIST / "assets").exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")


def _mode(mode: str | None) -> str:
    return mode or ("live" if CFG["settings"]["mode"] == "live" else "paper")


def _closed(mode: str) -> pd.DataFrame:
    df = pd.DataFrame(LEDGER.trades(mode))
    if df.empty:
        return df
    return df[df.status == "CLOSED"].copy()


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


@app.get("/")
def index():
    return FileResponse(WEB_DIST / "index.html" if (WEB_DIST / "index.html").exists() else STATIC)


@app.get("/api/summary")
def summary(mode: str | None = None):
    mode = _mode(mode)
    capital = float(CFG["risk"]["capital"])
    closed = _closed(mode)
    today = datetime.now(IST).date().isoformat()
    m = compute_metrics(closed, capital) if len(closed) else {"trades": 0}
    realized = float(closed.net_pnl.sum()) if len(closed) else 0.0
    today_pnl = float(closed[closed.trading_day == today].net_pnl.sum()) if len(closed) else 0.0
    st = LEDGER.get_status()
    pos = (st.get("position") or {}).get("value")
    open_outlay = (pos or {}).get("entry", 0) * (pos or {}).get("qty", 0) if pos else 0
    return _clean({
        "mode": mode, "config_mode": CFG["settings"]["mode"], "capital": capital,
        "available_balance": capital + realized - open_outlay, "current_pnl": (pos or {}).get("mtm", 0.0),
        "today_pnl": today_pnl, "total_realized_pnl": realized, "total_trades": m.get("trades", 0),
        "win_rate_pct": m.get("win_rate_pct"), "profit_factor": m.get("profit_factor"), "roi_pct": m.get("roi_pct"),
        "max_drawdown": m.get("max_drawdown"), "position": pos, "kill_switch": KILL.info(),
        "health": health()["overall"]})


_BROKER_CACHE: dict = {"at": 0.0, "data": None}


@app.get("/api/broker")
def broker_account():
    """REAL Groww account funds (read-only, cached 60 s). Independent of paper capital."""
    if _BROKER_CACHE["data"] and time.time() - _BROKER_CACHE["at"] < 60:
        return _BROKER_CACHE["data"]
    try:
        from ingestion.providers.groww import GrowwProvider

        p = GrowwProvider(CFG["settings"])
        r = p._call("non_trading", p.client.get_available_margin_details)
        fno = r.get("fno_margin_details") or {}
        data = {"ok": True, "option_buy_balance": fno.get("option_buy_balance_available"),
                "clear_cash": r.get("clear_cash"), "net_margin_used": r.get("net_margin_used"),
                "fetched_at": datetime.now(IST).isoformat()}
    except Exception as e:  # noqa: BLE001
        data = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300], "fetched_at": datetime.now(IST).isoformat()}
    _BROKER_CACHE.update(at=time.time(), data=data)
    return data


@app.get("/api/pnl")
def pnl(mode: str | None = None):
    mode = _mode(mode)
    capital = float(CFG["risk"]["capital"])
    c = _closed(mode)
    if c.empty:
        return {"daily": [], "weekly": [], "monthly": [], "yearly": [], "equity": [], "by_option_type": {},
                "by_strategy": {}}
    c["d"] = pd.to_datetime(c.trading_day)
    grp = lambda f: [{"period": str(k), "net_pnl": round(float(v), 2)} for k, v in  # noqa: E731
                     c.groupby(c.d.dt.to_period(f)).net_pnl.sum().items()]
    eq = equity_curve(c, capital)
    by = lambda col: {str(k): {"trades": int(len(g)), "net_pnl": round(float(g.net_pnl.sum()), 2),  # noqa: E731
                               "win_rate": round(float((g.net_pnl > 0).mean() * 100), 1)} for k, g in c.groupby(col)}
    return _clean({"daily": grp("D"), "weekly": grp("W"), "monthly": grp("M"), "yearly": grp("Y"),
                   "equity": eq.to_dict("records"), "by_option_type": by("option_type"),
                   "by_strategy": by("strategy_id")})


@app.get("/api/market")
def market():
    st = LEDGER.get_status()
    g = lambda k: (st.get(k) or {}).get("value")  # noqa: E731
    return _clean({"market": g("market"), "position": g("position"), "last_decision": g("last_decision"),
                   "updated_at": (st.get("market") or {}).get("updated_at")})


@app.get("/api/trades")
def trades(mode: str | None = None):
    rows = LEDGER.trades(_mode(mode))
    for r in rows:
        if r.get("net_pnl") is not None and r.get("entry_price"):
            r["roi_pct"] = round(r["net_pnl"] / (r["entry_price"] * r["qty"]) * 100, 2)
        r.pop("details", None)
    return _clean(rows[::-1])


@app.get("/api/orders")
def orders(day: str | None = None):
    if day:
        return LEDGER.query("SELECT * FROM orders WHERE trading_day=? ORDER BY created_at DESC", (day,))
    return LEDGER.query("SELECT * FROM orders ORDER BY created_at DESC LIMIT 200")


@app.get("/api/decisions")
def decisions(limit: int = 50):
    rows = LEDGER.query("SELECT * FROM risk_decisions ORDER BY id DESC LIMIT ?", (limit,))
    for r in rows:
        r["payload"] = json.loads(r["payload"])
    return rows


@app.get("/api/export/trades.csv")
def export_trades(mode: str | None = None):
    df = pd.DataFrame(LEDGER.trades(_mode(mode)))
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=trades.csv"})


@app.get("/api/health")
def health():
    st = LEDGER.get_status()
    now = datetime.now(IST)
    hb = st.get("heartbeat")
    hb_age = None
    if hb:
        hb_age = (now - datetime.fromisoformat(hb["updated_at"])).total_seconds()
    tok = get_secret("GROWW_ACCESS_TOKEN")
    exp = jwt_expiry(tok) if tok else None
    try:
        LEDGER.query("SELECT 1")
        db_ok = True
    except Exception:  # noqa: BLE001
        db_ok = False
    mk = st.get("market")
    recon = LEDGER.query("SELECT COUNT(*) n FROM trades WHERE status='RECONCILIATION_REQUIRED'")[0]["n"]
    manifests = resolve_path(CFG, "data_lake") / "manifests"
    ingest = {}
    for f in manifests.glob("*.jsonl"):
        lines = f.read_text().splitlines()
        if lines:
            last = json.loads(lines[-1])
            ingest[f.stem] = {"entries": len(lines), "last": last.get("recorded_at"), "last_status": last.get("status")}
    checks = {
        "database": db_ok,
        "worker_heartbeat_age_s": hb_age,
        "worker_alive": hb_age is not None and hb_age < 60,
        "worker": (st.get("worker") or {}).get("value"),
        "last_market_data": (mk or {}).get("updated_at"),
        "broker_token_expires": datetime.fromtimestamp(exp, IST).isoformat() if exp else None,
        "broker_token_valid": bool(exp and exp > time.time()) or bool(get_secret("GROWW_TOTP_SECRET")),
        "reconciliation_required": recon,
        "kill_switch": KILL.info(),
        "historical_ingestion": ingest,
        "websocket": "not used (REST polling)",
        "latest_errors": LEDGER.recent_errors(15),
    }
    overall = "ok" if db_ok and checks["broker_token_valid"] and not recon and not KILL.active() else "attention"
    return _clean({"overall": overall, "checks": checks})


@app.post("/api/kill")
def kill_on(reason: str = Body("dashboard", embed=True)):
    KILL.activate(reason, "dashboard")
    return {"kill_switch": KILL.info()}


@app.post("/api/kill/off")
def kill_off(confirm: str = Body(..., embed=True)):
    if confirm != "RESUME":
        raise HTTPException(400, "send {\"confirm\": \"RESUME\"}")
    KILL.deactivate()
    return {"kill_switch": None}


# ------------------------------------------------------------------ backtests
def _bt_root() -> Path:
    return resolve_path(CFG, "backtests")


@app.get("/api/backtests")
def backtests():
    out = []
    for d in sorted(_bt_root().glob("*/summary.json"), reverse=True):
        s = json.loads(d.read_text())
        m, rob = s.get("metrics", {}), s.get("robustness") or {}
        out.append({"run_id": s["run_id"], "experiment": s["config"].get("experiment"),
                    "scenario": s["config"].get("scenario"), "start": s["config"].get("start"),
                    "end": s["config"].get("end"), **{k: m.get(k) for k in (
                        "trades", "net_pnl", "roi_pct", "win_rate_pct", "profit_factor", "expectancy",
                        "max_drawdown", "sharpe", "charges")},
                    "oos_net": (rob.get("out_of_sample") or {}).get("net_pnl"),
                    "is_net": (rob.get("in_sample") or {}).get("net_pnl")})
    return _clean(out)


@app.get("/api/backtests/{run_id}")
def backtest(run_id: str):
    d = _bt_root() / Path(run_id).name
    if not (d / "summary.json").exists():
        raise HTTPException(404)
    s = json.loads((d / "summary.json").read_text())
    s["trades"] = pd.read_csv(d / "trades.csv").to_dict("records") if (d / "trades.csv").stat().st_size > 1 else []
    s["equity"] = pd.read_csv(d / "equity.csv").to_dict("records") if (d / "equity.csv").stat().st_size > 1 else []
    return _clean(s)


@app.get("/api/backtests/{run_id}/trades.csv")
def backtest_csv(run_id: str):
    p = _bt_root() / Path(run_id).name / "trades.csv"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, filename=f"{run_id}-trades.csv")


# ------------------------------------------------------------------ research
@app.get("/api/research")
def research():
    from ingestion.historical.datalake import DataLake
    from quant.volatility import ewma, garch, hv

    lake = DataLake(resolve_path(CFG, "data_lake"))
    df = lake.read_silver("nifty", CFG["instruments"]["index_groww_symbol"], "5minute")
    if df.empty:
        return {"note": "no NIFTY data in the lake yet; run a backtest or scripts/download_history.py"}
    daily = df.groupby(df.timestamp.dt.date).close.last()
    daily.index = pd.to_datetime(daily.index)
    out: dict = {"days": len(daily), "last_date": str(daily.index[-1].date())}
    t = hv.hv_table(daily)
    out["hv"] = {c: (None if pd.isna(t[c].iloc[-1]) else round(float(t[c].iloc[-1]), 2)) for c in t}
    out["hv_series"] = [{"date": str(i.date()), **{c: (None if pd.isna(v) else round(float(v), 2)) for c, v in r.items()}}
                        for i, r in t.tail(120).iterrows()]
    r = hv.log_returns(daily).dropna()
    if len(r) > 25:
        ev = ewma.ewma_vol(r, 0.94).dropna()
        out["ewma_vol_next"] = round(float(ev.iloc[-1]), 2) if len(ev) else None
    rv = hv.intraday_realized_vol(df, 5).tail(60)
    out["intraday_rv"] = [{"date": str(k), "rv": round(float(v), 2)} for k, v in rv.items()]
    if len(r) >= 120:
        fits = {}
        for m in ("garch", "gjr", "egarch"):
            f = garch.fit(r, m)
            fits[m] = {"params": {k: round(v, 5) for k, v in f.params.items()}, "aic": round(f.aic, 2),
                       "converged": f.converged, "persistence": round(f.persistence(), 4),
                       "forecast_5d_vol": round(garch.forecast_annual_vol(f, 5), 2)}
        out["garch"] = fits
        out["ewma_lambda_oos"] = ewma.compare_lambdas(r).to_dict("records")
    else:
        out["garch"] = f"need >=120 daily returns, have {len(r)} (download more history)"
    out["regime"] = ((LEDGER.get_status().get("market") or {}).get("value") or {}).get("regime")
    return _clean(out)


@app.get("/api/research/chain")
def research_chain(expiry: str = Query(..., description="YYYY-MM-DD")):
    """Live IV smile/skew (needs a valid token)."""
    from datetime import date as _d

    from ingestion.providers.groww import GrowwProvider
    from quant.options.surface import atm_iv, chain_iv_table, greeks_comparison, skew_25d_proxy, smile

    p = GrowwProvider(CFG["settings"])
    ch = p.get_option_chain(CFG["instruments"]["underlying"], _d.fromisoformat(expiry))
    now = datetime.now(IST)
    tbl = chain_iv_table(ch, now, CFG["settings"]["backtest"]["risk_free_rate"])
    spot = ch.underlying_ltp
    near = tbl[(tbl.strike - spot).abs() <= spot * 0.05]
    return _clean({"spot": spot, "expiry": expiry, "atm_iv_model": atm_iv(tbl, spot),
                   "atm_iv_provider": atm_iv(tbl, spot, col="provider_iv"), "skew_2pct": skew_25d_proxy(tbl, spot),
                   "model_vs_provider": greeks_comparison(near),
                   "smile": smile(near, spot).to_dict("records"), "table": near.to_dict("records")})


# ====================================================================== additions (phase 2)
import threading  # noqa: E402
import uuid as _uuid  # noqa: E402

from fastapi.responses import PlainTextResponse  # noqa: E402

JOBS: dict[str, dict] = {}


def _gold(*parts) -> Path:
    p = resolve_path(CFG, "data_lake") / "gold"
    for x in parts:
        p = p / x
    return p


@app.get("/api/strategies")
def strategies():
    from governance.registry import registry_table

    return _clean(registry_table(load_config()))


@app.post("/api/backtests/run")
def run_backtest(body: dict = Body(...)):
    """Launch a backtest in a background thread. Body: strategy_id, start, end, scenario, capital,
    risk_per_trade_max, reward_risk, itm_steps, avoid_expiry_day, charges_multiplier."""
    jid = _uuid.uuid4().hex[:8]
    JOBS[jid] = {"status": "running", "started": datetime.now(IST).isoformat(), "request": body}

    def work():
        try:
            from datetime import date as _d

            from backtesting.engine.orb_runner import run_orb_backtest
            from ingestion.providers.groww import GrowwProvider

            cfg = load_config()
            num = lambda k, f=float: f(body[k]) if body.get(k) not in (None, "") else None  # noqa: E731
            res = run_orb_backtest(cfg, GrowwProvider(cfg["settings"]), _d.fromisoformat(body["start"]),
                                   _d.fromisoformat(body["end"]), body.get("scenario", "base"), body.get("strategy_id"),
                                   capital=num("capital"), risk=num("risk_per_trade_max"),
                                   square_off=body.get("square_off") or None, itm_steps=num("itm_steps", int),
                                   charges_multiplier=num("charges_multiplier") or 1.0)
            JOBS[jid].update(status="done", run_id=res.run_id, metrics={k: res.metrics.get(k) for k in (
                "trades", "net_pnl", "profit_factor", "win_rate_pct")})
        except Exception as e:  # noqa: BLE001
            JOBS[jid].update(status="failed", error=f"{type(e).__name__}: {e}"[:500])

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": jid}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid not in JOBS:
        raise HTTPException(404)
    return _clean(JOBS[jid])


@app.get("/api/backtests_compare")
def compare_backtests(ids: str):
    out = []
    for rid in ids.split(","):
        d = _bt_root() / Path(rid).name
        if not (d / "summary.json").exists():
            continue
        s = json.loads((d / "summary.json").read_text())
        eq = pd.read_csv(d / "equity.csv").to_dict("records") if (d / "equity.csv").stat().st_size > 1 else []
        rob = s.get("robustness") or {}
        out.append({"run_id": rid, "experiment": s["config"].get("experiment"), "scenario": s["config"].get("scenario"),
                    "metrics": {k: s["metrics"].get(k) for k in ("trades", "net_pnl", "roi_pct", "win_rate_pct",
                                                                "profit_factor", "expectancy", "max_drawdown", "sharpe",
                                                                "sortino", "charges")},
                    "in_sample": {k: (rob.get("in_sample") or {}).get(k) for k in ("trades", "net_pnl", "profit_factor")},
                    "out_of_sample": {k: (rob.get("out_of_sample") or {}).get(k) for k in ("trades", "net_pnl", "profit_factor")},
                    "equity": eq})
    return _clean(out)


@app.get("/api/research/models")
def research_models():
    """Next-session model forecasts, HMM regime, RV vs IV series, model evaluation (from gold tables)."""
    fp = _gold("volatility", "daily_features.parquet")
    if not fp.exists():
        return {"note": "run python scripts/build_gold.py features"}
    f = pd.read_parquet(fp)
    last = f.iloc[-1].to_dict()
    cols = [c for c in ("hv20", "ewma_fc", "garch_fc", "gjr_fc", "atm_iv_prev", "vix_prev") if c in f]
    ser = f[["date", *cols]].tail(250).copy()
    ser["date"] = ser["date"].astype(str)
    out = {"next_session": str(last["date"])[:10],
           "forecasts": {k: last.get(k) for k in ("hv5", "hv10", "hv20", "hv30", "hv60", "ewma_fc", "garch_fc",
                                                  "gjr_fc", "vix_prev", "atm_iv_prev", "iv_pctile", "rv_iv_ratio")},
           "hmm": {"label": last.get("hmm_label"), "probs": [last.get(f"hmm_p{i}") for i in range(3)]},
           "events_next_session": last.get("events"),
           "series": ser.to_dict("records")}
    if "hmm_label" in f:
        hl = f.dropna(subset=["hmm_label"]).tail(500)
        out["hmm_label_counts_500d"] = hl["hmm_label"].value_counts().to_dict()
    for name, p in (("vol_model_eval", _gold("volatility", "model_eval.parquet")),
                    ("ml_metrics", _gold("ml", "metrics_breakout_vwap_v1.parquet"))):
        if p.exists():
            out[name] = pd.read_parquet(p).to_dict("records")
    ivp = _gold("implied_volatility", "atm_iv_daily.parquet")
    if ivp.exists():
        iv = pd.read_parquet(ivp)
        from quant.options.surface import iv_percentile, iv_rank

        s = iv.set_index("date")["iv_atm_close"].dropna()
        out["iv_rank_252"], out["iv_percentile_252"] = iv_rank(s), iv_percentile(s)
    return _clean(out)


@app.get("/api/research/term_structure")
def term_structure(n: int = 4):
    """Live ATM IV per upcoming expiry (needs a valid token)."""
    from datetime import date as _d

    from ingestion.instruments.calendar import ExpiryCalendar
    from ingestion.providers.groww import GrowwProvider
    from quant.options.surface import atm_iv, chain_iv_table

    p = GrowwProvider(CFG["settings"])
    today = datetime.now(IST).date()
    cal = ExpiryCalendar(p, "NIFTY")
    exps = [e for e in cal.around(today) + cal.around((today.replace(day=28) + timedelta(days=7))) if e >= today]
    rows = []
    for e in sorted(set(exps))[:n]:
        ch = p.get_option_chain("NIFTY", e)
        tbl = chain_iv_table(ch, datetime.now(IST), CFG["settings"]["backtest"]["risk_free_rate"])
        rows.append({"expiry": e.isoformat(), "days": (e - today).days, "atm_iv_model": atm_iv(tbl, ch.underlying_ltp),
                     "atm_iv_provider": atm_iv(tbl, ch.underlying_ltp, col="provider_iv")})
    return _clean(rows)


@app.get("/api/alerts")
def alerts(n: int = 50):
    try:
        return LEDGER.query("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (n,))
    except Exception:  # noqa: BLE001
        return []


@app.get("/api/paper_vs_backtest")
def paper_vs_backtest():
    from monitoring.paper_vs_backtest import compare

    return _clean(compare(LEDGER.trades("paper"), CFG["settings"]["backtest"]["scenarios"], CFG["instruments"]["tick_size"]))


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus():
    from monitoring.metrics import render_prometheus

    st = LEDGER.get_status()
    hb = st.get("heartbeat")
    age = (datetime.now(IST) - datetime.fromisoformat(hb["updated_at"])).total_seconds() if hb else None
    c = _closed("paper")
    gauges = {"worker_heartbeat_age_seconds": age, "kill_switch_active": int(KILL.active()),
              "open_positions": len(LEDGER.open_trades()),
              "working_orders": len(LEDGER.working_orders()),
              "reconciliation_required": LEDGER.query("SELECT COUNT(*) n FROM trades WHERE status='RECONCILIATION_REQUIRED'")[0]["n"],
              "paper_realized_pnl": float(c.net_pnl.sum()) if len(c) else 0.0,
              "errors_total": LEDGER.query("SELECT COUNT(*) n FROM errors")[0]["n"]}
    return render_prometheus((st.get("metrics") or {}).get("value"), gauges)


@app.get("/api/data_quality")
def data_quality():
    from ingestion.historical.datalake import DataLake
    from monitoring.data_quality import completeness

    lake = DataLake(resolve_path(CFG, "data_lake"))
    rep = completeness(lake)
    p = lake.root / "manifests" / "data_report.json"
    if p.exists():
        saved = json.loads(p.read_text())
        rep["instrument_quality"] = saved.get("instrument_quality")
    return _clean(rep)


