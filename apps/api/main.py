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
                    "is_net": (rob.get("in_sample") or {}).get("net_pnl"),
                    "payoff_ratio": round(m["avg_win"] / -m["avg_loss"], 2) if m.get("avg_win") and m.get("avg_loss") else None,
                    "avg_charges_per_trade": round(m["charges"] / m["trades"], 2) if m.get("trades") and m.get("charges") else None,
                    "rr_target": (lambda rr: None if rr is None or rr >= 20 else rr)((s["config"].get("risk") or {}).get("reward_risk"))})
    return _clean(out)


@app.get("/api/backtests/{run_id}")
def backtest(run_id: str):
    d = _bt_root() / Path(run_id).name
    if not (d / "summary.json").exists():
        raise HTTPException(404)
    s = json.loads((d / "summary.json").read_text())
    tdf = pd.read_csv(d / "trades.csv") if (d / "trades.csv").stat().st_size > 1 else pd.DataFrame()
    if len(tdf):  # recompute so older runs also get payoff / R-multiple / per-trade charge metrics
        s["metrics"] = {**s.get("metrics", {}), **compute_metrics(tdf, float(s["config"]["risk"]["capital"]),
                                                                   s["metrics"].get("trading_days_in_sample"))}
    s["trades"] = tdf.to_dict("records")
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
    """Launch backtest(s) in a background thread. Body: strategy_id, start, end, scenario, capital,
    risk_per_trade_max, itm_steps, square_off, charges_multiplier, reward_risk (0/None = no fixed target),
    rr_sweep (list of reward_risk values -> one run each, for side-by-side comparison)."""
    jid = _uuid.uuid4().hex[:8]
    sweep = body.get("rr_sweep") or []
    JOBS[jid] = {"status": "running", "started": datetime.now(IST).isoformat(), "request": body,
                 "progress": f"0/{len(sweep) or 1}"}

    def work():
        try:
            from datetime import date as _d

            from backtesting.engine.orb_runner import run_orb_backtest
            from ingestion.providers.groww import GrowwProvider

            cfg = load_config()
            num = lambda k, f=float: f(body[k]) if body.get(k) not in (None, "") else None  # noqa: E731
            provider = GrowwProvider(cfg["settings"])
            rrs = [float(x) if x not in (None, "", 0, "0", "none") else None for x in sweep] or [num("reward_risk")]
            runs = []
            for i, rr in enumerate(rrs):
                res = run_orb_backtest(cfg, provider, _d.fromisoformat(body["start"]), _d.fromisoformat(body["end"]),
                                       body.get("scenario", "base"), body.get("strategy_id"),
                                       capital=num("capital"), risk=num("risk_per_trade_max"),
                                       square_off=body.get("square_off") or None, itm_steps=num("itm_steps", int),
                                       charges_multiplier=num("charges_multiplier") or 1.0, reward_risk=rr or None)
                runs.append({"run_id": res.run_id, "reward_risk": rr, **{k: res.metrics.get(k) for k in (
                    "trades", "net_pnl", "profit_factor", "win_rate_pct")}})
                JOBS[jid]["progress"] = f"{i + 1}/{len(rrs)}"
            JOBS[jid].update(status="done", runs=runs, run_id=runs[-1]["run_id"], metrics=runs[-1])
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




CHARGE_INFO = {
    "brokerage": ("Groww", "₹20 per executed order (buy and sell are separate orders)"),
    "stt": ("Government (Securities Transaction Tax)", "0.1% of the SELL-side premium"),
    "exchange": ("NSE transaction charge", "0.03503% of total premium turnover (buy + sell)"),
    "sebi": ("SEBI turnover fee", "₹10 per crore = 0.0001% of turnover"),
    "stamp": ("State stamp duty", "0.003% of the BUY-side premium"),
    "ipft": ("NSE Investor Protection Fund", "0.0005% of turnover"),
    "gst": ("Government GST", "18% on brokerage + exchange + SEBI + IPFT"),
}


@app.get("/api/charges")
def charges(entry: float = 250.0, exit: float = 260.0, lots: int = 1):
    """Round-trip option-buying charges for a given premium path (rates from config/instruments.yaml)."""
    from backtesting.costs.charges import option_charges

    cfg = CFG["instruments"]["charges"]
    lot = 65
    try:
        from backtesting.engine.backtester import lot_size_on

        lot = lot_size_on(datetime.now(IST).date(), CFG["instruments"]["lot_size_schedule"])
    except Exception:  # noqa: BLE001
        pass
    qty = lot * max(1, lots)
    b = option_charges(entry * qty, exit * qty, cfg)
    gross = (exit - entry) * qty
    rows = [{"component": k, "amount": v, "paid_to": CHARGE_INFO[k][0], "rule": CHARGE_INFO[k][1]}
            for k, v in b.to_dict().items() if k != "total"]
    return _clean({"entry": entry, "exit": exit, "qty": qty, "lot_size": lot, "buy_value": entry * qty,
                   "sell_value": exit * qty, "gross_pnl": gross, "total_charges": b.total, "net_pnl": gross - b.total,
                   "breakeven_move_points": round(b.total / qty, 2), "components": rows, "rates": cfg,
                   "note": "Rates are configured in config/instruments.yaml; verify against Groww/NSE periodically. "
                           "Slippage (spread) is a separate cost, not a charge."})


# ---------------------------------------------------------------- Stocks ML (BUY / HOLD / SELL)
STOCK_ML_DIR = PROJECT_ROOT / "ml_lab" / "results"
_STOCK_CACHE: dict = {}


def _bhs_run() -> Path:
    runs = sorted(p for p in STOCK_ML_DIR.glob("bhs_*") if (p / "summary.csv").exists() and (p / "current_signals.csv").exists())
    if not runs:
        raise HTTPException(404, "No stock ML run yet. Run: python3 -m ml_lab.stocks_bhs")
    return runs[-1]


def _stock_names() -> pd.DataFrame:
    """sym -> company name / exchange / ISIN from the latest Groww stock list."""
    if "names" not in _STOCK_CACHE:
        uni = sorted((resolve_path(CFG, "data_lake") / "silver" / "stock_universe").glob("groww_stocks_2*.parquet"))
        if not uni:
            _STOCK_CACHE["names"] = pd.DataFrame(columns=["sym", "name", "exchange", "isin", "groww_symbol"])
        else:
            u = pd.read_parquet(uni[-1])
            u["sym"] = u.nse_symbol.fillna("BSE:" + u.bse_code.astype(str))
            _STOCK_CACHE["names"] = u.rename(columns={"primary_exchange": "exchange"})[["sym", "name", "exchange", "isin", "groww_symbol"]]
    return _STOCK_CACHE["names"]


def _last_close(syms: list[str], as_of: str) -> dict:
    path = resolve_path(CFG, "data_lake") / "silver" / "yahoo_stocks" / "all_daily.parquet"
    if not path.exists():
        return {}
    key = ("px", as_of)
    if key not in _STOCK_CACHE:
        d = pd.read_parquet(path, columns=["sym", "date", "close"], filters=[("date", "==", pd.Timestamp(as_of))])
        _STOCK_CACHE[key] = dict(zip(d.sym, d.close.astype(float)))
    px = _STOCK_CACHE[key]
    return {s: px.get(s) for s in syms}


@app.get("/api/stocks/ml")
def stocks_ml():
    """Model comparison, yearly returns and equity curves of the latest BUY/HOLD/SELL walk-forward run."""
    rd = _bhs_run()
    rep = json.loads((rd / "report.json").read_text()) if (rd / "report.json").exists() else {}
    tab = pd.read_csv(rd / "summary.csv", index_col=0)
    tab.index.name = "model"
    yearly = pd.read_csv(rd / "yearly.csv", index_col=0)
    eq = pd.read_csv(rd / "equity.csv", index_col=0, parse_dates=True)
    curves = {m: [{"date": str(d.date()), "value": round(float(v), 0)} for d, v in eq[m].dropna().items()] for m in eq.columns}
    return _clean({
        "run": rd.name, "chosen": rep.get("chosen_on_2013_2019"), "signals_as_of": rep.get("signals_as_of"),
        "base_rate_up_pct": rep.get("base_rate_up_pct"), "class_share": rep.get("class_share"),
        "models": tab.reset_index().replace({np.nan: None}).to_dict("records"),
        "yearly": [{"year": int(y), **{k: (None if pd.isna(v) else float(v)) for k, v in r.items()}} for y, r in yearly.iterrows()],
        "equity": curves,
        "method": {"universe": "all NSE/BSE stocks with price >= Rs 20, 1y history, median traded value >= Rs 2 crore/day",
                   "labels": "next-month return: BUY > +4%, SELL < -4%, HOLD otherwise",
                   "test": "yearly walk-forward 2013-2026; model chosen on 2013-2019, 2020-2026 is unseen",
                   "backtest": "Rs 1,00,000, max 10 stocks, BUY -> buy, HOLD -> keep, SELL -> sell; Groww charges + slippage",
                   "caveat": "Survivorship bias: only companies listed today; real results will likely be lower."}})


@app.get("/api/stocks/signals")
def stocks_signals(model: str = "", signal: str = "", q: str = "", limit: int = 2000, capital: float = 100_000.0, slots: int = 10):
    """Latest BUY/HOLD/SELL per stock plus the model's top-N portfolio with whole-share quantities."""
    rd = _bhs_run()
    rep = json.loads((rd / "report.json").read_text()) if (rd / "report.json").exists() else {}
    cur = pd.read_csv(rd / "current_signals.csv")
    models = sorted({c[: -len("_signal")] for c in cur.columns if c.endswith("_signal")})
    m = model if model in models else (rep.get("chosen_on_2013_2019") if rep.get("chosen_on_2013_2019") in models else models[0])
    as_of = rep.get("signals_as_of")
    df = pd.DataFrame({"sym": cur.sym, "signal": cur[f"{m}_signal"], "p_buy": cur[f"{m}_p_buy"], "p_sell": cur[f"{m}_p_sell"],
                       "traded_value_cr": (cur.tv / 1e7).round(1)})
    df["score"] = (df.p_buy - df.p_sell).round(3)
    df = df.merge(_stock_names(), on="sym", how="left")
    df["last_close"] = df.sym.map(_last_close(df.sym.tolist(), as_of) if as_of else {})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    counts = df.signal.value_counts().to_dict()
    top = df[df.signal == "BUY"].head(slots).copy()
    per = capital / max(1, slots)
    top["pos"] = np.arange(1, len(top) + 1)
    top["qty"] = [int(per // p) if p and p > 0 else 0 for p in top.last_close]
    top["amount"] = (top.qty * top.last_close).round(0)
    out = df
    if signal:
        out = out[out.signal == signal.upper()]
    if q:
        ql = q.lower()
        out = out[out.sym.str.lower().str.contains(ql, regex=False) | out.name.fillna("").str.lower().str.contains(ql, regex=False)]
    return _clean({"run": rd.name, "model": m, "models": models, "as_of": as_of, "counts": counts, "total": int(len(df)),
                   "portfolio": top.replace({np.nan: None}).to_dict("records"),
                   "portfolio_note": f"Top {slots} BUY signals by score, Rs {per:,.0f} each at the last close (whole shares). "
                                     "Research output for paper trading, not investment advice.",
                   "rows": out.head(limit).replace({np.nan: None}).to_dict("records")})


def _forecast_run() -> Path | None:
    runs = sorted(p for p in STOCK_ML_DIR.glob("forecast_*") if (p / "forecast.csv").exists())
    return runs[-1] if runs else None


def _trading_days_after(d: pd.Timestamp, n: int) -> pd.Timestamp:
    """n NSE trading days after d (exchange holidays from the market calendar; weekdays as fallback)."""
    try:
        from core.clock import MarketCalendar

        cal = MarketCalendar(CFG["settings"], CFG["instruments"])
        x = d.date()
        for _ in range(n):
            x = cal.next_trading_day(x)
        return pd.Timestamp(x)
    except Exception:  # noqa: BLE001
        return pd.bdate_range(d, periods=n + 1)[-1]


@app.get("/api/stocks/forecast")
def stocks_forecast(sym: str, history_days: int = 250):
    """Next-day and ~1-month price RANGE for one stock (calibrated 10/50/90 % quantiles) + recent closes."""
    rd = _forecast_run()
    if rd is None:
        raise HTTPException(404, "No price forecast yet. Run: python3 -m ml_lab.stock_price_forecast")
    key = ("fc", rd.name)
    if key not in _STOCK_CACHE:
        _STOCK_CACHE[key] = (pd.read_csv(rd / "forecast.csv").set_index("sym"), json.loads((rd / "metrics.json").read_text()))
    fc, met = _STOCK_CACHE[key]
    if sym not in fc.index:
        raise HTTPException(404, f"{sym}: no forecast (not in the liquid universe: price >= Rs 20 and >= Rs 2 crore traded/day)")
    r = fc.loc[sym]
    as_of = pd.Timestamp(r.as_of)
    path = resolve_path(CFG, "data_lake") / "silver" / "yahoo_stocks" / "all_daily.parquet"
    h = pd.read_parquet(path, columns=["date", "close"], filters=[("sym", "==", sym), ("date", ">", as_of - pd.Timedelta(days=history_days * 1.5))])
    h = h.sort_values("date").tail(history_days)
    targets = {k: _trading_days_after(as_of, int(r[f"{k}_days"])) for k in ("d1", "m1")}
    names = _stock_names().set_index("sym")
    acc = met.get("accuracy", {})
    horizon = {}
    for k, label in (("d1", "next trading day"), ("m1", "about 1 month (21 trading days)")):
        a = acc.get(k, {})
        horizon[k] = {"label": label, "target_date": str(targets[k].date()),
                      **{q: float(r[f"{k}_{q}"]) for q in ("lo", "mid", "hi")},
                      **{f"{q}_pct": float(r[f"{k}_{q}_ret_pct"]) for q in ("lo", "mid", "hi")},
                      "range_hit_rate_pct": a.get("range_contains_actual_pct"),
                      "mid_error_pct": a.get("avg_error_pct_model"), "same_price_error_pct": a.get("avg_error_pct_same_price"),
                      "direction_correct_pct": a.get("direction_correct_pct")}
    return _clean({"sym": sym, "name": names.name.get(sym) if sym in names.index else None, "as_of": r.as_of,
                   "close": float(r.close), "run": rd.name, "horizons": horizon,
                   "history": [{"date": str(d.date()), "close": round(float(c), 2)} for d, c in zip(h.date, h.close)],
                   "note": "Range = where the price should land about 3 times in 4 (tested 2024-2026 on unseen data). "
                           "The middle value is not more accurate than today's price; prices are close to unpredictable."})


_SIG = ["SELL", "HOLD", "BUY"]


def _bhs_preds(rd: Path) -> pd.DataFrame:
    key = ("preds", rd.name)
    if key not in _STOCK_CACHE:
        fs = sorted(rd.glob("pred_*.parquet"))
        _STOCK_CACHE[key] = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
    return _STOCK_CACHE[key]


@app.get("/api/stocks/model_signals")
def stocks_model_signals(sym: str):
    """One stock: today's BUY/HOLD/SELL from every ML/DL model + its month-by-month backtest signals and outcomes."""
    rd = _bhs_run()
    rep = json.loads((rd / "report.json").read_text()) if (rd / "report.json").exists() else {}
    fam = pd.read_csv(rd / "summary.csv", index_col=0)["family"].to_dict()
    cur = pd.read_csv(rd / "current_signals.csv")
    models = [c[: -len("_signal")] for c in cur.columns if c.endswith("_signal")]
    row = cur[cur.sym == sym]
    current = []
    if len(row):
        r = row.iloc[0]
        current = [{"model": m, "family": fam.get(m), "signal": r[f"{m}_signal"], "p_buy": float(r[f"{m}_p_buy"]),
                    "p_sell": float(r[f"{m}_p_sell"])} for m in models]
    votes = {s: sum(1 for c in current if c["signal"] == s and c["model"] != "ensemble") for s in _SIG}
    P = _bhs_preds(rd)
    hist, stats = [], []
    if len(P):
        h = P[P.sym == sym].sort_values("date")
        if rep.get("signals_as_of"):                       # today's row lives in "current" (fresh all-data models)
            h = h[h.date < pd.Timestamp(rep["signals_as_of"])]
        pm = [m for m in (fam.keys()) if f"{m}_p0" in h.columns]
        for _, x in h.iterrows():
            rec = {"date": str(pd.Timestamp(x.date).date()), "fwd_pct": None if pd.isna(x.fwd) else round(float(x.fwd) * 100, 2),
                   "actual": None if x.y < 0 else _SIG[int(x.y)]}
            for m in pm:
                rec[m] = _SIG[int(np.argmax([x[f"{m}_p0"], x[f"{m}_p1"], x[f"{m}_p2"]]))]
            hist.append(rec)
        done = [r for r in hist if r["actual"]]
        for m in pm:
            buys = [r["fwd_pct"] for r in done if r[m] == "BUY"]
            sells = [r["fwd_pct"] for r in done if r[m] == "SELL"]
            stats.append({"model": m, "family": fam.get(m), "months": len(done),
                          "hit_rate_pct": round(100 * sum(r[m] == r["actual"] for r in done) / len(done), 1) if done else None,
                          "buy_months": len(buys), "buy_avg_next_month_pct": round(float(np.mean(buys)), 2) if buys else None,
                          "buy_up_pct": round(100 * float(np.mean([b > 0 for b in buys])), 1) if buys else None,
                          "sell_months": len(sells), "sell_avg_next_month_pct": round(float(np.mean(sells)), 2) if sells else None})
    return _clean({"sym": sym, "as_of": rep.get("signals_as_of"), "chosen": rep.get("chosen_on_2013_2019"),
                   "current": current, "votes": votes, "n_models": sum(votes.values()),
                   "history": hist[::-1], "stock_stats": stats,
                   "note": "History = out-of-sample: each month's signal came from models trained only on earlier years. "
                           "A stock appears only in months when it met the liquidity filter."})
