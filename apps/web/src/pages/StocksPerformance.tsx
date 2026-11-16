import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { Col, DataTable } from "../components/DataTable";
import { Card, Chip, Seg, State } from "../components/ui";
import { inr, pct, time, tone } from "../format";
import { usePoll } from "../hooks";

const SIG_TONE: Record<string, "ok" | "bad" | ""> = { BUY: "ok", SELL: "bad", HOLD: "" };
const ICON: Record<string, string> = { pending: "○", running: "◐", done: "✓", failed: "✕" };
const signed = (v: unknown, d = 2) => <span className={tone(v)}>{typeof v === "number" ? `${v > 0 ? "+" : ""}${v.toFixed(d)}%` : "—"}</span>;

/** One-click: new prices -> retrain the main model (XGBoost) -> new BUY/HOLD/SELL -> new price forecasts. */
export function RefreshBar({ onDone }: { onDone: () => void }) {
  const [st, setSt] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const prev = useRef<string | undefined>();
  const running = st?.state === "running";

  useEffect(() => {
    let alive = true;
    const load = () => api.get("/api/stocks/refresh").then((s) => { if (alive) { setSt(s); setErr(null); } })
      .catch((e) => alive && setErr(e.message));
    load();
    const id = window.setInterval(load, running ? 3000 : 30000);
    return () => { alive = false; window.clearInterval(id); };
  }, [running]);

  useEffect(() => {                          // finished -> reload everything on the page
    if (prev.current === "running" && st?.state === "done") onDone();
    prev.current = st?.state;
  }, [st?.state, onDone]);

  const start = async () => {
    setBusy(true);
    try { setSt(await api.post("/api/stocks/refresh", {})); setErr(null); }
    catch (e: any) { setErr(e.message); }
    finally { setBusy(false); }
  };
  const steps: any[] = st?.steps || [];

  return (
    <Card title="Data & models" right={
      <button className="btn primary" disabled={running || busy} onClick={start}
              title="Download the latest prices, retrain the main model (XGBoost) on all data, write new signals and price forecasts">
        {running ? "Updating…" : "⟳ Update data & retrain"}
      </button>}>
      <div style={{ display: "flex", gap: 18, flexWrap: "wrap", fontSize: 13 }}>
        <span>Prices up to <b className="mono">{st?.data_to || "—"}</b></span>
        <span>Signals as of <b className="mono">{st?.signals_as_of || "—"}</b></span>
        <span>Last refresh: {st?.state === "idle" || !st ? "—" :
          <><Chip tone={st.state === "done" ? "ok" : st.state === "failed" ? "bad" : "warn"}>{st.state}</Chip>{" "}
            <span className="muted">{time(st.finished || st.started)}</span></>}</span>
      </div>
      {err && <div className="error" style={{ marginTop: 8 }}>⚠ {err}</div>}
      {st?.error && <div className="error" style={{ marginTop: 8 }}>⚠ {st.error}</div>}
      {steps.length > 0 && (
        <ol style={{ margin: "10px 0 0", paddingLeft: 18, fontSize: 13, lineHeight: 1.8 }}>
          {steps.map((s, i) => (
            <li key={i} className={s.state === "failed" ? "neg" : s.state === "done" ? "" : s.state === "running" ? "warn" : "muted"}>
              {ICON[s.state] || "○"} {s.name}{s.state === "running" ? " — running…" : ""}
            </li>))}
        </ol>)}
      {(running || st?.state === "failed") && st?.log_tail?.length > 0 && (
        <details style={{ marginTop: 8 }} open={st?.state === "failed"}>
          <summary className="muted" style={{ cursor: "pointer", fontSize: 12 }}>Log</summary>
          <pre className="mono" style={{ fontSize: 11, maxHeight: 220, overflow: "auto", margin: "6px 0 0", whiteSpace: "pre-wrap" }}>{st.log_tail.join("\n")}</pre>
        </details>)}
      <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
        Run after market close (~3:45 PM IST or later). Takes a few minutes on the CPU (the GPU is not used): index data → prices for all stocks → retrain the main model (XGBoost) on all data → today's BUY/HOLD/SELL (saved as a dated snapshot) → price forecasts. Same as <span className="mono">python3 -m ml_lab.daily_refresh</span>. All 13 models: <span className="mono">python3 -m ml_lab.stocks_bhs --current-only --all-models</span> (slow on CPU). The yearly walk-forward backtest is not re-run.
      </div>
    </Card>
  );
}

type Grp = "top" | "BUY" | "HOLD" | "SELL" | "";

/** How the stocks predicted on a past day actually did. */
export function PredictionPerformance({ name, reloadKey, onPick }: { name: (m: string) => string; reloadKey: number; onPick: (sym: string) => void }) {
  const [date, setDate] = useState("");
  const [model, setModel] = useState("");
  const [grp, setGrp] = useState<Grp>("top");
  const [q, setQ] = useState("");
  const perf = usePoll(() => api.get(`/api/stocks/performance?date=${date}&model=${model}`), 0, [date, model, reloadKey]);
  const d = perf.data;

  const rows = useMemo(() => {
    const ql = q.trim().toLowerCase();
    return (d?.rows || []).filter((r: any) => (grp === "top" ? r.top : !grp || r.signal === grp) &&
      (!ql || r.sym.toLowerCase().includes(ql) || (r.name || "").toLowerCase().includes(ql)));
  }, [d, grp, q]);

  const sumRows = useMemo(() => d ? [...(d.groups || []).map((g: any) => ({ ...g })),
    { group: "NIFTY 50", stocks: null, next: { avg: d.nifty?.next }, since: { avg: d.nifty?.since } }] : [], [d]);
  const sameDay = d?.next_day && d?.latest && d.next_day === d.latest;

  const sumCols: Col<any>[] = [
    { key: "group", label: "Group", render: (r) => <b>{r.group}</b> },
    { key: "stocks", label: "Stocks", num: true, render: (r) => r.stocks ?? "—" },
    { key: "n", label: `Next day ${d?.next_day || ""}`, num: true, value: (r) => r.next?.avg, render: (r) => signed(r.next?.avg) },
    { key: "med", label: "Median", num: true, value: (r) => r.next?.median, render: (r) => signed(r.next?.median) },
    { key: "ud", label: "Up / down", num: true, render: (r) => r.next?.n ? <><span className="pos">{r.next.up}</span> / <span className="neg">{r.next.down}</span></> : "—" },
    { key: "upp", label: "% up", num: true, value: (r) => r.next?.n ? r.next.up / r.next.n : null, render: (r) => r.next?.n ? pct((100 * r.next.up) / r.next.n, 0) : "—" },
    ...(sameDay ? [] : [{ key: "s", label: `Since signal → ${d?.latest || ""}`, num: true, value: (r: any) => r.since?.avg, render: (r: any) => signed(r.since?.avg) }]),
  ];
  const trackCols: Col<any>[] = [
    { key: "date", label: "Signal date", render: (r) => <span className="mono">{r.date}</span> },
    { key: "next_day", label: "Checked on", render: (r) => <span className="mono">{r.next_day}</span> },
    { key: "top", label: "Top-10", num: true, render: (r) => signed(r.top) },
    { key: "buy", label: "BUY", num: true, render: (r) => signed(r.buy) },
    { key: "hold", label: "HOLD", num: true, render: (r) => signed(r.hold) },
    { key: "sell", label: "SELL", num: true, render: (r) => signed(r.sell) },
    { key: "all", label: "All stocks", num: true, render: (r) => signed(r.all) },
    { key: "nifty", label: "NIFTY 50", num: true, render: (r) => signed(r.nifty) },
    { key: "buy_minus_all", label: "BUY − all", num: true, render: (r) => signed(r.buy_minus_all) },
    { key: "buy_up_pct", label: "BUY % up", num: true, render: (r) => pct(r.buy_up_pct, 0) },
  ];
  const stockCols: Col<any>[] = [
    { key: "sym", label: "Symbol", render: (r) => <span className="mono">{r.sym}{r.top ? " ★" : ""}</span> },
    { key: "name", label: "Company" },
    { key: "signal", label: "Signal", render: (r) => <Chip tone={SIG_TONE[r.signal]}>{r.signal}</Chip> },
    { key: "p_buy", label: "P(BUY)", num: true, render: (r) => pct(r.p_buy * 100, 1) },
    { key: "score", label: "Score", num: true },
    { key: "close_signal", label: `Close ${d?.date || ""}`, num: true, render: (r) => inr(r.close_signal, 2) },
    { key: "close_next", label: `Close ${d?.next_day || ""}`, num: true, render: (r) => inr(r.close_next, 2) },
    { key: "ret_next_pct", label: "Next day", num: true, render: (r) => signed(r.ret_next_pct) },
    ...(sameDay ? [] : [{ key: "ret_since_pct", label: "Since signal", num: true, render: (r: any) => signed(r.ret_since_pct) }]),
  ];
  const g = (k: string) => (d?.groups || []).find((x: any) => x.group.startsWith(k));
  const top = g("Top"); const buy = g("BUY"); const all = g("All");

  return (
    <Card title="Prediction performance · how past signals actually did" right={
      <span style={{ display: "flex", gap: 8, alignItems: "center", textTransform: "none" }}>
        <select value={d?.date || date} onChange={(e) => setDate(e.target.value)} title="Signal date (saved snapshot)">
          {(d?.snapshots || []).slice().reverse().map((s: string) => <option key={s} value={s}>Signals of {s}</option>)}
        </select>
        <select value={d?.model || model} onChange={(e) => setModel(e.target.value)} title="Model">
          {(d?.models || []).map((m: string) => <option key={m} value={m}>{name(m)}{m === d?.chosen ? " ⭐" : ""}</option>)}
        </select>
      </span>}>
      <State loading={perf.loading && !d} error={perf.error} empty={d && !d.snapshots?.length && (d.note || "No snapshots yet")}>
        {d?.note && <div className="banner warn" style={{ marginBottom: 10 }}>{d.note}</div>}
        {d?.next_day && (
          <div style={{ fontSize: 14, marginBottom: 10 }}>
            Signals of <b className="mono">{d.date}</b> ({name(d.model)}), checked on <b className="mono">{d.next_day}</b>:{" "}
            Top-10 {signed(top?.next?.avg)}, all {buy?.stocks} BUY stocks {signed(buy?.next?.avg)} ({buy?.next?.up} up / {buy?.next?.down} down),
            all stocks {signed(all?.next?.avg)}, NIFTY 50 {signed(d.nifty?.next)}.
            {" "}With ₹10,000 in each BUY stock: <b className={tone(buy?.next?.avg)}>{inr(((buy?.next?.avg || 0) / 100) * 10000 * (buy?.next?.n || 0))}</b> on {inr(10000 * (buy?.next?.n || 0))}.
          </div>)}
        <DataTable rows={sumRows} cols={sumCols} />

        <h3 style={{ fontSize: 13, margin: "16px 0 6px" }}>Day-by-day track record (next trading day after each signal)</h3>
        <DataTable rows={d?.track} cols={trackCols} maxHeight={260} empty="No checked days yet" />

        <h3 style={{ fontSize: 13, margin: "16px 0 6px", display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          Stocks
          <Seg<Grp> value={grp} onChange={setGrp} options={[{ v: "top", label: "Top-10 ★" }, { v: "BUY", label: `BUY ${buy?.stocks ?? ""}` },
            { v: "HOLD", label: "HOLD" }, { v: "SELL", label: "SELL" }, { v: "", label: "All" }]} />
          <input placeholder="Search symbol or company" value={q} onChange={(e) => setQ(e.target.value)} style={{ width: 200, fontWeight: 400 }} />
        </h3>
        <DataTable rows={rows} cols={stockCols} maxHeight={420} empty="No stocks" onRowClick={(r: any) => onPick(r.sym)} />
        <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
          Return = close on the next trading day ÷ close on the signal day − 1 (before charges). ★ = in that day's top-10 (BUY ranked by P(BUY) − P(SELL)).
          The models predict about one month ahead — single days are mostly noise; judge them over many days.
        </div>
      </State>
    </Card>
  );
}
