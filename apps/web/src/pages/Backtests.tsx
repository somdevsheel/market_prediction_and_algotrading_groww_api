import { useMemo, useState } from "react";
import { api } from "../api";
import { Chart } from "../components/Chart";
import { DataTable } from "../components/DataTable";
import { Card, Stat, State } from "../components/ui";
import { inr, num, pct, tone } from "../format";
import { usePoll } from "../hooks";

const today = () => new Date().toISOString().slice(0, 10);

export default function Backtests({ theme }: { theme: string }) {
  const list = usePoll(() => api.get("/api/backtests"));
  const st = usePoll(() => api.get("/api/strategies"));
  const [form, setForm] = useState({ start: "2024-01-01", end: today(), scenario: "conservative", capital: "20000",
    risk_per_trade_max: "1000", charges_multiplier: "1", itm_steps: "1", square_off: "15:15", reward_risk: "none" });
  const [customRR, setCustomRR] = useState("2.5");
  const [sweep, setSweep] = useState(false);
  const SWEEP = ["none", "1", "1.5", "2", "3"];
  const [job, setJob] = useState<{ id?: string; status?: string; msg?: string }>({});
  const [sel, setSel] = useState<string[]>([]);
  const [open, setOpen] = useState<string | null>(null);
  const detail = usePoll(() => (open ? api.get(`/api/backtests/${open}`) : Promise.resolve(null)), 0, [open]);
  const cmp = usePoll(() => (sel.length ? api.get(`/api/backtests_compare?ids=${sel.join(",")}`) : Promise.resolve([])), 0, [sel.join(",")]);
  const active = (st.data || []).find((x: any) => x.active);

  const run = async () => {
    const rr = form.reward_risk === "custom" ? customRR : form.reward_risk;
    setJob({ status: "running", msg: sweep ? `Running R:R sweep ${SWEEP.map((x) => (x === "none" ? "none" : `1:${x}`)).join(", ")}…` : "Running… (loads real option candles; ~1–3 min, longer if new data must download)" });
    try {
      const { job_id } = await api.post("/api/backtests/run", { ...form, strategy_id: active?.strategy_id,
        reward_risk: rr === "none" ? null : rr, rr_sweep: sweep ? SWEEP : undefined });
      const poll = window.setInterval(async () => {
        const s = await api.get(`/api/jobs/${job_id}`);
        if (s.status === "running") { setJob((j) => ({ ...j, msg: `Running… ${s.progress || ""}` })); return; }
        window.clearInterval(poll);
        const rrLabel = (x: number | null) => (x ? `1:${x}` : "no target");
        setJob({ id: job_id, status: s.status, msg: s.status !== "done" ? `Failed: ${s.error}`
          : (s.runs || []).map((r: any) => `${rrLabel(r.reward_risk)} → ${inr(r.net_pnl)} (PF ${num(r.profit_factor)}, win ${pct(r.win_rate_pct)})`).join("  ·  ") });
        list.reload();
        if ((s.runs || []).length > 1) setSel(s.runs.map((r: any) => r.run_id).slice(-5));
        else if (s.run_id) setOpen(s.run_id);
      }, 3000);
    } catch (e: any) { setJob({ status: "failed", msg: e.message }); }
  };
  const set = (k: string) => (e: any) => setForm({ ...form, [k]: e.target.value });
  const toggle = (id: string) => setSel((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id].slice(-5)));
  const rrOf = (exp: string) => { const m = /_rr([\d.]+)/.exec(exp || ""); return m ? `1:${m[1]}` : "no target"; };
  const cmpSeries = useMemo(() => (cmp.data || []).map((c: any, i: number) => ({ name: `${rrOf(c.experiment)} · ${c.scenario}`, color: ["--accent", "--pos", "--warn", "--neg", "--muted"][i % 5],
    data: (c.equity || []).map((e: any) => ({ time: e.date, value: e.equity })) })), [cmp.data]);
  const dEq = useMemo(() => [{ name: "Equity", type: "area" as const, color: "--accent", data: (detail.data?.equity || []).map((e: any) => ({ time: e.date, value: e.equity })) }], [detail.data]);
  const m = detail.data?.metrics || {};

  return (
    <>
      <Card title={`Run backtest · ${active?.strategy_id || "…"}`}>
        <div className="form">
          <label>Start<input type="date" value={form.start} onChange={set("start")} /></label>
          <label>End<input type="date" value={form.end} onChange={set("end")} /></label>
          <label>Costs<select value={form.scenario} onChange={set("scenario")}><option>optimistic</option><option>base</option><option>conservative</option></select></label>
          <label>Capital ₹<input value={form.capital} onChange={set("capital")} size={8} /></label>
          <label>Max risk / trade ₹<input value={form.risk_per_trade_max} onChange={set("risk_per_trade_max")} size={6} /></label>
          <label>Charges ×<input value={form.charges_multiplier} onChange={set("charges_multiplier")} size={3} /></label>
          <label>ITM steps<input value={form.itm_steps} onChange={set("itm_steps")} size={2} /></label>
          <label>Square-off<input value={form.square_off} onChange={set("square_off")} size={5} /></label>
          <label title="Profit target on the option premium = entry + R:R × stop distance (1R = the risk-budget stop)">Target (risk : reward)
            <select value={form.reward_risk} onChange={set("reward_risk")} disabled={sweep}>
              <option value="none">no fixed target (strategy default)</option>
              <option value="1">1 : 1</option><option value="1.5">1 : 1.5</option><option value="2">1 : 2</option>
              <option value="3">1 : 3</option><option value="4">1 : 4</option><option value="custom">custom…</option>
            </select></label>
          {form.reward_risk === "custom" && !sweep && <label>Reward (R)<input value={customRR} onChange={(e) => setCustomRR(e.target.value)} size={4} /></label>}
          <label style={{ flexDirection: "row", alignItems: "center", gap: 6, paddingBottom: 8 }} title="Runs no-target, 1:1, 1:1.5, 1:2 and 1:3 and opens the comparison">
            <input type="checkbox" checked={sweep} onChange={(e) => setSweep(e.target.checked)} /> R:R sweep</label>
          <button className="btn primary" disabled={job.status === "running"} onClick={run}>Run</button>
        </div>
        {job.msg && <div className={`banner ${job.status === "failed" ? "bad" : ""}`} style={{ marginTop: 12 }}>{job.msg}</div>}
        <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>Simulated fills on real historical option candles with modelled spread/slippage; not actual fills.</div>
      </Card>

      <div className="section">
        <Card title="Runs" right={<span className="muted" style={{ fontSize: 12 }}>tick up to 5 to compare · click a row for details</span>}>
          <State loading={list.loading && !list.data} error={list.error}>
            <DataTable rows={list.data} empty="No runs yet." onRowClick={(r) => setOpen(r.run_id)} maxHeight={360} cols={[
              { key: "sel", label: "", render: (r) => <input type="checkbox" checked={sel.includes(r.run_id)} onClick={(e) => e.stopPropagation()} onChange={() => toggle(r.run_id)} /> },
              { key: "run_id", label: "Run", render: (r) => <span className="mono">{r.run_id.slice(0, 15)}</span> },
              { key: "experiment", label: "Strategy" }, { key: "scenario", label: "Costs" },
              { key: "rr_target", label: "Target", render: (r) => (r.rr_target ? `1:${r.rr_target}` : "none") },
              { key: "start", label: "From" }, { key: "end", label: "To" },
              { key: "trades", label: "Trades", num: true },
              { key: "net_pnl", label: "Net", num: true, render: (r) => <span className={tone(r.net_pnl)}>{inr(r.net_pnl)}</span> },
              { key: "profit_factor", label: "PF", num: true }, { key: "win_rate_pct", label: "Win %", num: true },
              { key: "payoff_ratio", label: "R:R", num: true, render: (r) => (r.payoff_ratio ? `${num(r.payoff_ratio, 2)}:1` : "—") },
              { key: "avg_charges_per_trade", label: "Charges/trade", num: true, render: (r) => inr(r.avg_charges_per_trade, 0) },
              { key: "max_drawdown", label: "Max DD", num: true, render: (r) => inr(r.max_drawdown) },
              { key: "oos_net", label: "OOS net", num: true, render: (r) => <span className={tone(r.oos_net)}>{inr(r.oos_net)}</span> },
            ]} />
          </State>
        </Card>
      </div>

      {sel.length > 0 && (
        <div className="section">
          <Card title="Comparison">
            <State loading={cmp.loading && !cmp.data} error={cmp.error}>
              <Chart series={cmpSeries} theme={theme} />
              <div className="section"><DataTable rows={(cmp.data || []).map((c: any) => ({ run: `${c.experiment} / ${c.scenario}`, ...c.metrics, is: c.in_sample?.net_pnl, oos: c.out_of_sample?.net_pnl }))} cols={[
                { key: "rr", label: "Target", render: (r) => { const m = /_rr([\d.]+)/.exec(r.run); return m ? `1:${m[1]}` : "none"; } },
                { key: "run", label: "Run" }, { key: "trades", label: "Trades", num: true },
                { key: "net_pnl", label: "Net", num: true, render: (r) => inr(r.net_pnl) }, { key: "profit_factor", label: "PF", num: true },
                { key: "win_rate_pct", label: "Win %", num: true }, { key: "max_drawdown", label: "Max DD", num: true, render: (r) => inr(r.max_drawdown) },
                { key: "sharpe", label: "Sharpe", num: true }, { key: "is", label: "In-sample", num: true, render: (r) => inr(r.is) },
                { key: "oos", label: "Out-of-sample", num: true, render: (r) => inr(r.oos) }]} /></div>
            </State>
          </Card>
        </div>
      )}

      {open && (
        <div className="section">
          <Card title={<span className="mono" style={{ textTransform: "none" }}>{open}</span>}
                right={<><a className="btn" href={`/api/backtests/${open}/trades.csv`}>Trades CSV</a><button className="btn" onClick={() => setOpen(null)}>Close</button></>}>
            <State loading={detail.loading} error={detail.error}>
              <div className="grid">
                <Stat k="Net P&L" v={inr(m.net_pnl)} tone={tone(m.net_pnl)} />
                <Stat k="Gross / charges" v={`${inr(m.gross_pnl)} / ${inr(m.charges)}`} />
                <Stat k="Trades" v={num(m.trades, 0)} />
                <Stat k="Win rate" v={pct(m.win_rate_pct)} s={`avg win ${inr(m.avg_win)} · loss ${inr(m.avg_loss)}`} />
                <Stat k="Profit factor" v={num(m.profit_factor, 2)} />
                <Stat k="Max drawdown" v={inr(m.max_drawdown)} tone={tone(m.max_drawdown)} />
                <Stat k="Risk : reward (payoff)" v={m.payoff_ratio ? `${num(m.payoff_ratio, 2)} : 1` : "—"} s={`break-even win rate ${pct(m.breakeven_win_rate_pct)}`} />
                <Stat k="Expectancy" v={`${inr(m.expectancy)} · ${num(m.avg_r_multiple, 2)}R`} tone={tone(m.expectancy)} />
                <Stat k="Charges / trade" v={inr(m.avg_charges_per_trade, 2)} s={`${num(m.cost_impact_pct_of_gross, 1)}% of gross P&L`} />
              </div>
              <div className="section"><Chart series={dEq} theme={theme} /></div>
              <div className="section">
                <DataTable rows={detail.data?.trades} maxHeight={420} cols={[
                  { key: "trading_day", label: "Day" }, { key: "direction", label: "Dir" }, { key: "trading_symbol", label: "Contract" },
                  { key: "entry_time", label: "Entry", render: (r) => String(r.entry_time).slice(11, 16) }, { key: "entry_price", label: "Entry ₹", num: true },
                  { key: "exit_time", label: "Exit", render: (r) => String(r.exit_time).slice(11, 16) }, { key: "exit_price", label: "Exit ₹", num: true },
                  { key: "exit_reason", label: "Exit reason" }, { key: "iv_at_entry", label: "IV", num: true },
                  { key: "r", label: "R", num: true, value: (r) => r.net_pnl / r.planned_risk, render: (r) => (r.planned_risk ? `${(r.net_pnl / r.planned_risk).toFixed(2)}R` : "—") },
                  { key: "charges", label: "Charges", num: true, render: (r) => inr(r.charges, 2) },
                  { key: "net_pnl", label: "Net", num: true, render: (r) => <span className={tone(r.net_pnl)}>{inr(r.net_pnl)}</span> },
                ]} />
              </div>
            </State>
          </Card>
        </div>
      )}
    </>
  );
}
