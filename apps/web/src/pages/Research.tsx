import { useMemo, useState } from "react";
import { api } from "../api";
import { Chart } from "../components/Chart";
import { DataTable } from "../components/DataTable";
import { Card, Stat, State } from "../components/ui";
import { num, pct } from "../format";
import { usePoll } from "../hooks";

export default function Research({ theme }: { theme: string }) {
  const m = usePoll(() => api.get("/api/research/models"));
  const r = usePoll(() => api.get("/api/research"));
  const [ts, setTs] = useState<{ loading?: boolean; data?: any[]; error?: string }>({});
  const f = m.data?.forecasts || {};
  const series = useMemo(() => {
    const s = m.data?.series || [];
    const mk = (k: string, name: string, color: string) => ({ name, color, data: s.map((x: any) => ({ time: x.date.slice(0, 10), value: x[k] })) });
    return [mk("garch_fc", "GARCH forecast", "--accent"), mk("atm_iv_prev", "ATM IV", "--warn"), mk("hv20", "HV20", "--muted")];
  }, [m.data]);
  const loadTerm = async () => {
    setTs({ loading: true });
    try { setTs({ data: await api.get("/api/research/term_structure") }); } catch (e: any) { setTs({ error: e.message }); }
  };
  const cheap = f.atm_iv_prev && f.garch_fc ? f.atm_iv_prev / f.garch_fc : null;
  return (
    <State loading={m.loading && !m.data} error={m.error} empty={m.data?.note}>
      <Card title={`Next session · ${m.data?.next_session || ""}`} right={<span className="muted" style={{ fontSize: 12, textTransform: "none" }}>features use data through the previous close</span>}>
        <div className="grid">
          <Stat k="GARCH forecast (strategy filter)" v={pct(f.garch_fc)} s="option IV must be ≤ this" />
          <Stat k="ATM IV (prev close)" v={pct(f.atm_iv_prev)} s={cheap ? (cheap <= 1 ? "options cheap vs forecast" : "options rich vs forecast") : ""} tone={cheap ? (cheap <= 1 ? "pos" : "warn") : ""} />
          <Stat k="EWMA forecast" v={pct(f.ewma_fc)} />
          <Stat k="HV20 / HV60" v={`${num(f.hv20, 1)} / ${num(f.hv60, 1)}`} />
          <Stat k="India VIX (prev)" v={num(f.vix_prev)} />
          <Stat k="IV rank / percentile (1y)" v={`${num(m.data?.iv_rank_252, 0)} / ${num(m.data?.iv_percentile_252, 0)}`} />
          <Stat k="HMM regime" v={m.data?.hmm?.label?.replace("hmm_", "") || "—"} />
          <Stat k="Scheduled events" v={m.data?.events_next_session || "none"} />
        </div>
      </Card>
      <div className="section">
        <Card title="Forecast volatility vs implied volatility (annualised %)"><Chart series={series} theme={theme} /></Card>
      </div>
      <div className="cols-2 section">
        <Card title="Volatility models · out-of-sample QLIKE (lower is better)">
          <DataTable rows={m.data?.vol_model_eval} empty="Run: python3 scripts/build_gold.py features" cols={[
            { key: "model", label: "Model" }, { key: "oos_qlike", label: "QLIKE", num: true, render: (x) => num(x.oos_qlike, 4) }, { key: "n", label: "N", num: true }]} />
        </Card>
        <Card title="GARCH family (daily NIFTY)">
          <State loading={r.loading && !r.data} error={r.error}>
            {typeof r.data?.garch === "string" ? <div className="muted">{r.data.garch}</div> : (
              <DataTable rows={Object.entries(r.data?.garch || {}).map(([k, v]: any) => ({ model: k, ...v }))} cols={[
                { key: "model", label: "Model" }, { key: "aic", label: "AIC", num: true }, { key: "persistence", label: "Persistence", num: true },
                { key: "forecast_5d_vol", label: "5-day vol %", num: true }, { key: "converged", label: "Converged" }]} />)}
          </State>
        </Card>
      </div>
      <div className="section">
        <Card title="IV term structure (live)" right={<button className="btn" disabled={ts.loading} onClick={loadTerm}>{ts.loading ? "Loading…" : "Load"}</button>}>
          {ts.error ? <div className="error">⚠ {ts.error}</div> : ts.data ? (
            <DataTable rows={ts.data} cols={[{ key: "expiry", label: "Expiry" }, { key: "days", label: "Days", num: true },
              { key: "atm_iv_model", label: "ATM IV (model)", num: true, render: (x) => pct(x.atm_iv_model, 2) },
              { key: "atm_iv_provider", label: "ATM IV (Groww)", num: true, render: (x) => pct(x.atm_iv_provider, 2) }]} />
          ) : <div className="muted">Fetches option chains for the next expiries (needs a valid Groww token).</div>}
        </Card>
      </div>
    </State>
  );
}
