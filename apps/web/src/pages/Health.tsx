import { api } from "../api";
import { DataTable } from "../components/DataTable";
import { Card, Chip, Stat, State } from "../components/ui";
import { ago, num, time } from "../format";
import { usePoll } from "../hooks";

export default function Health() {
  const h = usePoll(() => api.get("/api/health"), 15000);
  const al = usePoll(() => api.get("/api/alerts"), 30000);
  const dq = usePoll(() => api.get("/api/data_quality"), 0);
  const c = h.data?.checks || {};
  const ok = (b: any) => (b ? "ok" : "bad");
  const iq = dq.data?.instrument_quality;
  return (
    <State loading={h.loading && !h.data} error={h.error}>
      <div className="grid">
        <Stat k="Overall" v={<Chip tone={h.data?.overall === "ok" ? "ok" : "warn"}>{h.data?.overall}</Chip>} />
        <Stat k="Database" v={<Chip tone={ok(c.database)}>{c.database ? "ok" : "FAIL"}</Chip>} />
        <Stat k="Worker" v={<Chip tone={c.worker_alive ? "ok" : "warn"}>{c.worker_alive ? "alive" : "not running"}</Chip>} s={c.worker_heartbeat_age_s != null ? `last heartbeat ${ago(new Date(Date.now() - c.worker_heartbeat_age_s * 1000).toISOString())}` : "no heartbeat yet"} />
        <Stat k="Broker token" v={<Chip tone={ok(c.broker_token_valid)}>{c.broker_token_valid ? "valid" : "EXPIRED"}</Chip>} s={c.broker_token_expires ? `expires ${time(c.broker_token_expires)}` : ""} />
        <Stat k="Last market data" v={ago(c.last_market_data)} />
        <Stat k="Reconciliation" v={<Chip tone={c.reconciliation_required ? "bad" : "ok"}>{c.reconciliation_required ? `${c.reconciliation_required} required` : "clean"}</Chip>} />
        <Stat k="Kill switch" v={<Chip tone={c.kill_switch ? "bad" : "ok"}>{c.kill_switch ? "ACTIVE" : "off"}</Chip>} />
        <Stat k="Market feed" v={<span style={{ fontSize: 14 }}>REST polling</span>} s="WebSocket not used" />
      </div>
      <div className="cols-2 section">
        <Card title="Alerts">
          <State loading={al.loading && !al.data} error={al.error}>
            <DataTable rows={al.data} maxHeight={320} empty="No alerts." cols={[
              { key: "ts", label: "Time", render: (r) => time(r.ts).slice(5, 16) },
              { key: "level", label: "Level", render: (r) => <span className={r.level === "CRITICAL" || r.level === "ERROR" ? "neg" : "warn"}>{r.level}</span> },
              { key: "code", label: "Code" }, { key: "message", label: "Message" }]} />
          </State>
        </Card>
        <Card title="Recent errors">
          <DataTable rows={c.latest_errors} maxHeight={320} empty="No errors." cols={[
            { key: "ts", label: "Time", render: (r) => time(r.ts).slice(5, 16) }, { key: "source", label: "Source" }, { key: "message", label: "Message" }]} />
        </Card>
      </div>
      <div className="section">
        <Card title="Data completeness" right={<a className="btn" href="/metrics" target="_blank" rel="noreferrer">Prometheus metrics</a>}>
          <State loading={dq.loading && !dq.data} error={dq.error}>
            <DataTable rows={Object.entries(dq.data || {}).filter(([k]) => k !== "instrument_quality").map(([k, v]: any) => ({ dataset: k, ...v,
              complete: v.status?.complete, partial: v.status?.partial, failed: v.status?.failed }))} cols={[
              { key: "dataset", label: "Dataset" }, { key: "symbols", label: "Symbols", num: true }, { key: "windows", label: "Windows", num: true },
              { key: "complete", label: "Complete", num: true }, { key: "failed", label: "Failed", num: true },
              { key: "rows_valid", label: "Valid rows", num: true }, { key: "missing_intervals", label: "Missing intervals", num: true },
              { key: "outside_session_quarantined", label: "Quarantined", num: true }]} />
            {iq && <div className={`banner ${iq.ok ? "" : "bad"}`} style={{ marginTop: 12 }}>
              Instrument master: {iq.contracts} NIFTY F&amp;O contracts · lot sizes {JSON.stringify(iq.lot_sizes)} · {iq.symbol_mapping_errors} mapping errors · {iq.invalid_or_past_expiry} invalid expiries
            </div>}
          </State>
        </Card>
      </div>
    </State>
  );
}
