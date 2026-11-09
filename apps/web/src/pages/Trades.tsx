import { api } from "../api";
import { DataTable } from "../components/DataTable";
import { Card, State } from "../components/ui";
import { inr, num, time, tone } from "../format";
import { usePoll } from "../hooks";

export default function Trades({ mode }: { mode: string }) {
  const tr = usePoll(() => api.get(`/api/trades?mode=${mode}`), 15000, [mode]);
  const od = usePoll(() => api.get("/api/orders"), 15000);
  const dc = usePoll(() => api.get("/api/decisions"), 30000);
  const pv = usePoll(() => api.get("/api/paper_vs_backtest"), 60000);
  const money = (k: string) => ({ key: k, label: k.replace("_", " "), num: true, render: (r: any) => <span className={tone(r[k])}>{inr(r[k])}</span> });
  return (
    <>
      <Card title="Trade book" right={<a className="btn" href={`/api/export/trades.csv?mode=${mode}`}>Export CSV</a>}>
        <State loading={tr.loading && !tr.data} error={tr.error}>
          <DataTable rows={tr.data} empty={`No ${mode} trades yet.`} cols={[
            { key: "trading_day", label: "Day" }, { key: "trading_symbol", label: "Contract" }, { key: "option_type", label: "Type" },
            { key: "status", label: "Status" }, { key: "qty", label: "Qty", num: true },
            { key: "entry_time", label: "Entry", render: (r) => time(r.entry_time).slice(11, 16) }, { key: "entry_price", label: "Entry ₹", num: true },
            { key: "exit_time", label: "Exit", render: (r) => time(r.exit_time).slice(11, 16) }, { key: "exit_price", label: "Exit ₹", num: true },
            { key: "exit_reason", label: "Exit reason" }, money("charges"), money("gross_pnl"), money("net_pnl"),
            { key: "roi_pct", label: "ROI %", num: true },
          ]} />
        </State>
      </Card>
      <div className="cols-2 section">
        <Card title="Orders">
          <State loading={od.loading && !od.data} error={od.error}>
            <DataTable rows={od.data} maxHeight={360} empty="No orders yet." cols={[
              { key: "created_at", label: "Time", render: (r) => time(r.created_at).slice(5, 16) }, { key: "role", label: "Role" },
              { key: "side", label: "Side" }, { key: "trading_symbol", label: "Contract" }, { key: "qty", label: "Qty", num: true },
              { key: "order_type", label: "Type" }, { key: "price", label: "Price", num: true }, { key: "state", label: "State" },
              { key: "filled_qty", label: "Filled", num: true }, { key: "avg_price", label: "Avg", num: true },
            ]} />
          </State>
        </Card>
        <Card title="Risk decisions">
          <State loading={dc.loading && !dc.data} error={dc.error}>
            <DataTable rows={(dc.data || []).map((d: any) => ({ ts: d.ts, approved: d.approved, stage: d.payload.stage,
              reason: d.payload.reason || (d.payload.failures || []).join(", "), contract: d.payload.contract?.trading_symbol,
              planned: d.payload.planned_loss }))} maxHeight={360} empty="No decisions yet." cols={[
              { key: "ts", label: "Time", render: (r) => time(r.ts).slice(5, 16) },
              { key: "approved", label: "Result", render: (r) => <span className={r.approved ? "pos" : "warn"}>{r.approved ? "approved" : "rejected"}</span> },
              { key: "stage", label: "Stage" }, { key: "reason", label: "Reason" }, { key: "contract", label: "Contract" },
              { key: "planned", label: "Planned loss", num: true, render: (r) => inr(r.planned) },
            ]} />
          </State>
        </Card>
      </div>
      <div className="section">
        <Card title="Paper execution vs backtest assumptions">
          <State loading={pv.loading && !pv.data} error={pv.error}>
            {pv.data?.trades ? (
              <div className="kv mono">
                <span className="k">Closed paper trades</span><span>{pv.data.trades}</span>
                <span className="k">Avg entry cost vs mid</span><span>{num(pv.data.avg_entry_cost_vs_mid_pts)} pts</span>
                <span className="k">Avg exit slippage</span><span>{num(pv.data.avg_exit_slip_vs_expected_pts)} pts</span>
                <span className="k">Backtest assumptions</span><span>{JSON.stringify(pv.data.assumed_cost_per_side_pts)}</span>
                <span className="k">Closest scenario</span><span>{pv.data.closest_scenario}</span>
              </div>) : <div className="muted">{pv.data?.note || "No closed paper trades yet."}</div>}
          </State>
        </Card>
      </div>
    </>
  );
}
