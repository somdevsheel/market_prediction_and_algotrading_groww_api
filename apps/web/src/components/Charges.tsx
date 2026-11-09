import { useState } from "react";
import { api } from "../api";
import { DataTable } from "./DataTable";
import { Card, Stat, State } from "./ui";
import { inr, num, tone } from "../format";
import { usePoll } from "../hooks";

/** Round-trip charge calculator using the exact rates the system applies (config/instruments.yaml). */
export function ChargesCard({ avgPerTrade, total, impactPct }: { avgPerTrade?: number; total?: number; impactPct?: number }) {
  const [f, setF] = useState({ entry: "250", exit: "260", lots: "1" });
  const q = usePoll(() => api.get(`/api/charges?entry=${+f.entry || 0}&exit=${+f.exit || 0}&lots=${+f.lots || 1}`), 0, [f.entry, f.exit, f.lots]);
  const d = q.data;
  const set = (k: string) => (e: any) => setF({ ...f, [k]: e.target.value });
  return (
    <Card title="Charges (per round trip: 1 buy + 1 sell)">
      {avgPerTrade != null && (
        <div className="grid" style={{ marginBottom: 14 }}>
          <Stat k="Average charges per trade (backtest)" v={inr(avgPerTrade, 2)} />
          <Stat k="Total charges (backtest)" v={inr(total)} />
          <Stat k="Charges as % of gross P&L" v={impactPct != null ? `${num(impactPct, 1)}%` : "—"} s="how much of the strategy's edge goes to costs" />
        </div>
      )}
      <div className="form" style={{ marginBottom: 12 }}>
        <label>Buy premium ₹<input value={f.entry} onChange={set("entry")} size={6} /></label>
        <label>Sell premium ₹<input value={f.exit} onChange={set("exit")} size={6} /></label>
        <label>Lots<input value={f.lots} onChange={set("lots")} size={3} /></label>
        {d && <span className="muted" style={{ fontSize: 12 }}>qty {d.qty} (lot {d.lot_size}) · buy ₹{num(d.buy_value, 0)} · sell ₹{num(d.sell_value, 0)}</span>}
      </div>
      <State loading={q.loading && !d} error={q.error}>
        {d && (
          <>
            <DataTable rows={d.components} cols={[
              { key: "component", label: "Charge", render: (r) => <b>{r.component.toUpperCase()}</b> },
              { key: "amount", label: "Amount", num: true, render: (r) => inr(r.amount, 2) },
              { key: "rule", label: "How it is calculated" },
              { key: "paid_to", label: "Paid to" },
            ]} />
            <div className="grid" style={{ marginTop: 12 }}>
              <Stat k="Gross P&L" v={inr(d.gross_pnl, 2)} tone={tone(d.gross_pnl)} />
              <Stat k="Total charges" v={inr(d.total_charges, 2)} tone="neg" />
              <Stat k="Net P&L" v={inr(d.net_pnl, 2)} tone={tone(d.net_pnl)} />
              <Stat k="Break-even move" v={`${num(d.breakeven_move_points, 2)} pts`} s="premium rise needed just to cover charges" />
            </div>
            <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>{d.note} Backtests also deduct spread/slippage (≈0.5% of premium per side in the base scenario).</div>
          </>
        )}
      </State>
    </Card>
  );
}
