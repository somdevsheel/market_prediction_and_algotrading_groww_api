import { useMemo, useState } from "react";
import { api } from "../api";
import { Chart } from "../components/Chart";
import { DataTable } from "../components/DataTable";
import { Card, Seg, State } from "../components/ui";
import { inr, tone } from "../format";
import { usePoll } from "../hooks";

type Period = "daily" | "weekly" | "monthly" | "yearly";

export default function Pnl({ mode, theme }: { mode: string; theme: string }) {
  const p = usePoll(() => api.get(`/api/pnl?mode=${mode}`), 30000, [mode]);
  const [per, setPer] = useState<Period>("daily");
  const d = p.data;
  const eq = useMemo(() => [{ name: "Equity", type: "area" as const, color: "--accent", data: (d?.equity || []).map((e: any) => ({ time: e.date, value: e.equity })) }], [d]);
  const dd = useMemo(() => [{ name: "Drawdown", type: "area" as const, color: "--neg", data: (d?.equity || []).map((e: any) => ({ time: e.date, value: e.drawdown })) }], [d]);
  const daily = useMemo(() => [{ name: "Daily P&L", type: "histogram" as const, data: (d?.daily || []).map((x: any) => ({ time: x.period, value: x.net_pnl, color: x.net_pnl >= 0 ? getComputedStyle(document.documentElement).getPropertyValue("--pos").trim() : getComputedStyle(document.documentElement).getPropertyValue("--neg").trim() })) }], [d, theme]);
  const obj = (o: any) => Object.entries(o || {}).map(([k, v]: any) => ({ key: k, ...v }));
  const pnlCol = { key: "net_pnl", label: "Net P&L", num: true, render: (r: any) => <span className={tone(r.net_pnl)}>{inr(r.net_pnl)}</span> };
  const empty = !(d?.equity || []).length && `No closed ${mode} trades yet.`;
  return (
    <State loading={p.loading && !d} error={p.error} empty={empty}>
      <div className="cols-2">
        <Card title="Equity"><Chart series={eq} theme={theme} /></Card>
        <Card title="Drawdown"><Chart series={dd} theme={theme} /></Card>
      </div>
      <div className="section"><Card title="Daily P&L"><Chart series={daily} theme={theme} height={220} /></Card></div>
      <div className="cols-2 section">
        <Card title="Periods" right={<Seg value={per} onChange={setPer} options={[{ v: "daily", label: "Day" }, { v: "weekly", label: "Week" }, { v: "monthly", label: "Month" }, { v: "yearly", label: "Year" }]} />}>
          <DataTable rows={[...(d?.[per] || [])].reverse()} cols={[{ key: "period", label: "Period" }, pnlCol]} maxHeight={360} />
        </Card>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <Card title="CE vs PE"><DataTable rows={obj(d?.by_option_type)} cols={[{ key: "key", label: "Type" }, { key: "trades", label: "Trades", num: true }, pnlCol, { key: "win_rate", label: "Win %", num: true }]} /></Card>
          <Card title="By strategy"><DataTable rows={obj(d?.by_strategy)} cols={[{ key: "key", label: "Strategy" }, { key: "trades", label: "Trades", num: true }, pnlCol, { key: "win_rate", label: "Win %", num: true }]} /></Card>
        </div>
      </div>
    </State>
  );
}
