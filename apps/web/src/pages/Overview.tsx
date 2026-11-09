import { useMemo, useState } from "react";
import { api } from "../api";
import { Chart } from "../components/Chart";
import { Card, Stat, State } from "../components/ui";
import { inr, num, pct, time, tone } from "../format";
import { usePoll } from "../hooks";

export default function Overview({ mode, theme }: { mode: string; theme: string }) {
  const sum = usePoll(() => api.get(`/api/summary?mode=${mode}`), 10000, [mode]);
  const broker = usePoll(() => api.get("/api/broker"), 60000);
  const pnl = usePoll(() => api.get(`/api/pnl?mode=${mode}`), 30000, [mode]);
  const [busy, setBusy] = useState(false);
  const s = sum.data, b = broker.data;
  const virt = mode === "paper";
  const eq = useMemo(() => [{ name: "Equity", type: "area" as const, color: "--accent",
    data: (pnl.data?.equity || []).map((e: any) => ({ time: e.date, value: e.equity })) }], [pnl.data]);

  const kill = async (on: boolean) => {
    if (on) { const r = window.prompt("Reason for activating the kill switch?", "manual"); if (r == null) return;
      setBusy(true); try { await api.post("/api/kill", { reason: r }); } finally { setBusy(false); sum.reload(); } }
    else { const c = window.prompt("Type RESUME to clear the kill switch"); if (c !== "RESUME") return;
      setBusy(true); try { await api.post("/api/kill/off", { confirm: c }); } finally { setBusy(false); sum.reload(); } }
  };

  return (
    <>
      <Card title="Real Groww account" right={<span className="muted mono" style={{ fontSize: 12 }}>{b?.ok ? `updated ${time(b.fetched_at).slice(11)}` : ""}</span>}>
        <State loading={broker.loading && !b} error={broker.error}>
          {b && !b.ok ? <div className="banner bad">Groww balance unavailable — {b.error}</div> : (
            <div className="grid">
              <Stat k="Option-buy balance" v={inr(b?.option_buy_balance)} tone={(b?.option_buy_balance ?? 0) < 20000 ? "warn" : ""}
                    s={(b?.option_buy_balance ?? 0) < 20000 ? "below one lot (~₹12–19k) — live entries rejected" : "sufficient for 1 lot"} />
              <Stat k="Clear cash" v={inr(b?.clear_cash)} />
              <Stat k="Margin used" v={inr(b?.net_margin_used)} />
            </div>)}
        </State>
      </Card>

      <div className="section">
        <Card title={`Trading performance · ${mode} ledger`}>
          {virt && <div className="banner" style={{ marginBottom: 12 }}>Paper mode: capital, balance and P&amp;L are <b>simulated</b> from the paper ledger — not your Groww money.</div>}
          <State loading={sum.loading && !s} error={sum.error}>
            <div className="grid">
              <Stat k={`Capital${virt ? " (virtual)" : ""}`} v={inr(s?.capital)} />
              <Stat k={`Available${virt ? " (virtual)" : ""}`} v={inr(s?.available_balance)} />
              <Stat k="Open P&L" v={inr(s?.current_pnl)} tone={tone(s?.current_pnl)} />
              <Stat k="Today's P&L" v={inr(s?.today_pnl)} tone={tone(s?.today_pnl)} />
              <Stat k="Realised P&L" v={inr(s?.total_realized_pnl)} tone={tone(s?.total_realized_pnl)} />
              <Stat k="Trades" v={num(s?.total_trades, 0)} />
              <Stat k="Win rate" v={pct(s?.win_rate_pct)} />
              <Stat k="Profit factor" v={num(s?.profit_factor, 2)} />
              <Stat k="ROI" v={pct(s?.roi_pct)} tone={tone(s?.roi_pct)} />
              <Stat k="Max drawdown" v={inr(s?.max_drawdown)} tone={tone(s?.max_drawdown)} />
            </div>
          </State>
        </Card>
      </div>

      <div className="cols-3 section">
        <Card title="Equity curve">
          <State loading={pnl.loading && !pnl.data} error={pnl.error} empty={!(pnl.data?.equity || []).length && "No closed trades yet — the curve appears after the first trade."}>
            <Chart series={eq} theme={theme} />
          </State>
        </Card>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <Card title="Open position">
            {s?.position ? (
              <div className="kv mono">
                <span className="k">Symbol</span><span>{s.position.symbol}</span>
                <span className="k">Qty</span><span>{s.position.qty}</span>
                <span className="k">Entry</span><span>{num(s.position.entry)}</span>
                <span className="k">Bid / Ask</span><span>{num(s.position.bid)} / {num(s.position.ask)}</span>
                <span className="k">Stop / Target</span><span>{num(s.position.stop)} / {num(s.position.target)}</span>
                <span className="k">MTM</span><span className={tone(s.position.mtm)}>{inr(s.position.mtm)}</span>
              </div>) : <div className="muted">No open position</div>}
          </Card>
          <Card title="Kill switch">
            {s?.kill_switch ? (
              <><div className="banner bad">ACTIVE — {s.kill_switch.reason} <span className="muted">({s.kill_switch.source}, {time(s.kill_switch.at)})</span></div>
                <button className="btn" style={{ marginTop: 10 }} disabled={busy} onClick={() => kill(false)}>Resume trading</button></>
            ) : (
              <><div className="muted" style={{ marginBottom: 10 }}>Inactive. Activating exits any open position and blocks new entries.</div>
                <button className="btn danger" disabled={busy} onClick={() => kill(true)}>Activate kill switch</button></>
            )}
          </Card>
        </div>
      </div>
    </>
  );
}
