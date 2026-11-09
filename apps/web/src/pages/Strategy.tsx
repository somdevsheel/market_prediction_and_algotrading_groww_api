import { useMemo } from "react";
import { api } from "../api";
import { Chart } from "../components/Chart";
import { DataTable } from "../components/DataTable";
import { Card, Chip, Stat, State } from "../components/ui";
import { inr, num, pct, time, tone } from "../format";
import { usePoll } from "../hooks";

export default function Strategy({ theme }: { theme: string }) {
  const st = usePoll(() => api.get("/api/strategies"));
  const bts = usePoll(() => api.get("/api/backtests"));
  const active = (st.data || []).find((x: any) => x.active);
  const runs = (bts.data || []).filter((r: any) => active && (r.experiment === active.strategy_id || r.experiment?.startsWith("retest30_cheapIV")));
  // evidence = longest-period conservative-cost run (newest first among equals), not a short ad-hoc test
  const ranked = [...runs].sort((a: any, b: any) => (a.start || "").localeCompare(b.start || "") || (b.run_id || "").localeCompare(a.run_id || ""));
  const best = ranked.find((r: any) => r.scenario === "conservative") || ranked[0];
  const detail = usePoll(() => (best ? api.get(`/api/backtests/${best.run_id}`) : Promise.resolve(null)), 0, [best?.run_id]);
  const eq = useMemo(() => [{ name: "Equity", type: "area" as const, color: "--pos",
    data: (detail.data?.equity || []).map((e: any) => ({ time: e.date, value: e.equity })) }], [detail.data]);
  const years = Object.entries(detail.data?.metrics?.by_year || {}).map(([y, v]: any) => ({ year: y, ...v }));
  const p = active?.params || {};
  const tones: Record<string, "ok" | "warn" | "bad" | ""> = { APPROVED_FOR_LIVE: "ok", PAPER_TRADING: "warn", BACKTESTED: "warn", RESEARCH: "", RETIRED: "bad" };

  return (
    <State loading={st.loading && !st.data} error={st.error} empty={!active && "No active strategy configured."}>
      {active && (
        <>
          <Card title={<>{active.strategy_id} <span className="muted" style={{ textTransform: "none" }}>v{active.version}</span></>}
                right={<Chip tone={tones[active.status] || ""}>{active.status}</Chip>}>
            <div className="cols-2">
              <div>
                <div style={{ marginBottom: 8 }}>{active.description}</div>
                <ul className="rules">
                  <li>Opening range: first <b>{p.or_minutes} min</b> (09:15–{p.or_minutes === 30 ? "09:45" : "…"}); skip if range &gt; <b>{p.orb_filters?.max_range_pct}%</b> of open</li>
                  <li>Entry: 1-min close beyond the range → close back inside → close beyond the same side again, within <b>{p.orb_filters?.max_gap_minutes} min</b></li>
                  <li>Contract: 1 strike ITM CE (bull) / PE (bear), nearest non-expiry-day weekly</li>
                  <li>Filter: buy only if option IV ≤ <b>{active.filters?.contract_iv_max_over_rv}×</b> GARCH volatility forecast</li>
                  <li>Exit: NIFTY touches the opposite range boundary · ₹ risk-budget premium stop · 15:15 square-off</li>
                </ul>
              </div>
              <div className="kv">
                <span className="k">Config hash</span><span className="mono">{active.config_hash}</span>
                <span className="k">Status history</span><span>{active.history?.length ? active.history.map((h: any) => `${h.status} (${time(h.at).slice(0, 10)})`).join(" → ") : "from config (no transitions yet)"}</span>
                <span className="k">Go-live needs</span><span>funds ≥ ₹25k · passing live smoke test · governance approval · same-day approval</span>
              </div>
            </div>
          </Card>

          <div className="section">
            <Card title="Backtest evidence" right={best && <span className="muted mono" style={{ fontSize: 12 }}>{best.run_id}</span>}>
              <State loading={bts.loading && !bts.data} error={bts.error} empty={!best && "No backtest yet — run: python3 scripts/backtest_strategy.py"}>
                {best && (
                  <>
                    <div className="grid">
                      <Stat k="Net P&L" v={inr(best.net_pnl)} tone={tone(best.net_pnl)} s={`${best.scenario} costs · ${best.start} → ${best.end}`} />
                      <Stat k="Trades" v={num(best.trades, 0)} />
                      <Stat k="Win rate" v={pct(best.win_rate_pct)} />
                      <Stat k="Profit factor" v={num(best.profit_factor, 2)} />
                      <Stat k="Max drawdown" v={inr(best.max_drawdown)} tone={tone(best.max_drawdown)} />
                      <Stat k="Out-of-sample (last 30%)" v={inr(best.oos_net)} tone={tone(best.oos_net)} />
                    </div>
                    <div className="cols-3 section">
                      <div><Chart series={eq} theme={theme} /></div>
                      <DataTable rows={years} cols={[
                        { key: "year", label: "Year" },
                        { key: "trades", label: "Trades", num: true },
                        { key: "net_pnl", label: "Net", num: true, render: (r) => <span className={tone(r.net_pnl)}>{inr(r.net_pnl)}</span> },
                        { key: "win_rate", label: "Win %", num: true },
                      ]} />
                    </div>
                  </>)}
              </State>
            </Card>
          </div>
          <div className="section banner warn">
            Selection bias: this was the best of many tested variants, and 10 trades produced more than its total profit.
            Expect live results to be weaker; trade 1 lot, every day, and judge over months.
          </div>
        </>
      )}
    </State>
  );
}
