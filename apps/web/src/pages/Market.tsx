import { api } from "../api";
import { Card, Stat, State } from "../components/ui";
import { ago, inr, num } from "../format";
import { usePoll } from "../hooks";

export default function Market() {
  const m = usePoll(() => api.get("/api/market"), 5000);
  const k = m.data?.market || {}, ps = m.data?.position, d = m.data?.last_decision;
  const g = ps?.greeks || {};
  return (
    <>
      <div className="muted" style={{ marginBottom: 10, fontSize: 12 }}>
        {m.data?.updated_at ? `Snapshot written by the worker ${ago(m.data.updated_at)} (each completed candle)` : "No snapshot yet — start the worker (scripts/run_trader.py) during market hours."}
      </div>
      <State loading={m.loading && !m.data} error={m.error}>
        <div className="grid">
          <Stat k="NIFTY spot" v={num(k.spot)} />
          <Stat k="NIFTY futures" v={num(k.futures)} />
          <Stat k="India VIX" v={num(k.vix)} />
          <Stat k="Futures VWAP" v={num(k.futures_vwap)} />
          <Stat k="EMA 9 / 20" v={`${num(k.ema9, 0)} / ${num(k.ema20, 0)}`} />
          <Stat k="ATR (14)" v={num(k.atr14)} />
          <Stat k="Regime" v={k.regime?.regime || "—"} />
          <Stat k="Opening range" v={k.or_high ? `${num(k.or_low, 0)} – ${num(k.or_high, 0)}` : "—"} />
        </div>
        <div className="cols-2 section">
          <Card title="Strategy state (latest evaluation)">
            {(k.strategy_reasons || []).length ? <ul className="rules">{k.strategy_reasons.map((r: string, i: number) => <li key={i} className="mono">{r}</li>)}</ul>
              : <div className="muted">Waiting for the first evaluation.</div>}
          </Card>
          <Card title="Open position · Greeks">
            {ps ? (
              <div className="kv mono">
                <span className="k">Contract</span><span>{ps.symbol}</span>
                <span className="k">Bid / Ask</span><span>{num(ps.bid)} / {num(ps.ask)}</span>
                <span className="k">Δ / Γ</span><span>{num(g.delta, 3)} / {num(g.gamma, 4)}</span>
                <span className="k">Θ / Vega</span><span>{num(g.theta)} / {num(g.vega)}</span>
                <span className="k">IV</span><span>{num(g.iv)}</span>
                <span className="k">MTM</span><span>{inr(ps.mtm)}</span>
              </div>) : <div className="muted">No open position.</div>}
          </Card>
        </div>
        <div className="section">
          <Card title="Last trade decision">
            {d ? (
              <>
                <div className={`banner ${d.approved ? "" : "warn"}`} style={{ marginBottom: 10 }}>
                  {d.approved ? "APPROVED" : "REJECTED"} · {d.contract?.trading_symbol || "—"} · stage {d.stage}
                  {d.failures?.length ? ` · ${d.failures.join(", ")}` : ""}
                </div>
                <div className="kv mono">
                  <span className="k">Signal</span><span>{d.signal?.direction} at {d.signal?.timestamp?.slice(11, 16)} ({(d.signal?.reason_codes || []).join(", ")})</span>
                  <span className="k">Quote</span><span>bid {num(d.quote?.bid)} / ask {num(d.quote?.ask)}</span>
                  <span className="k">Limit / Stop / Target</span><span>{num(d.limit)} / {num(d.stop)} / {num(d.target)}</span>
                  <span className="k">Planned loss</span><span>{inr(d.planned_loss)} (worst case {inr(d.worst_case_loss)})</span>
                </div>
              </>) : <div className="muted">No decision yet today.</div>}
          </Card>
        </div>
      </State>
    </>
  );
}
