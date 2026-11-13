import { useMemo, useState } from "react";
import { api } from "../api";
import { Chart, Series } from "../components/Chart";
import { Col, DataTable } from "../components/DataTable";
import { Card, Chip, Seg, Stat, State } from "../components/ui";
import { inr, pct, tone } from "../format";
import { usePoll } from "../hooks";

const LABEL: Record<string, string> = {
  logistic: "Logistic", knn: "KNN", random_forest: "Random Forest", extra_trees: "Extra Trees", hist_gb: "Hist GB",
  xgboost: "XGBoost", lightgbm: "LightGBM", catboost: "CatBoost", mlp: "MLP", lstm: "LSTM", gru: "GRU", cnn: "1-D CNN",
  transformer: "Transformer", ensemble: "Ensemble",
};
const name = (m: string) => LABEL[m] || m;
const PALETTE = ["#2f6fed", "#e8590c", "#2b8a3e", "#9c36b5", "#c92a2a", "#0c8599", "#e67700", "#5c7cfa"];
type Mode = "top10" | "all_buy";
const signed = (v: unknown, d = 2) => <span className={tone(v)}>{typeof v === "number" ? `${v > 0 ? "+" : ""}${v.toFixed(d)}%` : "—"}</span>;
const rs = (v: unknown) => <span className={tone(v)}>{typeof v === "number" ? `${v > 0 ? "+" : ""}${inr(v)}` : "—"}</span>;

/** Simulated Rs 1 lakh portfolio for every ML / DL model, following the daily signals (no real money). */
export default function Virtual({ theme, initialMode = "top10" }: { theme: string; initialMode?: Mode }) {
  const v = usePoll(() => api.get("/api/stocks/virtual"), 5 * 60_000);
  const d = v.data;
  const [sel, setSel] = useState<string>("");
  const [shown, setShown] = useState<string[] | null>(null);
  const [mode, setMode] = useState<Mode>(initialMode);
  const [hq, setHq] = useState("");
  const allBuy = mode === "all_buy";
  const ms: Record<string, any> = d?.portfolios?.[mode] || (mode === "top10" ? d?.models : null) || {};
  const metric = allBuy ? "gross_return_pct" : "return_pct";      // all-BUY basket: judge the signals before charges

  const board = useMemo(() => Object.values(ms).sort((a: any, b: any) => b[metric] - a[metric])
    .map((m: any, i: number) => ({ ...m, rank: i + 1 })), [ms, metric]);
  const chosen: string = d?.chosen || "xgboost";
  const active = sel || chosen;
  const cur = ms[active];
  const holdings = useMemo(() => {
    const ql = hq.trim().toLowerCase();
    return (cur?.holdings || []).filter((h: any) => !ql || h.sym.toLowerCase().includes(ql) || (h.name || "").toLowerCase().includes(ql));
  }, [cur, hq]);
  const plot = shown ?? [chosen, "ensemble", board[0]?.model].filter((x, i, a) => x && a.indexOf(x) === i);
  const toggle = (m: string) => setShown((s) => { const x = s ?? plot; return x.includes(m) ? x.filter((y) => y !== m) : [...x, m]; });

  const series: Series[] = useMemo(() => {
    if (!d) return [];
    const line = (n: string, c: string, pts: any[]) => ({ name: n, color: c, data: pts.map((p: any) => ({ time: p.date, value: p.value })) });
    const out: Series[] = plot.filter((m) => ms[m]).map((m, i) => line(name(m), PALETTE[i % PALETTE.length], allBuy ? ms[m].gross_curve : ms[m].curve));
    out.push(line("NIFTY 50", "--muted", d.benchmarks?.nifty?.curve || []));
    out.push(line("All liquid stocks", "--warn", d.benchmarks?.all?.curve || []));
    return out;
  }, [d, plot, ms, allBuy]);

  const best = board[0];
  const nifty = d?.benchmarks?.nifty?.return_pct;
  const beat = board.filter((m: any) => typeof nifty === "number" && m[metric] > nifty).length;

  const boardCols: Col<any>[] = [
    { key: "rank", label: "#", num: true },
    { key: "model", label: "Model", render: (r) => <b>{name(r.model)}{r.model === chosen ? " ⭐" : ""}</b> },
    { key: "family", label: "Type" },
    { key: "value", label: "Value now", num: true, render: (r) => inr(r.value) },
    { key: "gross_return_pct", label: "Before charges", num: true, render: (r) => signed(r.gross_return_pct) },
    { key: "return_pct", label: "After buy charges", num: true, render: (r) => signed(r.return_pct) },
    { key: "pnl", label: "Profit / loss", num: true, render: (r) => rs(r.pnl) },
    { key: "vs", label: `vs NIFTY${allBuy ? " (before charges)" : ""}`, num: true, value: (r) => r[metric] - (nifty ?? 0), render: (r) => signed(typeof nifty === "number" ? r[metric] - nifty : null) },
    { key: "net_if_sold", label: "If sold today (net)", num: true, render: (r) => rs(r.net_if_sold) },
    { key: "charges_paid", label: "Charges paid", num: true, render: (r) => inr(r.charges_paid) },
    { key: "n_holdings", label: "Stocks", num: true },
    { key: "cash", label: "Cash", num: true, render: (r) => inr(r.cash) },
    { key: "plot", label: "Chart", render: (r) => (
      <input type="checkbox" checked={plot.includes(r.model)} onClick={(e) => e.stopPropagation()} onChange={() => toggle(r.model)} title="Show on the chart" />) },
  ];
  const holdCols: Col<any>[] = [
    { key: "sym", label: "Symbol", render: (r) => <span className="mono">{r.sym}</span> },
    { key: "name", label: "Company" },
    { key: "qty", label: "Qty", num: true, render: (r) => (Number.isInteger(r.qty) ? r.qty : r.qty.toFixed(3)) },
    { key: "buy_price", label: "Bought at", num: true, render: (r) => inr(r.buy_price, 2) },
    { key: "last", label: "Last close", num: true, render: (r) => inr(r.last, 2) },
    { key: "value", label: "Value", num: true, render: (r) => inr(r.value) },
    { key: "pnl", label: "P&L (after buy charges)", num: true, render: (r) => rs(r.pnl) },
    { key: "ret_pct", label: "Return", num: true, render: (r) => signed(r.ret_pct) },
    { key: "entry_date", label: "Bought on", render: (r) => <span className="mono">{r.entry_date}</span> },
    { key: "p_buy", label: "P(BUY) at entry", num: true, render: (r) => pct(r.p_buy * 100, 1) },
  ];
  const tradeCols: Col<any>[] = [
    { key: "date", label: "Date", render: (r) => <span className="mono">{r.date}</span> },
    { key: "side", label: "Side", render: (r) => <Chip tone={r.side === "BUY" ? "ok" : "bad"}>{r.side}</Chip> },
    { key: "sym", label: "Symbol", render: (r) => <span className="mono">{r.sym}</span> },
    { key: "qty", label: "Qty", num: true },
    { key: "price", label: "Price (open)", num: true, render: (r) => inr(r.price, 2) },
    { key: "pnl", label: "Realised P&L", num: true, render: (r) => (r.pnl == null ? "—" : rs(r.pnl)) },
    { key: "signal_date", label: "Signals of", render: (r) => <span className="mono">{r.signal_date}</span> },
  ];

  return (
    <State loading={v.loading && !d} error={v.error} empty={d && d.state !== "running" && (d.msg || "Not started yet")}>
      <div className="banner warn">
        Virtual money only — every ML / DL model gets a simulated <b>₹1,00,000</b> and follows its own signals from the{" "}
        <b className="mono">{d?.start_signal}</b> close. Nothing is bought or sold for real.
      </div>

      <div className="section" style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <Seg<Mode> value={mode} onChange={(m) => { setMode(m); setHq(""); }} options={[
          { v: "top10", label: "Top-10 BUY portfolio" }, { v: "all_buy", label: "All BUY stocks" }]} />
        <span className="muted" style={{ fontSize: 13 }}>
          {allBuy ? "₹1,00,000 split equally across EVERY stock the model rates BUY (hundreds of stocks)."
                  : "₹1,00,000 in the model's 10 highest-ranked BUY stocks (₹10,000 each, whole shares)."}
        </span>
      </div>
      {allBuy && (
        <div className="banner bad section" style={{ fontSize: 13 }}>
          <b>Read “Before charges” for this view.</b> With ₹1 lakh over hundreds of stocks each position is only ~₹100–200, so it uses
          fractional shares (not possible on Groww), and Groww's minimum charges — ₹5 brokerage per order and ₹15.93 DP charge per stock
          sold — eat a big share of such tiny positions (see “After buy charges” / “If sold today”). It shows how good the BUY signals are
          as a whole; to hold every BUY stock for real you would need roughly ₹20,000 per stock (₹1 crore+).
        </div>)}

      <div className="grid section">
        <Stat k="Started" v={<span className="mono">{d?.first_buy}</span>} s={`signals of ${d?.start_signal} · bought at the open`} />
        <Stat k="Marked at" v={<span className="mono">{d?.last_mark}</span>} s={`${d?.days ?? "—"} trading days · next rebalance ${d?.next_rebalance ?? "—"}`} />
        <Stat k={`Best model${allBuy ? " (before charges)" : ""}`} v={best ? name(best.model) : "—"} tone={tone(best?.[metric])} s={best ? <>{signed(best[metric])} · {best.n_holdings} stocks</> : ""} />
        <Stat k={`${name(chosen)} ⭐ (main model)`} v={signed(ms[chosen]?.[metric])} s={ms[chosen] ? <>{allBuy ? `${ms[chosen].n_holdings} stocks · after charges ` : `${inr(ms[chosen].value)} · `}{allBuy ? signed(ms[chosen].return_pct) : <>net if sold {rs(ms[chosen].net_if_sold)}</>}</> : ""} />
        <Stat k="NIFTY 50 / all stocks" v={<>{signed(nifty)} / {signed(d?.benchmarks?.all?.return_pct)}</>} s={`${beat} of ${board.length} models ahead of NIFTY`} />
      </div>

      <div className="section">
        <Card title={`Value of ₹1,00,000 · each model vs NIFTY 50 and all liquid stocks${allBuy ? " · before charges" : ""}`}>
          <Chart series={series} theme={theme} height={320} />
          <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
            Tick “Chart” in the table to add or remove models. {allBuy
              ? "Lines are BEFORE charges here (signal quality); the after-charges numbers are in the table."
              : "Values are after buying charges; selling charges are in “If sold today”."}
          </div>
        </Card>
      </div>

      <div className="section">
        <Card title={`Leaderboard · ${allBuy ? "all BUY stocks" : "top-10 BUY"} · 14 virtual portfolios`}>
          <DataTable rows={board} cols={boardCols} maxHeight={900} onRowClick={(r: any) => setSel(r.model)} />
          <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
            Click a row to see its holdings and trades below. Rules (same as the backtest): buy the top-10 BUY stocks (₹ value ÷ 10 each,
            whole shares) at the open; every {d?.rebalance_every_days} trading days sell stocks that turned SELL, keep the rest, refill free
            slots from the newest top BUY. Groww charges + slippage included. A few days of results are mostly noise — judge after a month or more.
          </div>
        </Card>
      </div>

      <div className="section">
        <Card title={`Holdings · ${name(active)}${active === chosen ? " ⭐" : ""} · ${cur?.n_holdings ?? 0} stocks`} right={
          <span style={{ display: "flex", gap: 10, alignItems: "center", textTransform: "none" }}>
            <input placeholder="Search symbol or company" value={hq} onChange={(e) => setHq(e.target.value)} style={{ width: 200 }} />
            <span className="muted">cash {inr(cur?.cash)}</span>
          </span>}>
          <DataTable rows={holdings} cols={holdCols} empty="No holdings" maxHeight={allBuy ? 520 : 460} />
        </Card>
      </div>

      <div className="section">
        <Card title={`Trades · ${name(active)}`}>
          <DataTable rows={[...(cur?.trade_log || [])].reverse()} cols={tradeCols} maxHeight={460}
            empty={allBuy ? `${cur?.buys ?? 0} stocks bought on ${d?.first_buy} (see holdings) · no sells yet — sells happen at the rebalance on ${d?.next_rebalance}` : "No trades yet"} />
        </Card>
      </div>
    </State>
  );
}
