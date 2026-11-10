import { useMemo, useState } from "react";
import { api } from "../api";
import { Chart } from "../components/Chart";
import { Col, DataTable } from "../components/DataTable";
import { StockItem, StockSearch } from "../components/StockSearch";
import { Card, Chip, Seg, Stat, State } from "../components/ui";
import { inr, num, pct, tone } from "../format";
import { usePoll } from "../hooks";

type Sig = "" | "BUY" | "HOLD" | "SELL";
const SIG_TONE: Record<string, "ok" | "bad" | ""> = { BUY: "ok", SELL: "bad", HOLD: "" };
const LABEL: Record<string, string> = {
  logistic: "Logistic", knn: "KNN", random_forest: "Random Forest", extra_trees: "Extra Trees", hist_gb: "Hist GB",
  xgboost: "XGBoost", lightgbm: "LightGBM", catboost: "CatBoost", mlp: "MLP", lstm: "LSTM", gru: "GRU", cnn: "1-D CNN",
  transformer: "Transformer", ensemble: "Ensemble", nifty: "NIFTY 50", ew: "All stocks (equal wt)",
  "NIFTY50 (price)": "NIFTY 50 (price)", "all universe equal-wt (gross)": "All stocks equal-wt (gross)",
};
const name = (m: string) => LABEL[m] || m;
const FAM: Record<string, string> = { logistic: "ML", knn: "ML", random_forest: "ML", extra_trees: "ML", hist_gb: "ML", xgboost: "ML",
  lightgbm: "ML", catboost: "ML", mlp: "DL", lstm: "DL", gru: "DL", cnn: "DL", transformer: "DL", ensemble: "ML+DL" };
const COLORS = ["--accent", "--pos", "--warn", "--neg", "--muted"];

type Marker = { time: string; position: "aboveBar" | "belowBar"; shape: "arrowUp" | "arrowDown"; color: string; text?: string };

function Forecast({ sym, theme, onSym, stocks, markers, markerModel }: { sym: string; theme: string; onSym: (s: string) => void;
  stocks: StockItem[]; markers: Marker[]; markerModel: string }) {
  const fc = usePoll(() => (sym ? api.get(`/api/stocks/forecast?sym=${encodeURIComponent(sym)}`) : Promise.resolve(null)), 0, [sym]);
  const d = fc.data;
  const series = useMemo(() => {
    if (!d) return [];
    const hist = (d.history || []).map((h: any) => ({ time: h.date, value: h.close }));
    const m = d.horizons?.m1;
    const leg = (v: number) => [{ time: d.as_of, value: d.close }, { time: m.target_date, value: v }];
    return [
      { name: "Close", color: "--accent", data: hist, markers },
      ...(m ? [{ name: "1-month high", color: "--pos", data: leg(m.hi) },
               { name: "1-month middle", color: "--warn", data: leg(m.mid) },
               { name: "1-month low", color: "--neg", data: leg(m.lo) }] : []),
    ];
  }, [d, markers]);
  const H = ({ k }: { k: "d1" | "m1" }) => {
    const h = d?.horizons?.[k];
    if (!h) return null;
    return (
      <div className="card" style={{ padding: 12 }}>
        <div className="muted" style={{ fontSize: 12 }}>{k === "d1" ? "Tomorrow" : "In about 1 month"} · {h.target_date}</div>
        <div className="grid" style={{ marginTop: 8, gridTemplateColumns: "repeat(3, 1fr)" }}>
          <Stat k="Low" v={inr(h.lo, 2)} tone="neg" s={pct(h.lo_pct, 1)} />
          <Stat k="Middle" v={inr(h.mid, 2)} s={pct(h.mid_pct, 1)} />
          <Stat k="High" v={inr(h.hi, 2)} tone="pos" s={pct(h.hi_pct, 1)} />
        </div>
        <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
          Range held the actual price {pct(h.range_hit_rate_pct)} of the time (2024–26, unseen). Middle-value error {pct(h.mid_error_pct, 2)} vs
          {" "}{pct(h.same_price_error_pct, 2)} for “same as today”; direction right {pct(h.direction_correct_pct)}.
        </div>
      </div>
    );
  };
  return (
    <Card title={<>Price forecast · {d ? <><span className="mono">{d.sym}</span>&nbsp;{d.name || ""}&nbsp;· last close {inr(d.close, 2)} ({d.as_of})</> : sym || "pick a stock"}</>}
      right={<StockSearch items={stocks} onPick={onSym} placeholder="Search stock — type a letter" width={260} />}>
      <State loading={fc.loading && !fc.data} error={fc.error} empty={!sym && "Click any stock in the tables below, or search above."}>
        {d && (
          <>
            <div className="cols-2">
              <H k="d1" />
              <H k="m1" />
            </div>
            <div className="section"><Chart series={series} theme={theme} height={300} /></div>
            {markers.length > 0 && <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
              <span className="pos">▲ B</span> = {name(markerModel)} said BUY, <span className="neg">▼ S</span> = SELL at that month-end (backtest, out-of-sample).</div>}
            <div className="banner" style={{ marginTop: 12, fontSize: 12 }}>{d.note}</div>
          </>
        )}
      </State>
    </Card>
  );
}

const SIG_CLS: Record<string, string> = { BUY: "pos", SELL: "neg", HOLD: "muted" };
const SigTxt = ({ v, short }: { v?: string | null; short?: boolean }) =>
  v ? <span className={SIG_CLS[v]} style={{ fontWeight: 600 }}>{short ? v[0] : v}</span> : <span className="muted">—</span>;

function ModelSignals({ ms, chosen }: { ms: any; chosen: string }) {
  const [view, setView] = useState<"now" | "history" | "record">("now");
  if (!ms) return null;
  const v = ms.votes || {};
  const nowCols: Col<any>[] = [
    { key: "model", label: "Model", render: (r) => <b>{name(r.model)}{r.model === chosen ? " ⭐" : ""}</b> },
    { key: "family", label: "Type" },
    { key: "signal", label: "Signal today", render: (r) => <Chip tone={SIG_TONE[r.signal]}>{r.signal}</Chip> },
    { key: "p_buy", label: "P(BUY)", num: true, render: (r) => pct(r.p_buy * 100, 1) },
    { key: "p_sell", label: "P(SELL)", num: true, render: (r) => pct(r.p_sell * 100, 1) },
  ];
  const models: string[] = (ms.stock_stats || []).map((x: any) => x.model);
  const histCols: Col<any>[] = [
    { key: "date", label: "Month-end" },
    { key: "fwd_pct", label: "Next month", num: true, render: (r) => <span className={tone(r.fwd_pct)}>{pct(r.fwd_pct, 1)}</span> },
    { key: "actual", label: "Actual", render: (r) => <SigTxt v={r.actual} /> },
    ...models.map((m) => ({ key: m, label: name(m).replace("Random Forest", "RF").replace("Extra Trees", "ET").replace("Transformer", "Trans."),
      render: (r: any) => <span title={`${name(m)}: ${r[m]}${r.actual ? (r[m] === r.actual ? " ✓" : " ✗") : ""}`}
        style={{ outline: r.actual && r[m] === r.actual ? "1px solid var(--line)" : "none", borderRadius: 4, padding: "0 3px" }}><SigTxt v={r[m]} short /></span> })),
  ];
  const recCols: Col<any>[] = [
    { key: "model", label: "Model", render: (r) => <b>{name(r.model)}{r.model === chosen ? " ⭐" : ""}</b> },
    { key: "family", label: "Type" },
    { key: "months", label: "Months", num: true },
    { key: "hit_rate_pct", label: "Signal = actual", num: true, render: (r) => pct(r.hit_rate_pct) },
    { key: "buy_months", label: "BUY months", num: true },
    { key: "buy_up_pct", label: "Rose after BUY", num: true, render: (r) => pct(r.buy_up_pct) },
    { key: "buy_avg_next_month_pct", label: "Avg after BUY", num: true, render: (r) => <span className={tone(r.buy_avg_next_month_pct)}>{pct(r.buy_avg_next_month_pct, 2)}</span> },
    { key: "sell_months", label: "SELL months", num: true },
    { key: "sell_avg_next_month_pct", label: "Avg after SELL", num: true, render: (r) => <span className={tone(r.sell_avg_next_month_pct)}>{pct(r.sell_avg_next_month_pct, 2)}</span> },
  ];
  return (
    <Card title={<>ML &amp; DL signals · <span className="mono">{ms.sym}</span></>} right={
      <Seg value={view} onChange={setView} options={[{ v: "now", label: "Today" }, { v: "history", label: "Backtest history" }, { v: "record", label: "Track record" }]} />}>
      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
        <span className="muted" style={{ fontSize: 13 }}>Today ({ms.as_of}), {ms.n_models} models vote:</span>
        <Chip tone="ok">BUY {v.BUY ?? 0}</Chip><Chip>HOLD {v.HOLD ?? 0}</Chip><Chip tone="bad">SELL {v.SELL ?? 0}</Chip>
        {ms.current?.length === 0 && <span className="muted">not in today's liquid universe — no current signal</span>}
      </div>
      {view === "now" && <DataTable rows={ms.current} cols={nowCols} maxHeight={560} empty="No current signal for this stock" />}
      {view === "history" && <DataTable rows={ms.history} cols={histCols} maxHeight={480} empty="This stock never met the liquidity filter in 2013–2026" />}
      {view === "record" && <DataTable rows={ms.stock_stats} cols={recCols} maxHeight={560} empty="No backtest history for this stock" />}
      <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
        {view === "history" ? "B / H / S = each model's BUY / HOLD / SELL at that month-end; boxed = matched what actually happened (BUY > +4%, SELL < −4%). " : ""}
        {ms.note} Per-stock track records are small samples — judge models by the all-stock results below.
      </div>
    </Card>
  );
}

export default function Stocks({ theme }: { theme: string }) {
  const ml = usePoll(() => api.get("/api/stocks/ml"));
  const [model, setModel] = useState<string>("");
  const [filter, setFilter] = useState<Sig>("");
  const [q, setQ] = useState("");
  const [compare, setCompare] = useState<string>("ensemble");
  const [pick, setPick] = useState<string>("");
  const sig = usePoll(() => api.get(`/api/stocks/signals?model=${model}`), 0, [model]);
  const chosen: string = ml.data?.chosen || "xgboost";
  const active: string = sig.data?.model || chosen;
  const best = (ml.data?.models || []).find((r: any) => r.model === chosen) || {};
  const selSym = pick || sig.data?.portfolio?.[0]?.sym || "";
  const choose = (r: any) => { setPick(r.sym); window.scrollTo({ top: 0, behavior: "smooth" }); };
  const ms = usePoll(() => (selSym ? api.get(`/api/stocks/model_signals?sym=${encodeURIComponent(selSym)}`) : Promise.resolve(null)), 0, [selSym]);
  const markers = useMemo<Marker[]>(() => (ms.data?.history || [])
    .filter((h: any) => h[active] === "BUY" || h[active] === "SELL")
    .map((h: any) => h[active] === "BUY"
      ? { time: h.date, position: "belowBar", shape: "arrowUp", color: "--pos", text: "B" }
      : { time: h.date, position: "aboveBar", shape: "arrowDown", color: "--neg", text: "S" }), [ms.data, active]);

  const rows = useMemo(() => {
    const all: any[] = sig.data?.rows || [];
    const ql = q.trim().toLowerCase();
    return all.filter((r) => (!filter || r.signal === filter) &&
      (!ql || r.sym.toLowerCase().includes(ql) || (r.name || "").toLowerCase().includes(ql)));
  }, [sig.data, filter, q]);

  const curve = useMemo(() => {
    const eq = ml.data?.equity || {};
    return [chosen, compare].filter((m, i, a) => m && eq[m] && a.indexOf(m) === i)
      .map((m, i) => ({ name: name(m), color: COLORS[i], data: eq[m].map((p: any) => ({ time: p.date, value: p.value })) }));
  }, [ml.data, chosen, compare]);

  const sigCols: Col<any>[] = [
    { key: "rank", label: "#", num: true },
    { key: "sym", label: "Symbol", render: (r) => <span className="mono">{r.sym}</span> },
    { key: "name", label: "Company" },
    { key: "exchange", label: "Exch" },
    { key: "signal", label: "Signal", render: (r) => <Chip tone={SIG_TONE[r.signal]}>{r.signal}</Chip> },
    { key: "p_buy", label: "P(BUY)", num: true, render: (r) => pct(r.p_buy * 100, 1) },
    { key: "p_sell", label: "P(SELL)", num: true, render: (r) => pct(r.p_sell * 100, 1) },
    { key: "score", label: "Score", num: true, render: (r) => <span className={tone(r.score)}>{num(r.score, 3)}</span> },
    { key: "last_close", label: "Last close", num: true, render: (r) => inr(r.last_close, 2) },
    { key: "traded_value_cr", label: "Traded ₹cr/day", num: true },
  ];
  const portCols: Col<any>[] = [
    { key: "pos", label: "#", num: true },
    { key: "sym", label: "Symbol", render: (r) => <span className="mono">{r.sym}</span> },
    { key: "name", label: "Company" },
    { key: "last_close", label: "Price", num: true, render: (r) => inr(r.last_close, 2) },
    { key: "qty", label: "Qty", num: true },
    { key: "amount", label: "Amount", num: true, render: (r) => inr(r.amount) },
    { key: "p_buy", label: "P(BUY)", num: true, render: (r) => pct(r.p_buy * 100, 1) },
    { key: "traded_value_cr", label: "Traded ₹cr/day", num: true },
  ];
  const modelCols: Col<any>[] = [
    { key: "model", label: "Model", render: (r) => <b>{name(r.model)}{r.model === chosen ? " ⭐" : ""}</b> },
    { key: "family", label: "Type" },
    { key: "cagr_pct", label: "Return / yr", num: true, render: (r) => <span className={tone(r.cagr_pct)}>{pct(r.cagr_pct)}</span> },
    { key: "max_dd_pct", label: "Worst fall", num: true, render: (r) => <span className="neg">{pct(r.max_dd_pct)}</span> },
    { key: "sharpe", label: "Sharpe", num: true },
    { key: "trade_win_rate_pct", label: "Trade win %", num: true, render: (r) => pct(r.trade_win_rate_pct) },
    { key: "BUY_went_up_pct", label: "BUY went up %", num: true, render: (r) => pct(r.BUY_went_up_pct) },
    { key: "BUY_avg_month_ret_pct", label: "BUY avg/mo", num: true, render: (r) => pct(r.BUY_avg_month_ret_pct, 2) },
    { key: "SELL_avg_month_ret_pct", label: "SELL avg/mo", num: true, render: (r) => pct(r.SELL_avg_month_ret_pct, 2) },
    { key: "sharpe_2013_19_select", label: "Sharpe 13–19", num: true },
    { key: "cagr_2020_26_holdout", label: "Unseen 20–26 /yr", num: true, render: (r) => <span className={tone(r.cagr_2020_26_holdout)}>{pct(r.cagr_2020_26_holdout)}</span> },
  ];
  const yearCols: Col<any>[] = useMemo(() => {
    const keys = [chosen, "ensemble", compare, "nifty", "ew"].filter((k, i, a) => a.indexOf(k) === i);
    return [{ key: "year", label: "Year", render: (r: any) => String(r.year) } as Col<any>, ...keys.map((k) => ({
      key: k, label: name(k), num: true, render: (r: any) => <span className={tone(r[k])}>{pct(r[k])}</span> }))];
  }, [chosen, compare]);

  const c = sig.data?.counts || {};
  const modelOptions = (ml.data?.models || []).filter((r: any) => !String(r.family).startsWith("bench")).map((r: any) => r.model);

  return (
    <State loading={ml.loading && !ml.data} error={ml.error}>
      <div className="banner warn">
        ML research for <b>stocks only</b> (delivery). Signals are model outputs for paper trading — not investment advice.
        Real-money trading stays locked until you approve it.
      </div>

      <div className="grid section">
        <Stat k="Best model (chosen on 2013–19)" v={name(chosen)} s={`run ${ml.data?.run || ""}`} />
        <Stat k="Return / yr 2013–26" v={pct(best.cagr_pct)} tone={tone(best.cagr_pct)} s={`unseen 2020–26: ${pct(best.cagr_2020_26_holdout)}`} />
        <Stat k="Worst fall" v={pct(best.max_dd_pct)} tone="neg" s="NIFTY 50: −29.7%" />
        <Stat k="Trade win rate" v={pct(best.trade_win_rate_pct)} s={`base rate ${pct(ml.data?.base_rate_up_pct)} of stock-months rose`} />
        <Stat k={`Signals · ${sig.data?.as_of || ""}`} v={<><span className="pos">{c.BUY ?? "—"}</span> / {c.HOLD ?? "—"} / <span className="neg">{c.SELL ?? "—"}</span></>} s="BUY / HOLD / SELL" />
      </div>

      <div className="section">
        <Forecast sym={selSym} theme={theme} onSym={setPick} stocks={sig.data?.rows || []} markers={markers} markerModel={active} />
      </div>

      <div className="section">
        <State error={ms.error}><ModelSignals ms={ms.data} chosen={chosen} /></State>
      </div>

      <div className="section">
        <Card title={`Model portfolio · ${name(active)} · top 10 BUY · ₹1,00,000`} right={
          <select value={active} onChange={(e) => setModel(e.target.value)} style={{ textTransform: "none" }} title="Which model's signals to use for the portfolio, the table and the chart arrows">
            {(sig.data?.models || [chosen]).map((m: string) => <option key={m} value={m}>{name(m)}{m === chosen ? " ⭐ (best)" : ""} · {FAM[m] || ""}</option>)}
          </select>}>
          <State loading={sig.loading && !sig.data} error={sig.error} empty={!sig.data?.portfolio?.length && "No BUY signals"}>
            <DataTable rows={sig.data?.portfolio} cols={portCols} onRowClick={choose} />
            <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
              {sig.data?.portfolio_note} Click a row for its price forecast. Rule: buy on BUY, keep on HOLD, sell on SELL — rebalanced once a month at the open after month-end.
            </div>
          </State>
        </Card>
      </div>

      <div className="section">
        <Card title={`All stocks · ${name(active)} signals`} right={
          <span style={{ display: "flex", gap: 8, alignItems: "center", textTransform: "none" }}>
            <input placeholder="Search symbol or company" value={q} onChange={(e) => setQ(e.target.value)} style={{ width: 220 }} />
            <Seg<Sig> value={filter} onChange={setFilter} options={[{ v: "", label: `All ${sig.data?.total ?? ""}` },
              { v: "BUY", label: `BUY ${c.BUY ?? ""}` }, { v: "HOLD", label: `HOLD ${c.HOLD ?? ""}` }, { v: "SELL", label: `SELL ${c.SELL ?? ""}` }]} />
          </span>}>
          <State loading={sig.loading && !sig.data} error={sig.error}>
            <DataTable rows={rows} cols={sigCols} maxHeight={480} empty="No stocks match" onRowClick={choose} />
          </State>
        </Card>
      </div>

      <div className="cols-2 section">
        <Card title="Backtest equity · ₹1,00,000 start" right={
          <select value={compare} onChange={(e) => setCompare(e.target.value)} style={{ textTransform: "none" }}>
            {modelOptions.filter((m: string) => m !== chosen).map((m: string) => <option key={m} value={m}>{`vs ${name(m)}`}</option>)}
          </select>}>
          <Chart series={curve} theme={theme} height={300} />
          <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
            Out-of-sample: every year is predicted by models trained only on earlier years. Final value {name(chosen)}: {inr(best.final_rs ?? (ml.data?.equity?.[chosen]?.slice(-1)[0]?.value))}.
          </div>
        </Card>
        <Card title="Yearly return %">
          <DataTable rows={ml.data?.yearly} cols={yearCols} maxHeight={900} />
        </Card>
      </div>

      <div className="section">
        <Card title="All models · out-of-sample 2013–2026 (after Groww charges + slippage)">
          <DataTable rows={ml.data?.models} cols={modelCols} maxHeight={900} />
          <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
            ⭐ chosen using 2013–2019 only; “Unseen 20–26 /yr” is the fair test. Picking the best of many models after the fact overstates future results.
          </div>
        </Card>
      </div>

      <div className="section">
        <Card title="Method & caveats">
          <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.7, fontSize: 13 }}>
            {Object.entries(ml.data?.method || {}).map(([k, v]) => <li key={k}><b>{k}:</b> {String(v)}</li>)}
            <li><b>refresh:</b> <span className="mono">python3 -m ml_lab.data.all_stocks</span> (prices) then <span className="mono">python3 -m ml_lab.stocks_bhs</span> (~40 min, retrains and writes new signals) and <span className="mono">python3 -m ml_lab.stock_price_forecast</span> (~1 min, price ranges)</li>
          </ul>
        </Card>
      </div>
    </State>
  );
}
