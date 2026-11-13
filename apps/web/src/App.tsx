import { useEffect, useState } from "react";
import { api } from "./api";
import { Chip } from "./components/ui";
import { usePoll, useStored, useTheme } from "./hooks";
import Backtests from "./pages/Backtests";
import Health from "./pages/Health";
import Market from "./pages/Market";
import Overview from "./pages/Overview";
import Pnl from "./pages/Pnl";
import Research from "./pages/Research";
import Stocks from "./pages/Stocks";
import Strategy from "./pages/Strategy";
import Trades from "./pages/Trades";
import Virtual from "./pages/Virtual";

const PAGES = [
  { id: "overview", label: "Overview", ic: "◎" },
  { id: "market", label: "Market", ic: "≋" },
  { id: "strategy", label: "Strategy", ic: "◆" },
  { id: "pnl", label: "P&L", ic: "₹" },
  { id: "trades", label: "Trades", ic: "☰" },
  { id: "backtests", label: "Backtests", ic: "↺" },
  { id: "research", label: "Research", ic: "σ" },
  { id: "stocks", label: "Stocks ML", ic: "▲" },
  { id: "virtual", label: "Virtual Portfolio", ic: "◇" },
  { id: "health", label: "Health", ic: "♥" },
] as const;
type PageId = (typeof PAGES)[number]["id"];

const fromHash = (): PageId => {
  const h = window.location.hash.slice(1);
  return (PAGES.find((p) => p.id === h)?.id || "overview") as PageId;
};

export default function App() {
  const [page, setPage] = useState<PageId>(fromHash());
  const [theme, toggleTheme] = useTheme();
  const [mode, setMode] = useStored<"paper" | "live">("nq-ledger", "paper");
  const [menu, setMenu] = useState(false);
  const health = usePoll(() => api.get("/api/health"), 20000);
  const summary = usePoll(() => api.get("/api/summary"), 20000);
  useEffect(() => {
    const h = () => setPage(fromHash());
    window.addEventListener("hashchange", h);
    return () => window.removeEventListener("hashchange", h);
  }, []);
  const go = (p: PageId) => { window.location.hash = p; setMenu(false); };
  const sysMode = summary.data?.config_mode;
  const kill = summary.data?.kill_switch;
  const title = PAGES.find((p) => p.id === page)?.label;

  return (
    <div className="app">
      <aside className={`side ${menu ? "open" : ""}`}>
        <div className="brand">Nifty Quant Desk<small>ORB retest · cheap IV</small></div>
        <nav className="nav">
          {PAGES.map((p) => (
            <button key={p.id} className={page === p.id ? "on" : ""} onClick={() => go(p.id)}><span className="ic">{p.ic}</span>{p.label}</button>
          ))}
        </nav>
        <div className="foot">Ledger view
          <div className="seg" style={{ marginTop: 6 }}>
            <button className={mode === "paper" ? "on" : ""} onClick={() => setMode("paper")}>paper</button>
            <button className={mode === "live" ? "on" : ""} onClick={() => setMode("live")}>live</button>
          </div>
          <div style={{ marginTop: 6 }}>Changes which trades are shown — not the trading mode.</div>
        </div>
      </aside>
      <main className="main">
        <header className="top">
          <button className="btn menu" onClick={() => setMenu(!menu)} aria-label="Menu">☰</button>
          <h1>{title}</h1>
          <span className="sp" />
          {sysMode && <Chip tone={sysMode === "live" ? "bad" : "ok"}>system mode: {sysMode.toUpperCase()}</Chip>}
          {kill && <Chip tone="bad">KILL SWITCH ON</Chip>}
          {health.data && <Chip tone={health.data.overall === "ok" ? "ok" : "warn"}>health: {health.data.overall}</Chip>}
          <button className="btn" onClick={toggleTheme} title="Toggle light/dark">{theme === "dark" ? "☀" : "☾"}</button>
        </header>
        <div className="content">
          {page === "overview" && <Overview mode={mode} theme={theme} />}
          {page === "market" && <Market />}
          {page === "strategy" && <Strategy theme={theme} />}
          {page === "pnl" && <Pnl mode={mode} theme={theme} />}
          {page === "trades" && <Trades mode={mode} />}
          {page === "backtests" && <Backtests theme={theme} />}
          {page === "research" && <Research theme={theme} />}
          {page === "stocks" && <Stocks theme={theme} />}
          {page === "virtual" && <Virtual theme={theme} />}
          {page === "health" && <Health />}
        </div>
      </main>
    </div>
  );
}
