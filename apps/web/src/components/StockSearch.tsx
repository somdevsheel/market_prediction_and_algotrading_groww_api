import { ReactNode, useMemo, useRef, useState } from "react";

export interface StockItem { sym: string; name?: string | null; signal?: string | null; exchange?: string | null }

const MAX = 100;

/** Rank: symbol starts with query > a word of the name starts with it > symbol contains > name contains. */
function search(items: StockItem[], q: string): StockItem[] {
  const Q = q.trim().toUpperCase();
  if (!Q) return [];
  const scored: [number, StockItem][] = [];
  for (const it of items) {
    const s = it.sym.toUpperCase();
    const n = (it.name || "").toUpperCase();
    const r = s.startsWith(Q) ? 0
      : n.split(/[\s.&()-]+/).some((w) => w.startsWith(Q)) ? 1
      : s.includes(Q) ? 2
      : n.includes(Q) ? 3 : -1;
    if (r >= 0) scored.push([r, it]);
  }
  return scored.sort((a, b) => a[0] - b[0] || a[1].sym.localeCompare(b[1].sym)).map((x) => x[1]);
}

function mark(text: string, q: string): ReactNode {
  const i = text.toUpperCase().indexOf(q.trim().toUpperCase());
  if (!q.trim() || i < 0) return text;
  return <>{text.slice(0, i)}<mark>{text.slice(i, i + q.trim().length)}</mark>{text.slice(i + q.trim().length)}</>;
}

/** Type-ahead stock picker: every keystroke lists all matching stocks (symbol or company name). */
export function StockSearch({ items, onPick, placeholder = "Search stock — type a letter", width = 240 }:
  { items: StockItem[]; onPick: (sym: string) => void; placeholder?: string; width?: number }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const list = useRef<HTMLDivElement>(null);
  const matches = useMemo(() => search(items, q), [items, q]);
  const shown = matches.slice(0, MAX);

  const pick = (it?: StockItem) => {
    if (!it) return;
    onPick(it.sym);
    setQ("");
    setOpen(false);
  };
  const move = (d: number) => {
    const i = Math.max(0, Math.min(shown.length - 1, active + d));
    setActive(i);
    list.current?.querySelectorAll(".ac-item")[i]?.scrollIntoView({ block: "nearest" });
  };

  return (
    <div className="ac" style={{ width }}>
      <input
        value={q}
        placeholder={placeholder}
        style={{ width: "100%" }}
        aria-autocomplete="list"
        onChange={(e) => { setQ(e.target.value); setOpen(true); setActive(0); }}
        onFocus={() => setOpen(true)}
        onBlur={() => window.setTimeout(() => setOpen(false), 120)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") { e.preventDefault(); setOpen(true); move(1); }
          else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
          else if (e.key === "Enter") { e.preventDefault(); pick(shown[active]); }
          else if (e.key === "Escape") { setOpen(false); }
        }}
      />
      {open && q.trim() && (
        <div className="ac-list" ref={list} role="listbox">
          <div className="ac-head">{matches.length ? `${matches.length} match${matches.length > 1 ? "es" : ""}${matches.length > MAX ? ` · showing ${MAX}` : ""}` : "No stock matches"}</div>
          {shown.map((it, i) => (
            <div key={it.sym} role="option" aria-selected={i === active}
                 className={`ac-item ${i === active ? "on" : ""}`}
                 onMouseDown={(e) => { e.preventDefault(); pick(it); }}
                 onMouseEnter={() => setActive(i)}>
              <span className="ac-sym">{mark(it.sym, q)}</span>
              <span className="ac-name">{mark(it.name || "", q)}</span>
              {it.signal && <span className={`ac-sig ${it.signal === "BUY" ? "pos" : it.signal === "SELL" ? "neg" : "muted"}`}>{it.signal}</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
