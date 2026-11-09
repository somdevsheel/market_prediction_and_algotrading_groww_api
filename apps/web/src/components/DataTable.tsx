import { ReactNode, useEffect, useMemo, useState } from "react";

export interface Col<T> {
  key: string;
  label: string;
  num?: boolean;
  render?: (row: T) => ReactNode;
  value?: (row: T) => unknown; // sort value
}

interface Props<T> {
  rows: T[] | null | undefined;
  cols: Col<T>[];
  empty?: string;
  onRowClick?: (row: T, i: number) => void;
  selected?: number;
  maxHeight?: number;
  keyNav?: boolean; // ↑/↓ moves selection
}

export function DataTable<T extends Record<string, any>>({ rows, cols, empty = "No data", onRowClick, selected, maxHeight, keyNav }: Props<T>) {
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(null);
  const sorted = useMemo(() => {
    const r = [...(rows || [])];
    if (!sort) return r;
    const c = cols.find((x) => x.key === sort.key);
    const val = (row: T) => (c?.value ? c.value(row) : row[sort.key]);
    return r.sort((a, b) => {
      const x = val(a) as any, y = val(b) as any;
      if (x == null) return 1;
      if (y == null) return -1;
      return (x > y ? 1 : x < y ? -1 : 0) * sort.dir;
    });
  }, [rows, cols, sort]);

  useEffect(() => {
    if (!keyNav || !onRowClick || !sorted.length) return;
    const h = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT") return;
      const cur = selected ?? -1;
      if (e.key === "ArrowDown") { e.preventDefault(); const i = Math.min(sorted.length - 1, cur + 1); onRowClick(sorted[i], i); }
      if (e.key === "ArrowUp") { e.preventDefault(); const i = Math.max(0, cur - 1); onRowClick(sorted[i], i); }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [keyNav, onRowClick, sorted, selected]);

  if (!rows || !rows.length) return <div className="empty">{empty}</div>;
  return (
    <div className="tw" style={maxHeight ? { maxHeight } : undefined}>
      <table>
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c.key} className={c.num ? "num" : ""}
                  onClick={() => setSort((s) => (s?.key === c.key ? { key: c.key, dir: (s.dir * -1) as 1 | -1 } : { key: c.key, dir: -1 }))}>
                {c.label}{sort?.key === c.key ? (sort.dir === 1 ? " ↑" : " ↓") : ""}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr key={i} className={`${onRowClick ? "click" : ""} ${selected === i ? "sel" : ""}`}
                onClick={onRowClick ? () => onRowClick(r, i) : undefined}>
              {cols.map((c) => (
                <td key={c.key} className={c.num ? "num" : ""}>{c.render ? c.render(r) : fmtCell(r[c.key])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function fmtCell(v: unknown): ReactNode {
  if (v == null) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString("en-IN") : v.toLocaleString("en-IN", { maximumFractionDigits: 2 });
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "object") return <span className="mono muted">{JSON.stringify(v)}</span>;
  return String(v);
}
