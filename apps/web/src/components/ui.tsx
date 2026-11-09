import { ReactNode } from "react";

export function Stat({ k, v, s, tone = "" }: { k: string; v: ReactNode; s?: ReactNode; tone?: string }) {
  return (
    <div className="stat">
      <div className="k">{k}</div>
      <div className={`v ${tone}`}>{v}</div>
      {s ? <div className="s">{s}</div> : null}
    </div>
  );
}

export function Card({ title, right, children, className = "" }: { title?: ReactNode; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <div className={`card ${className}`}>
      {title ? <h2>{title}<span className="sp" />{right}</h2> : null}
      {children}
    </div>
  );
}

export function State({ loading, error, empty, children }: { loading?: boolean; error?: string | null; empty?: string | false; children?: ReactNode }) {
  if (error) return <div className="error">⚠ {error}</div>;
  if (loading) return <div className="loading">Loading…</div>;
  if (empty) return <div className="empty">{empty}</div>;
  return <>{children}</>;
}

export function Seg<T extends string>({ value, options, onChange }: { value: T; options: { v: T; label: string }[]; onChange: (v: T) => void }) {
  return (
    <div className="seg">
      {options.map((o) => (
        <button key={o.v} className={o.v === value ? "on" : ""} onClick={() => onChange(o.v)}>{o.label}</button>
      ))}
    </div>
  );
}

export function Chip({ tone = "", children }: { tone?: "" | "ok" | "bad" | "warn"; children: ReactNode }) {
  return <span className={`chip ${tone}`}><span className="dot" />{children}</span>;
}
