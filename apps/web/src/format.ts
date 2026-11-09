export const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

export const num = (v: unknown, d = 2) =>
  isNum(v) ? v.toLocaleString("en-IN", { maximumFractionDigits: d, minimumFractionDigits: 0 }) : "—";

export const inr = (v: unknown, d = 0) =>
  isNum(v) ? `${v < 0 ? "−" : ""}₹${Math.abs(v).toLocaleString("en-IN", { maximumFractionDigits: d })}` : "—";

export const pct = (v: unknown, d = 1) => (isNum(v) ? `${v.toFixed(d)}%` : "—");

export const tone = (v: unknown) => (isNum(v) ? (v > 0 ? "pos" : v < 0 ? "neg" : "") : "");

export const time = (iso?: string | null) => (iso ? iso.replace("T", " ").slice(0, 19) : "—");

export const ago = (iso?: string | null) => {
  if (!iso) return "—";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
};
