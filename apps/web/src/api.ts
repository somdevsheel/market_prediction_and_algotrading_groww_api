// Thin typed-enough client for the FastAPI backend. All P&L comes from the server-side ledger.
export class ApiError extends Error {}

async function req<T = any>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); msg = j.detail || msg; } catch { /* not json */ }
    throw new ApiError(msg);
  }
  const ct = r.headers.get("content-type") || "";
  return (ct.includes("json") ? r.json() : r.text()) as Promise<T>;
}

export const api = {
  get: <T = any>(url: string) => req<T>(url),
  post: <T = any>(url: string, body: unknown) =>
    req<T>(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
};
