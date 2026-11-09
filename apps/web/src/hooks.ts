import { useCallback, useEffect, useRef, useState } from "react";

export interface Loadable<T> { data: T | null; error: string | null; loading: boolean; reload: () => void; }

/** Fetch on mount and every `ms` (0 = once). Keeps previous data while refreshing. */
export function usePoll<T>(fn: () => Promise<T>, ms = 0, deps: unknown[] = []): Loadable<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const alive = useRef(true);
  const run = useCallback(async () => {
    setLoading(true);
    try { const d = await fn(); if (alive.current) { setData(d); setError(null); } }
    catch (e: any) { if (alive.current) setError(e?.message || String(e)); }
    finally { if (alive.current) setLoading(false); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  useEffect(() => {
    alive.current = true;
    run();
    const id = ms ? window.setInterval(run, ms) : 0;
    return () => { alive.current = false; if (id) window.clearInterval(id); };
  }, [run, ms]);
  return { data, error, loading, reload: run };
}

export type Theme = "dark" | "light";
const read = (k: string) => { try { return localStorage.getItem(k); } catch { return null; } };
const write = (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } };

export function useTheme(): [Theme, () => void] {
  const sys: Theme = window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
  const [theme, setTheme] = useState<Theme>((read("nq-theme") as Theme) || sys);
  useEffect(() => { document.documentElement.dataset.theme = theme; write("nq-theme", theme); }, [theme]);
  return [theme, () => setTheme((t) => (t === "dark" ? "light" : "dark"))];
}

export function useStored<T extends string>(key: string, initial: T): [T, (v: T) => void] {
  const [v, setV] = useState<T>((read(key) as T) || initial);
  return [v, (x: T) => { setV(x); write(key, x); }];
}
