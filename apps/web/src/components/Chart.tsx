import { useEffect, useRef } from "react";
import { createChart, ColorType, IChartApi, LineStyle } from "lightweight-charts";

export interface Series {
  name: string;
  type?: "line" | "area" | "histogram";
  color?: string; // CSS var name like "--pos" or a literal color
  data: { time: string; value: number | null | undefined; color?: string }[];
}

const cssVar = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const resolve = (c?: string) => (c?.startsWith("--") ? cssVar(c) : c) || cssVar("--accent");

/** Theme-aware TradingView Lightweight Chart. Re-renders when data or theme changes. */
export function Chart({ series, height = 280, theme }: { series: Series[]; height?: number; theme: string }) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  useEffect(() => {
    if (!el.current) return;
    const c = createChart(el.current, {
      autoSize: true,
      height,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: cssVar("--muted"), fontFamily: "Inter, system-ui" },
      grid: { vertLines: { color: cssVar("--chart-grid") }, horzLines: { color: cssVar("--chart-grid") } },
      rightPriceScale: { borderColor: cssVar("--line") },
      timeScale: { borderColor: cssVar("--line") },
      crosshair: { horzLine: { style: LineStyle.Dashed }, vertLine: { style: LineStyle.Dashed } },
    });
    chart.current = c;
    for (const s of series) {
      const data = s.data.filter((d) => d.value != null && Number.isFinite(d.value as number)) as any[];
      const seen = new Set<string>();
      const clean = data.filter((d) => (seen.has(d.time) ? false : (seen.add(d.time), true)))
                        .sort((a, b) => (a.time < b.time ? -1 : 1));
      const color = resolve(s.color);
      const ser = s.type === "area"
        ? c.addAreaSeries({ lineColor: color, topColor: color + "40", bottomColor: color + "05", lineWidth: 2, title: s.name })
        : s.type === "histogram"
        ? c.addHistogramSeries({ color, title: s.name })
        : c.addLineSeries({ color, lineWidth: 2, title: s.name });
      ser.setData(clean);
    }
    c.timeScale().fitContent();
    return () => { c.remove(); chart.current = null; };
  }, [series, height, theme]);
  return <div ref={el} className="chart" style={{ height }} />;
}
