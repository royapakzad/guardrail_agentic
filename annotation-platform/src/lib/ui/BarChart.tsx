"use client";

import { useId, useState } from "react";

export type BarChartDatum = { label: string; value: number };

/**
 * Horizontal bar chart for ranking magnitudes (e.g. "count per domain",
 * "count per code"). Single-hue bars -- this ranks one measure across many
 * categories, so per-bar rainbow color would imply a categorical identity
 * that isn't there. Sorted descending, top-N with the rest folded into
 * "Other", hover tooltip with the exact count, direct value labels, and a
 * <table> fallback for accessibility/copy-paste.
 */
export function BarChart({
  data,
  topN = 10,
  unitLabel = "count",
}: {
  data: BarChartDatum[];
  topN?: number;
  unitLabel?: string;
}) {
  const [hovered, setHovered] = useState<number | null>(null);
  const [showTable, setShowTable] = useState(false);
  const gradientId = useId();

  if (data.length === 0) {
    return <p className="text-sm text-slate-400 dark:text-slate-500">No data.</p>;
  }

  const sorted = [...data].sort((a, b) => b.value - a.value);
  const visible = sorted.slice(0, topN);
  const rest = sorted.slice(topN);
  const otherTotal = rest.reduce((sum, d) => sum + d.value, 0);
  const bars: BarChartDatum[] = otherTotal > 0 ? [...visible, { label: `Other (${rest.length})`, value: otherTotal }] : visible;

  const max = Math.max(...bars.map((d) => d.value), 1);
  const barHeight = 22;
  const gap = 6;
  const labelWidth = 160;
  const chartWidth = 360;
  const height = bars.length * (barHeight + gap);

  return (
    <div className="flex flex-col gap-2">
      <svg
        role="img"
        aria-label={`Bar chart of ${unitLabel} by category, ${bars.length} bars shown`}
        width="100%"
        viewBox={`0 0 ${labelWidth + chartWidth} ${height}`}
        className="max-w-full"
      >
        <defs>
          <linearGradient id={gradientId} x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stopColor="#7dabe8" />
            <stop offset="100%" stopColor="#2a78d6" />
          </linearGradient>
        </defs>
        {bars.map((d, i) => {
          const y = i * (barHeight + gap);
          const w = Math.max((d.value / max) * chartWidth, 3);
          const isHovered = hovered === i;
          return (
            <g key={d.label} onMouseEnter={() => setHovered(i)} onMouseLeave={() => setHovered(null)}>
              <title>{`${d.label}: ${d.value.toLocaleString()} ${unitLabel}`}</title>
              <text
                x={labelWidth - 8}
                y={y + barHeight / 2}
                textAnchor="end"
                dominantBaseline="middle"
                className="fill-slate-600 dark:fill-slate-300"
                fontSize="11"
              >
                {d.label.length > 22 ? `${d.label.slice(0, 21)}…` : d.label}
              </text>
              <rect
                x={labelWidth}
                y={y + 2}
                width={w}
                height={barHeight - 4}
                rx={4}
                fill={isHovered ? "#184f95" : `url(#${gradientId})`}
              />
              <text
                x={labelWidth + w + 6}
                y={y + barHeight / 2}
                dominantBaseline="middle"
                className="fill-slate-500 dark:fill-slate-400 tabular-nums"
                fontSize="11"
              >
                {d.value.toLocaleString()}
              </text>
            </g>
          );
        })}
      </svg>
      <button
        type="button"
        onClick={() => setShowTable((v) => !v)}
        className="self-start text-xs text-sky-700 dark:text-sky-400 underline"
      >
        {showTable ? "Hide table view" : "Show as table"}
      </button>
      {showTable && (
        <div className="overflow-x-auto rounded-md border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-900">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-left dark:border-slate-700 dark:bg-slate-800">
                <th className="px-3 py-1.5 font-medium text-slate-600 dark:text-slate-300">Label</th>
                <th className="px-3 py-1.5 font-medium text-slate-600 dark:text-slate-300">{unitLabel}</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((d) => (
                <tr key={d.label} className="border-b border-slate-100 last:border-0 dark:border-slate-800">
                  <td className="px-3 py-1 max-w-xs truncate" title={d.label}>{d.label}</td>
                  <td className="px-3 py-1 tabular-nums">{d.value.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
