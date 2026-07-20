/*
 * Semantic color convention for this app (Tailwind slate/sky/emerald/amber/red/violet):
 *   slate   — structure & neutral chrome (nav, cards, secondary text)
 *   sky     — interactive/tool-related (evidence source, links, tool badges)
 *   emerald — compliant / agree / success
 *   amber   — not fully compliant / disagree / needs review
 *   red     — errors, destructive actions (delete)
 *   violet  — qualitative-coding accent, reserved for codebook/coding UI so
 *             "the judge's evaluation" (slate/sky/emerald/amber) stays visually
 *             distinct from "your qualitative coding" (violet)
 */
const VERDICT_COLORS: Record<string, string> = {
  COMPLIANT: "bg-emerald-100 text-emerald-800",
  NOT_FULLY_COMPLIANT: "bg-amber-100 text-amber-800",
  // Older runs predating the binary compliance scale (Issue #54 follow-up) may
  // still contain these — kept so historical data still renders with color.
  MINOR_ISSUE: "bg-amber-100 text-amber-800",
  MAJOR_ISSUE: "bg-orange-100 text-orange-800",
  CRITICAL: "bg-red-100 text-red-800",
};

export function VerdictBadge({ verdict }: { verdict: string }) {
  const cls = VERDICT_COLORS[verdict] ?? "bg-slate-100 text-slate-700";
  return <span className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${cls}`}>{verdict}</span>;
}

export function ValidBadge({ valid }: { valid: boolean | null }) {
  if (valid === null) return <span className="text-slate-400 text-xs">n/a</span>;
  return (
    <span
      className={`inline-block rounded-full px-2 py-0.5 text-xs font-semibold ${
        valid ? "bg-emerald-100 text-emerald-800" : "bg-red-100 text-red-800"
      }`}
    >
      {valid ? "VALID" : "INVALID"}
    </span>
  );
}

export function ScoreBar({ score }: { score: number | null }) {
  if (score === null) return <span className="text-slate-400 text-xs">n/a</span>;
  const pct = Math.max(0, Math.min(100, Math.round(score * 100)));
  const color = score >= 0.7 ? "bg-emerald-500" : score >= 0.4 ? "bg-amber-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-16 rounded-full bg-slate-200 overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs tabular-nums text-slate-600">{score.toFixed(2)}</span>
    </div>
  );
}
