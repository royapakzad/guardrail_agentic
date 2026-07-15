import { USE_CASES, getRecordsForDataset } from "@/lib/adapters";

// Reads live "most recent upload" state from Postgres each time — must not be
// frozen as a static page at build time (there's no ?searchParam to force
// Next to treat it as dynamic automatically here, unlike the [useCase] pages).
export const dynamic = "force-dynamic";
import { resolveDefaultDatasetId } from "@/lib/datasetSelection";
import type { UseCase } from "@/lib/types";
import {
  computeFlipRate,
  computeLatency,
  computeScoreDeltas,
  computeToolSelectionConsistency,
  computeToolUsage,
  type ScoreDeltaSummary,
} from "@/lib/metrics";

// The domain-specific, Claude-judged policy variant per use case, when present
// — the most directly comparable "headline" number, rather than mixing in the
// shared policy_generic fallback or a GPT-judged duplicate run. Falls back to
// whichever variant has the most records if this exact label isn't in the
// currently selected dataset (e.g. a freshly uploaded run may not reuse the
// exact same policy/model naming as the bundled sample).
const PRIMARY_LABEL: Record<UseCase, string> = {
  humanitarian: "humanitarian_policy_explicit_claude_sonnet_4_6",
  financial: "financial_policy",
  cybersecurity: "policy_policy_cybersecurity[anthropic:claude-sonnet-4-6]",
};

function pickLabel(scoreDeltas: ScoreDeltaSummary[], preferred: string): string | undefined {
  if (scoreDeltas.some((s) => s.label === preferred)) return preferred;
  return [...scoreDeltas].sort((a, b) => b.n - a.n)[0]?.label;
}

export default async function ComparePage() {
  const rows = await Promise.all(
    USE_CASES.map(async (useCase) => {
      const datasetId = await resolveDefaultDatasetId(useCase);
      const records = await getRecordsForDataset(useCase, datasetId);

      const scoreDeltas = computeScoreDeltas(records);
      const label = pickLabel(scoreDeltas, PRIMARY_LABEL[useCase]);
      const scoreDelta = scoreDeltas.find((s) => s.label === label);
      const flip = computeFlipRate(records).find((f) => f.label === label);
      const tools = computeToolUsage(records).find((t) => t.label === label);
      const selection = computeToolSelectionConsistency(records).find((t) => t.label === label);
      const latency = computeLatency(records).find((l) => l.label === label);

      return { useCase, n: records.length, label, scoreDelta, flip, tools, selection, latency };
    })
  );

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Cross-use-case comparison</h1>
        <p className="mt-1 text-sm text-slate-600 max-w-2xl">
          Each use case&apos;s most recently uploaded dataset (or the bundled sample if none has been
          uploaded yet), one representative policy variant per use case — see each use case&apos;s own
          dashboard to pick a different dataset or see the full breakdown across all policy variants.
        </p>
      </div>

      {rows.some((r) => r.n < 10) && (
        <div className="rounded-md border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          {rows
            .filter((r) => r.n < 10)
            .map((r) => `${r.useCase} (n=${r.n})`)
            .join(", ")}{" "}
          — small sample, not comparable in scale to the others yet.
        </div>
      )}

      <div className="overflow-x-auto rounded-md border border-slate-200 bg-white">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-left">
              <th className="px-3 py-2 font-medium text-slate-600">Metric</th>
              {rows.map((r) => (
                <th key={r.useCase} className="px-3 py-2 font-medium text-slate-600 capitalize whitespace-nowrap">
                  {r.useCase} <span className="text-slate-400 font-normal">(n={r.n})</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="[&>tr]:border-b [&>tr]:border-slate-100 [&>tr:last-child]:border-0">
            <MetricRow label="Mean score Δ" rows={rows} get={(r) => r.scoreDelta?.meanDelta} />
            <MetricRow label="Mean |Δ|" rows={rows} get={(r) => r.scoreDelta?.meanAbsDelta} />
            <MetricRow label="Harsher / Lenient / Unchanged" rows={rows} get={(r) => r.scoreDelta && `${r.scoreDelta.harsherCount} / ${r.scoreDelta.lenientCount} / ${r.scoreDelta.unchangedCount}`} />
            <MetricRow label="Flip rate" rows={rows} get={(r) => r.flip?.flipRate !== null && r.flip?.flipRate !== undefined ? `${(r.flip.flipRate * 100).toFixed(0)}%` : undefined} />
            <MetricRow label="Mean tool calls/scenario" rows={rows} get={(r) => r.tools?.meanCallsPerScenario} />
            <MetricRow label="Acronym tool-selection consistency" rows={rows} get={(r) => r.selection?.acronymConsistency !== null && r.selection?.acronymConsistency !== undefined ? `${(r.selection.acronymConsistency * 100).toFixed(0)}%` : undefined} />
            <MetricRow label="URL tool-selection consistency" rows={rows} get={(r) => r.selection?.urlConsistency !== null && r.selection?.urlConsistency !== undefined ? `${(r.selection.urlConsistency * 100).toFixed(0)}%` : undefined} />
            <MetricRow label="Mean latency, non-agentic (s)" rows={rows} get={(r) => r.latency?.meanNonagenticS} />
            <MetricRow label="Mean latency, agentic (s)" rows={rows} get={(r) => r.latency?.meanAgenticS} />
          </tbody>
        </table>
      </div>
    </div>
  );
}

function MetricRow<T extends { useCase: UseCase }>({
  label,
  rows,
  get,
}: {
  label: string;
  rows: T[];
  get: (r: T) => string | number | undefined | null;
}) {
  return (
    <tr>
      <td className="px-3 py-2 font-medium text-slate-700 whitespace-nowrap">{label}</td>
      {rows.map((r) => {
        const v = get(r);
        return (
          <td key={r.useCase} className="px-3 py-2 tabular-nums">
            {v === undefined || v === null ? "—" : v}
          </td>
        );
      })}
    </tr>
  );
}
