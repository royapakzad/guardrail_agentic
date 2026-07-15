import { notFound } from "next/navigation";
import Link from "next/link";
import { USE_CASES, getRecordsForDataset, SEED_DATASET_ID } from "@/lib/adapters";
import { resolveDatasetIdParam } from "@/lib/datasetSelection";
import { listDatasets } from "@/lib/db/queries";
import { DatasetPicker } from "@/lib/ui/DatasetPicker";
import type { UseCase } from "@/lib/types";
import {
  computeAcronymChecks,
  computeFlipRate,
  computeLanguageFlips,
  computeLatency,
  computeScoreDeltas,
  computeTokenUsage,
  computeToolSelectionConsistency,
  computeToolUsage,
} from "@/lib/metrics";

function isUseCase(value: string): value is UseCase {
  return (USE_CASES as string[]).includes(value);
}

export function generateStaticParams() {
  return USE_CASES.map((useCase) => ({ useCase }));
}

export default async function UseCaseDashboard({
  params,
  searchParams,
}: {
  params: Promise<{ useCase: string }>;
  searchParams: Promise<{ dataset?: string }>;
}) {
  const { useCase: useCaseParam } = await params;
  const { dataset: datasetParam } = await searchParams;
  if (!isUseCase(useCaseParam)) notFound();
  const useCase = useCaseParam;

  const datasetId = await resolveDatasetIdParam(useCase, datasetParam);
  const [records, availableDatasets] = await Promise.all([
    getRecordsForDataset(useCase, datasetId),
    listDatasets(useCase).catch(() => []),
  ]);
  const scoreDeltas = computeScoreDeltas(records);
  const flipRate = computeFlipRate(records);
  const toolUsage = computeToolUsage(records);
  const toolSelection = computeToolSelectionConsistency(records);
  const latency = computeLatency(records);
  const tokens = computeTokenUsage(records);
  const acronymChecks = computeAcronymChecks(records);
  const languageFlips = computeLanguageFlips(useCase, records);

  const labels = scoreDeltas.map((s) => s.label);

  return (
    <div className="flex flex-col gap-10">
      <div className="flex items-baseline justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight capitalize">{useCase} dashboard</h1>
          <p className="mt-1 text-sm text-slate-600">
            {records.length} scenarios · {labels.length} policy variants ·{" "}
            <Link href={`/${useCase}/scenarios`} className="underline">browse scenarios &amp; annotate</Link>
          </p>
        </div>
        <DatasetPicker useCase={useCase} datasets={availableDatasets} currentId={String(datasetId)} basePath={`/${useCase}`} />
      </div>

      {records.length < 10 && (
        <div className="rounded-md border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          Small sample ({records.length} scenario{records.length === 1 ? "" : "s"}) — numbers below are not
          statistically meaningful yet.{" "}
          {datasetId === SEED_DATASET_ID
            ? "This is the bundled sample — upload a full batch run above for real numbers."
            : "This uploaded run is small — consider uploading a larger batch."}
        </div>
      )}

      <Section title="Score deltas (non-agentic → agentic/merged)">
        <Table
          columns={["Policy variant", "n", "Mean Δ", "Median Δ", "Mean |Δ|", "Harsher", "Lenient", "Unchanged"]}
          rows={scoreDeltas.map((s) => [
            s.label,
            s.n,
            s.meanDelta ?? "—",
            s.medianDelta ?? "—",
            s.meanAbsDelta ?? "—",
            s.harsherCount,
            s.lenientCount,
            s.unchangedCount,
          ])}
        />
      </Section>

      <Section title="Verdict flip rate" note="Only counts records where the agentic pass produced criteria_verdicts to compare against — see adapter notes for historical records where it didn't.">
        <Table
          columns={["Policy variant", "Comparable n", "Flipped", "Flip rate", "Top flipped criteria"]}
          rows={flipRate.map((f) => [
            f.label,
            f.n,
            f.flippedCount,
            f.flipRate !== null ? `${(f.flipRate * 100).toFixed(0)}%` : "—",
            f.topFlippedCriteria.map((c) => `${c.criterion} (${c.count})`).join(", ") || "—",
          ])}
        />
      </Section>

      <Section title="Tool usage">
        <Table
          columns={["Policy variant", "n", "Mean calls/scenario", "Median calls/scenario", "Tool frequency"]}
          rows={toolUsage.map((t) => [
            t.label,
            t.n,
            t.meanCallsPerScenario ?? "—",
            t.medianCallsPerScenario ?? "—",
            t.toolFrequency.map((f) => `${f.tool} (${f.count})`).join(", ") || "—",
          ])}
        />
      </Section>

      <Section
        title="Tool-selection consistency"
        note="Best-effort text-pattern heuristic (no gold labels for this batch data) — see lib/metrics/toolSelectionConsistency.ts."
      >
        <Table
          columns={["Policy variant", "Acronym opportunities", "check_acronym used", "Consistency", "URL opportunities", "URL tool used", "Consistency"]}
          rows={toolSelection.map((t) => [
            t.label,
            t.acronymOpportunities,
            t.acronymToolUsed,
            t.acronymConsistency !== null ? `${(t.acronymConsistency * 100).toFixed(0)}%` : "—",
            t.urlOpportunities,
            t.urlToolUsed,
            t.urlConsistency !== null ? `${(t.urlConsistency * 100).toFixed(0)}%` : "—",
          ])}
        />
      </Section>

      <Section title="Latency (judgment time, seconds)">
        <Table
          columns={["Policy variant", "n", "Mean non-agentic", "Median non-agentic", "Mean agentic", "Median agentic"]}
          rows={latency.map((l) => [
            l.label,
            l.n,
            l.meanNonagenticS ?? "—",
            l.medianNonagenticS ?? "—",
            l.meanAgenticS ?? "—",
            l.medianAgenticS ?? "—",
          ])}
        />
      </Section>

      <Section title="Token usage">
        <Table
          columns={["Policy variant", "n", "Mean non-agentic total", "Mean agentic total", "Mean agentic peak prompt"]}
          rows={tokens.map((t) => [
            t.label,
            t.n,
            t.meanNonagenticTotal ?? "—",
            t.meanAgenticTotal ?? "—",
            t.meanAgenticPeakPrompt ?? "—",
          ])}
        />
      </Section>

      <Section
        title="Acronym checks (check_acronym tool)"
        note="Descriptive only — verdict_hint is the tool's own heuristic, not a verified-correct label. See analysis doc §3.4."
      >
        <Table
          columns={["Policy variant", "Total checks", "Mean match_score", "verdict_hint distribution"]}
          rows={acronymChecks
            .filter((a) => a.totalChecks > 0)
            .map((a) => [
              a.label,
              a.totalChecks,
              a.meanMatchScore ?? "—",
              a.verdictHintDistribution.map((h) => `${h.hint} (${h.count})`).join(", "),
            ])}
        />
      </Section>

      <Section title={`Language flips (en vs. ${languageFlips?.otherLanguage ?? "—"})`}>
        {languageFlips ? (
          <div className="flex flex-col gap-3">
            <div className="text-sm text-slate-700">
              {languageFlips.pairedScenarioCount} paired scenario/model combinations · mean |Δ|{" "}
              {languageFlips.meanAbsDelta ?? "—"} · {languageFlips.flippedCount} valid/invalid disagreements (
              {languageFlips.flipRate !== null ? `${(languageFlips.flipRate * 100).toFixed(0)}%` : "—"})
            </div>
            <Table
              columns={["Scenario (model)", "EN score", `${languageFlips.otherLanguage.toUpperCase()} score`, "Δ", "EN valid", `${languageFlips.otherLanguage.toUpperCase()} valid`, "Flipped"]}
              rows={languageFlips.rows.map((r) => [
                r.id,
                r.enScore ?? "—",
                r.otherScore ?? "—",
                r.delta !== null ? r.delta.toFixed(2) : "—",
                r.enValid === null ? "—" : r.enValid ? "valid" : "invalid",
                r.otherValid === null ? "—" : r.otherValid ? "valid" : "invalid",
                r.flipped ? "yes" : "no",
              ])}
            />
          </div>
        ) : (
          <p className="text-sm text-slate-500">No paired second language configured/available for this use case.</p>
        )}
      </Section>
    </div>
  );
}

function Section({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) {
  return (
    <section className="flex flex-col gap-2">
      <div>
        <h2 className="text-lg font-semibold">{title}</h2>
        {note && <p className="text-xs text-slate-500 mt-0.5">{note}</p>}
      </div>
      {children}
    </section>
  );
}

function Table({ columns, rows }: { columns: string[]; rows: (string | number)[][] }) {
  if (rows.length === 0) {
    return <p className="text-sm text-slate-400">No data.</p>;
  }
  return (
    <div className="overflow-x-auto rounded-md border border-slate-200 bg-white">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 bg-slate-50 text-left">
            {columns.map((c) => (
              <th key={c} className="px-3 py-2 font-medium text-slate-600 whitespace-nowrap">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-slate-100 last:border-0">
              {row.map((cell, j) => (
                <td key={j} className="px-3 py-2 tabular-nums whitespace-nowrap max-w-xs truncate" title={String(cell)}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
