import { notFound } from "next/navigation";
import Link from "next/link";
import { USE_CASES, getRecordsForDataset, SEED_DATASET_ID } from "@/lib/adapters";
import { resolveDatasetIdParam } from "@/lib/datasetSelection";
import { listDatasets, listCodeApplicationsForUseCase } from "@/lib/db/queries";
import { DatasetPicker } from "@/lib/ui/DatasetPicker";
import { BarChart } from "@/lib/ui/BarChart";
import { UseCaseNav } from "@/lib/ui/UseCaseNav";
import type { UseCase } from "@/lib/types";
import {
  computeAcronymChecks,
  computeComplianceByCriterion,
  computeCriterionFlips,
  computeDomainUsage,
  computeFlipRate,
  computeLanguageFlips,
  computeLatency,
  computeScoreDeltas,
  computeTokenUsage,
  computeToolSelectionConsistency,
  computeToolUsage,
  type CriterionComplianceRow,
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
  const complianceByCriterion = computeComplianceByCriterion(records);
  const criterionFlips = computeCriterionFlips(records);
  const toolUsage = computeToolUsage(records);
  const toolSelection = computeToolSelectionConsistency(records);
  const latency = computeLatency(records);
  const tokens = computeTokenUsage(records);
  const acronymChecks = computeAcronymChecks(records);
  const languageFlips = computeLanguageFlips(useCase, records);
  const domainUsage = computeDomainUsage(records);
  const codeApplications = await listCodeApplicationsForUseCase(useCase).catch(() => []);

  const codeFrequency = new Map<string, number>();
  const themeByCode = new Map<string, string>();
  for (const a of codeApplications) {
    const key = a.code_name;
    codeFrequency.set(key, (codeFrequency.get(key) ?? 0) + 1);
    themeByCode.set(key, a.code_theme ?? "(no theme)");
  }
  const codeFrequencyData = [...codeFrequency.entries()].map(([label, value]) => ({
    label: `${themeByCode.get(label)} / ${label}`,
    value,
  }));

  const labels = scoreDeltas.map((s) => s.label);

  return (
    <div className="flex flex-col gap-8">
      <UseCaseNav useCase={useCase} datasetId={String(datasetId)} />

      <div className="flex items-baseline justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight capitalize">{useCase} dashboard</h1>
          <p className="mt-1 text-sm text-slate-600">
            {records.length} scenarios · {labels.length} policy variants
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

      <Section
        title="Compliance by policy criterion"
        note="Every criterion's own compliance distribution — not just the aggregate valid/invalid score. Non-agentic (full policy, no tools) vs. agentic/final (tool-verified), per policy variant."
      >
        <div className="flex flex-col gap-4">
          {complianceByCriterion.map((c) => (
            <details key={c.label} className="rounded-md border border-slate-200 bg-white" open={complianceByCriterion.length === 1}>
              <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-slate-700 bg-slate-50">{c.label}</summary>
              <div className="p-2">
                <ComplianceTable rows={c.rows} />
              </div>
            </details>
          ))}
        </div>
      </Section>

      <Section
        title="Verdict flip rate (aggregate)"
        note="Did *any* criterion flip for this scenario. Only counts records where the agentic pass produced criteria_verdicts to compare against. See the per-criterion breakdown below for what's actually informative — non-tool criteria can't flip by construction."
      >
        <Table
          columns={["Policy variant", "Comparable n", "Flipped", "Flip rate"]}
          rows={flipRate.map((f) => [
            f.label,
            f.n,
            f.flippedCount,
            f.flipRate !== null ? `${(f.flipRate * 100).toFixed(0)}%` : "—",
          ])}
        />
      </Section>

      <Section
        title="Tool-requiring criteria: agentic vs. non-agentic, one at a time"
        note='Restricted to criteria tagged "(potentially needs tool calls)" in the policy — the only criteria where a flip is possible by design. Non-tool criteria always carry the non-agentic verdict forward unchanged.'
      >
        <div className="flex flex-col gap-4">
          {criterionFlips.map((c) => (
            <details key={c.label} className="rounded-md border border-slate-200 bg-white" open={criterionFlips.length === 1}>
              <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-slate-700 bg-slate-50">{c.label}</summary>
              <div className="p-2">
                <Table
                  columns={["Criterion", "n", "Flipped", "Flip rate", "Transitions"]}
                  rows={c.rows.map((r) => [
                    r.criterion,
                    r.n,
                    r.flippedCount,
                    r.flipRate !== null ? `${(r.flipRate * 100).toFixed(0)}%` : "—",
                    r.transitions.map((t) => `${t.from} → ${t.to} (${t.count})`).join(", ") || "—",
                  ])}
                />
              </div>
            </details>
          ))}
        </div>
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
        title="Domains used during judging"
        note="Which web domains show up during tool use — 'Search result domains' is from search_web results only; 'All domains' is every URL touched by any tool call (fetch_url, check_url_validity, urlscan_check, and any URL embedded in a domain-specific tool's own output)."
      >
        <div className="flex flex-col gap-4">
          {domainUsage.map((d) => (
            <details key={d.label} className="rounded-md border border-slate-200 bg-white" open={domainUsage.length === 1}>
              <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-slate-700 bg-slate-50">
                {d.label}{" "}
                <span className="font-normal text-slate-400">
                  ({d.totalUrlCount} URL touches · {d.distinctUrlCount} distinct URLs · {d.distinctDomainCount} distinct domains)
                </span>
              </summary>
              <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-2">Search result domains</h3>
                  <BarChart data={d.searchDomains.map((c) => ({ label: c.domain, value: c.count }))} unitLabel="results" />
                </div>
                <div>
                  <h3 className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-2">All domains (every tool call)</h3>
                  <BarChart data={d.allDomains.map((c) => ({ label: c.domain, value: c.count }))} unitLabel="occurrences" />
                </div>
              </div>
            </details>
          ))}
        </div>
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

      <Section
        title="Qualitative code frequency"
        note={
          <>
            How often each codebook code has been applied by annotators, grouped by theme — see the{" "}
            <Link href={`/${useCase}/codebook`} className="underline">codebook</Link> to add or refine codes.
          </>
        }
      >
        <BarChart data={codeFrequencyData} unitLabel="applications" />
      </Section>
    </div>
  );
}

function Section({ title, note, children }: { title: string; note?: React.ReactNode; children: React.ReactNode }) {
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

function ComplianceTable({ rows }: { rows: CriterionComplianceRow[] }) {
  if (rows.length === 0) {
    return <p className="text-sm text-slate-400 px-2 py-1">No criteria_verdicts data for this policy variant.</p>;
  }
  return (
    <div className="overflow-x-auto rounded-md border border-slate-200 bg-white">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 bg-slate-50 text-left">
            <th className="px-3 py-2 font-medium text-slate-600 whitespace-nowrap">Criterion</th>
            <th className="px-3 py-2 font-medium text-slate-600 whitespace-nowrap">Tool-tagged</th>
            <th className="px-3 py-2 font-medium text-slate-600 whitespace-nowrap" colSpan={3}>Non-agentic</th>
            <th className="px-3 py-2 font-medium text-slate-600 whitespace-nowrap" colSpan={3}>Agentic / final</th>
          </tr>
          <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs text-slate-500">
            <th className="px-3 py-1"></th>
            <th className="px-3 py-1"></th>
            <th className="px-3 py-1 whitespace-nowrap">Compliant</th>
            <th className="px-3 py-1 whitespace-nowrap">Not fully compliant</th>
            <th className="px-3 py-1 whitespace-nowrap">Rate</th>
            <th className="px-3 py-1 whitespace-nowrap">Compliant</th>
            <th className="px-3 py-1 whitespace-nowrap">Not fully compliant</th>
            <th className="px-3 py-1 whitespace-nowrap">Rate</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.criterion} className="border-b border-slate-100 last:border-0">
              <td className="px-3 py-2 max-w-xs truncate" title={r.criterion}>{r.criterion}</td>
              <td className="px-3 py-2">
                {r.toolTagged ? (
                  <span className="inline-block rounded-full bg-sky-100 text-sky-800 px-2 py-0.5 text-xs font-medium">tool</span>
                ) : (
                  <span className="text-slate-300 text-xs">—</span>
                )}
              </td>
              <td className="px-3 py-2 tabular-nums">{r.nonagentic.COMPLIANT}</td>
              <td className="px-3 py-2 tabular-nums">{r.nonagentic.NOT_FULLY_COMPLIANT}</td>
              <td className="px-3 py-2 tabular-nums">{r.nonagenticComplianceRate !== null ? `${(r.nonagenticComplianceRate * 100).toFixed(0)}%` : "—"}</td>
              <td className="px-3 py-2 tabular-nums">{r.agentic.COMPLIANT}</td>
              <td className="px-3 py-2 tabular-nums">{r.agentic.NOT_FULLY_COMPLIANT}</td>
              <td className="px-3 py-2 tabular-nums">{r.agenticComplianceRate !== null ? `${(r.agenticComplianceRate * 100).toFixed(0)}%` : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
