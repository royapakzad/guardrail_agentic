import { notFound } from "next/navigation";
import Link from "next/link";
import { getRecordsByIdForDataset, USE_CASES } from "@/lib/adapters";
import { resolveDatasetIdParam } from "@/lib/datasetSelection";
import type { UseCase, EvaluationRecord, PolicyVariant, CriterionVerdict, ToolCall, UrlCheck } from "@/lib/types";
import { VerdictBadge, ValidBadge } from "@/lib/ui/badges";
import { UseCaseNav } from "@/lib/ui/UseCaseNav";
import { listAnnotations, listCodebookCodes, listCodeApplications } from "@/lib/db/queries";
import type { Annotation, CodeApplicationWithCode } from "@/lib/db/queries";
import { computeScenarioToolSummary } from "@/lib/metrics";
import { isToolTaggedCriterion, normalizeCriterionName as toCanonicalCriterionName } from "@/lib/policyCriteria";
import { ScenarioReviewForm } from "./ScenarioReviewForm";
import { SavedReviewCard } from "./SavedReviewCard";

function isUseCase(value: string): value is UseCase {
  return (USE_CASES as string[]).includes(value);
}

type Panel = {
  record: EvaluationRecord;
  variant: PolicyVariant;
};

export default async function ScenarioDetailPage({
  params,
  searchParams,
}: {
  params: Promise<{ useCase: string; id: string }>;
  searchParams: Promise<{ variant?: string; dataset?: string }>;
}) {
  const { useCase: useCaseParam, id: idParam } = await params;
  const { variant: variantParam, dataset: datasetParam } = await searchParams;
  if (!isUseCase(useCaseParam)) notFound();
  const useCase = useCaseParam;
  const id = decodeURIComponent(idParam);

  const datasetId = await resolveDatasetIdParam(useCase, datasetParam);
  // All language variants of this one scenario id (e.g. "IR01" in en + fa),
  // shown side by side below so language-related judgment differences can be
  // compared without navigating between pages.
  const records = await getRecordsByIdForDataset(useCase, datasetId, id);
  if (records.length === 0) notFound();

  let annotations: Annotation[] = [];
  let codes: Awaited<ReturnType<typeof listCodebookCodes>> = [];
  let codeApplications: CodeApplicationWithCode[] = [];
  let reviewDbError: string | null = null;
  try {
    [annotations, codes, codeApplications] = await Promise.all([
      listAnnotations(useCase, id),
      listCodebookCodes(useCase),
      listCodeApplications(useCase, id),
    ]);
  } catch (err) {
    reviewDbError = err instanceof Error ? err.message : "Could not load saved reviews";
  }

  // One shared review/codebook section for the whole scenario, not one per
  // language -- a bilingual scenario is one coding unit for an annotator, so
  // it needs exactly one review, not a duplicate per language. Ties to
  // whichever policy variant label the (shared) variant selector points at,
  // falling back to the first record's first variant.
  const reviewPolicyLabel = variantParam ?? records[0].policyVariants[0]?.label ?? "";
  const reviewLanguage = records.map((r) => r.language).join("+");
  const reviewAnnotations = annotations.filter((a) => a.policy_label === reviewPolicyLabel);
  const reviewCodeApplications = codeApplications.filter((a) => a.policy_label === reviewPolicyLabel);
  const annotatorNames = [
    ...new Set([
      ...reviewAnnotations.map((a) => a.annotator_name),
      ...reviewCodeApplications.map((a) => a.annotator_name),
    ]),
  ];
  const reviewGroups = annotatorNames
    .map((name) => {
      const annotation = reviewAnnotations.find((a) => a.annotator_name === name) ?? null;
      const apps = reviewCodeApplications.filter((a) => a.annotator_name === name);
      const latest = [annotation?.updated_at, ...apps.map((a) => a.updated_at)]
        .filter((d): d is string => Boolean(d))
        .sort()
        .at(-1);
      return { annotatorName: name, annotation, codeApplications: apps, latest: latest ?? "" };
    })
    .sort((a, b) => b.latest.localeCompare(a.latest));

  // Section-major layout: every StepSection below lays out ALL languages side
  // by side in one row, instead of each language owning one independent
  // top-to-bottom stack. That's the point -- with a language-major layout, a
  // long EN response pushes EN's judge table lower than FA's, so "agentic" and
  // "non-agentic" and "en" and "fa" stop lining up the moment content length
  // differs. Computing `panels` once, up front, and passing the same array
  // into each section keeps every row aligned at the same scroll position.
  const panels: Panel[] = records.map((record) => ({
    record,
    variant: record.policyVariants.find((v) => v.label === variantParam) ?? record.policyVariants[0],
  }));
  const gridStyle = { gridTemplateColumns: `repeat(${panels.length}, minmax(0, 1fr))` };

  return (
    <div className="flex flex-col gap-8">
      <UseCaseNav useCase={useCase} datasetId={String(datasetId)} />

      <div>
        <h1 className="text-2xl font-semibold tracking-tight">{id}</h1>
        <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
          <Link href={`/${useCase}/scenarios?dataset=${datasetId}`} className="underline">back to scenario list</Link>
          {records.length > 1 && (
            <span className="ml-2 text-slate-400 dark:text-slate-500">
              · {records.length} languages shown side by side: {records.map((r) => r.language.toUpperCase()).join(", ")}
            </span>
          )}
        </p>
      </div>

      {/* Breaks out of the shared <main> max-width -- with N languages times
          a criterion table times a tool-activity table, this page needs real
          width to keep everything on one screen without horizontal scrolling
          inside every card. */}
      <div className="relative left-1/2 right-1/2 -mx-[50vw] w-screen">
        <div className="mx-auto max-w-[1800px] px-6 flex flex-col gap-8">
          <StepSection number={1} title="The case">
            <div className="grid gap-6" style={gridStyle}>
              {panels.map(({ record }, i) => (
                <div key={record.language} className={`flex flex-col gap-4 ${i > 0 ? "md:border-l md:border-slate-200 md:pl-6 dark:md:border-slate-700" : ""}`}>
                  <LanguageChip language={record.language} />
                  <div className="rounded-md border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-900">
                    <div className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-2">Scenario</div>
                    <p className="text-sm whitespace-pre-wrap">{record.scenario}</p>
                  </div>
                  <div className="rounded-md border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-900">
                    <div className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-2">
                      Assistant response {record.assistantModel && <span className="normal-case font-normal">({record.assistantModel})</span>}
                    </div>
                    <p className="text-sm whitespace-pre-wrap">{record.assistantResponse}</p>
                  </div>
                </div>
              ))}
            </div>
          </StepSection>

          <StepSection
            number={2}
            title="Compliance by criterion"
            subtitle="Compliant vs. not fully compliant, before and after tool access -- no scores. A highlighted row means tool evidence changed the verdict for that criterion."
          >
            <div className="grid gap-6" style={gridStyle}>
              {panels.map(({ record, variant }, i) => (
                <div key={record.language} className={`flex flex-col gap-3 ${i > 0 ? "md:border-l md:border-slate-200 md:pl-6 dark:md:border-slate-700" : ""}`}>
                  <div className="flex items-center justify-between gap-2 flex-wrap">
                    <LanguageChip language={record.language} />
                    {record.policyVariants.length > 1 && (
                      <div className="flex flex-wrap gap-2">
                        {record.policyVariants.map((v) => (
                          <Link
                            key={v.label}
                            href={`/${useCase}/scenarios/${encodeURIComponent(record.id)}?dataset=${datasetId}&variant=${encodeURIComponent(v.label)}`}
                            className={`rounded-full border px-3 py-1 text-xs ${
                              v.label === variant.label
                                ? "border-slate-900 bg-slate-900 text-white dark:border-slate-100 dark:bg-slate-100 dark:text-slate-900"
                                : "border-slate-300 text-slate-600 hover:border-slate-500 dark:border-slate-600 dark:text-slate-400 dark:hover:border-slate-400"
                            }`}
                          >
                            {v.label}
                          </Link>
                        ))}
                      </div>
                    )}
                  </div>
                  <ComplianceTable variant={variant} />
                </div>
              ))}
            </div>
          </StepSection>

          <StepSection
            number={3}
            title="Tool activity"
            subtitle="Every tool call the agentic judge made, and what it returned -- shown as a table by default, not tucked behind a disclosure."
          >
            <div className="grid gap-6" style={gridStyle}>
              {panels.map(({ record, variant }, i) => (
                <div key={record.language} className={`flex flex-col gap-3 ${i > 0 ? "md:border-l md:border-slate-200 md:pl-6 dark:md:border-slate-700" : ""}`}>
                  <LanguageChip language={record.language} />
                  <ToolActivity variant={variant} />
                </div>
              ))}
            </div>
          </StepSection>
        </div>
      </div>

      <StepSection
        number={4}
        title="Your review"
        subtitle="One shared review for this scenario, covering every language shown above — record your structured judgment and apply qualitative codes together, then save once. Need a refresher on coding? See the help page."
      >
        {reviewDbError ? (
          <p className="text-sm text-amber-700 bg-amber-50 border border-amber-300 rounded px-3 py-2 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-200">
            Could not load saved reviews: {reviewDbError}
          </p>
        ) : (
          <>
            {reviewGroups.length > 0 && (
              <div className="flex flex-col gap-2">
                <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300">Saved reviews for this policy variant ({reviewGroups.length})</h3>
                {reviewGroups.map((g) => (
                  <SavedReviewCard
                    key={g.annotatorName}
                    annotatorName={g.annotatorName}
                    annotation={g.annotation}
                    codeApplications={g.codeApplications}
                  />
                ))}
              </div>
            )}
            <ScenarioReviewForm useCase={useCase} scenarioId={id} language={reviewLanguage} policyLabel={reviewPolicyLabel} codes={codes} />
          </>
        )}
      </StepSection>
    </div>
  );
}

function LanguageChip({ language }: { language: string }) {
  return (
    <span className="inline-flex w-fit rounded-full bg-slate-900 px-2.5 py-1 text-xs font-semibold uppercase text-white dark:bg-slate-100 dark:text-slate-900">
      {language}
    </span>
  );
}

function StepSection({
  number,
  title,
  subtitle,
  children,
}: {
  number: number;
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="flex flex-col gap-4">
      <div className="flex items-baseline gap-2.5 border-b border-slate-200 dark:border-slate-700 pb-2">
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-slate-900 text-xs font-semibold text-white dark:bg-slate-100 dark:text-slate-900">
          {number}
        </span>
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">{title}</h2>
      </div>
      {subtitle && <p className="-mt-2 text-sm text-slate-600 dark:text-slate-400">{subtitle}</p>}
      {children}
    </section>
  );
}

/**
 * One row per policy criterion, matching the non-agentic (no tools) verdict
 * against the agentic (with tools) verdict for that same criterion. This
 * replaces two separate stacked judge cards -- putting both verdicts in the
 * same row is what actually answers "did tool access change this," instead
 * of asking the reader to hold both cards in their head while scrolling.
 * No score or score arithmetic is shown anywhere in this table, by design.
 */
function ComplianceTable({ variant }: { variant: PolicyVariant }) {
  const agenticByCriterion = new Map(variant.agentic.criteriaVerdicts.map((c) => [c.criterion, c]));
  const changedCriteria = new Set(variant.agentic.toolChangedVerdictFor ?? []);
  const rows = variant.nonagentic.criteriaVerdicts.map((naC) => {
    const canonicalName = toCanonicalCriterionName(variant.policyName, naC.criterion);
    return {
      criterion: naC.criterion,
      nonagentic: naC,
      agentic: agenticByCriterion.get(naC.criterion) ?? naC,
      toolTagged: isToolTaggedCriterion(variant.policyName, canonicalName),
    };
  });

  if (rows.length === 0) {
    return <p className="text-sm text-slate-400 dark:text-slate-500">No criteria returned for this judge/policy.</p>;
  }

  const nonagenticTexts = splitExplanationByCriterion(variant.nonagentic.explanation);
  const agenticTexts = splitExplanationByCriterion(variant.agentic.explanation);
  const toolRows = rows.filter((r) => r.toolTagged);
  const noToolRows = rows.filter((r) => !r.toolTagged);

  return (
    <div className="flex flex-col gap-3">
      <div className="text-xs text-slate-500 dark:text-slate-400">
        {variant.agentic.toolCallsMade ?? variant.agentic.toolCallLog.length} tool call(s)
        {variant.agentic.judgmentTimeS !== null ? ` · ${variant.agentic.judgmentTimeS.toFixed(1)}s` : ""}
      </div>

      {toolRows.length > 0 && (
        <div className="flex flex-col gap-2 rounded-md border-2 border-sky-300 bg-sky-50/30 p-2 dark:border-sky-800 dark:bg-sky-950/10">
          <div className="text-xs font-semibold uppercase tracking-wide text-sky-800 dark:text-sky-300">
            Tool-requiring criteria ({toolRows.length}) — this is what you&apos;re primarily annotating
          </div>
          <CriterionRows rows={toolRows} changedCriteria={changedCriteria} nonagenticTexts={nonagenticTexts} agenticTexts={agenticTexts} />
        </div>
      )}

      {noToolRows.length > 0 && (
        <details className="rounded-md border border-slate-200 dark:border-slate-700" open={toolRows.length === 0}>
          <summary className="cursor-pointer select-none px-3 py-2 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
            Criteria that don&apos;t require tool use ({noToolRows.length})
          </summary>
          <div className="p-2">
            <CriterionRows rows={noToolRows} changedCriteria={changedCriteria} nonagenticTexts={nonagenticTexts} agenticTexts={agenticTexts} />
          </div>
        </details>
      )}
    </div>
  );
}

type ComplianceRow = {
  criterion: string;
  nonagentic: CriterionVerdict;
  agentic: CriterionVerdict;
  toolTagged: boolean;
};

function CriterionRows({
  rows,
  changedCriteria,
  nonagenticTexts,
  agenticTexts,
}: {
  rows: ComplianceRow[];
  changedCriteria: Set<string>;
  nonagenticTexts: Map<string, string>;
  agenticTexts: Map<string, string>;
}) {
  return (
    <div className="overflow-x-auto rounded-md border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-900">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b border-slate-200 bg-slate-50 text-left dark:border-slate-700 dark:bg-slate-800">
            <th className="px-3 py-2 font-medium text-slate-600 dark:text-slate-400">Criterion</th>
            <th className="px-3 py-2 font-medium text-slate-600 dark:text-slate-400 whitespace-nowrap">No tools</th>
            <th className="px-3 py-2 font-medium text-slate-600 dark:text-slate-400 whitespace-nowrap">With tools</th>
            <th className="px-3 py-2 font-medium text-slate-600 dark:text-slate-400">Tools used</th>
            <th className="px-3 py-2 font-medium text-slate-600 dark:text-slate-400">Suggested Review and Improvement</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const changed = changedCriteria.has(row.criterion) || row.nonagentic.verdict !== row.agentic.verdict;
            const key = normalizeCriterionName(row.criterion);
            const nonagenticText = nonagenticTexts.get(key);
            const agenticText = agenticTexts.get(key);
            return (
              <tr
                key={row.criterion}
                className={`border-b border-slate-100 last:border-0 align-top dark:border-slate-800 ${
                  changed ? "bg-sky-50/60 dark:bg-sky-950/20" : ""
                }`}
              >
                <td className="px-3 py-2 font-medium text-slate-800 dark:text-slate-200">
                  {row.criterion}
                  {changed && (
                    <div className="mt-1 inline-flex items-center gap-1 rounded-full bg-sky-100 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-sky-800 dark:bg-sky-950/50 dark:text-sky-300">
                      Changed by tool evidence
                    </div>
                  )}
                </td>
                <td className="px-3 py-2 max-w-sm">
                  <div className="flex flex-col gap-1.5">
                    <VerdictBadge verdict={row.nonagentic.verdict} />
                    {nonagenticText && (
                      <p className="text-xs text-slate-600 dark:text-slate-400 whitespace-pre-wrap break-words">{nonagenticText}</p>
                    )}
                  </div>
                </td>
                <td className="px-3 py-2 max-w-sm">
                  <div className="flex flex-col gap-1.5">
                    <VerdictBadge verdict={row.agentic.verdict} />
                    {agenticText && (
                      <p className="text-xs text-slate-600 dark:text-slate-400 whitespace-pre-wrap break-words">{agenticText}</p>
                    )}
                  </div>
                </td>
                <td className="px-3 py-2">
                  <ToolChips tools={row.agentic.tools_used} />
                </td>
                <td className="px-3 py-2 max-w-md">
                  <Evidence criterion={row.agentic} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** Normalizes a criterion name for matching between criteriaVerdicts entries
 * and the numbered headings inside a raw explanation string -- strips a
 * trailing parenthetical/bracket annotation some judge models append (see
 * agentic_runner.py's own _normalize_criterion_name, which this mirrors) and
 * compares case-insensitively so minor judge formatting drift doesn't break
 * the match. */
function normalizeCriterionName(name: string): string {
  return name
    .replace(/\s*[([][^)\]]*[)\]]\s*$/, "")
    .trim()
    .toUpperCase();
}

/** Splits a judge's free-text explanation into one block of text per
 * criterion, keyed by normalized criterion name, so each policy criterion's
 * assessment can be shown next to its own row instead of as one long
 * undifferentiated paragraph. Explanations follow the numbered format
 * "N. CRITERION NAME: assessment text... → Verdict: ..." repeated once per
 * criterion, followed by a DEDUCTION SUMMARY section -- which is deliberately
 * excluded here (score arithmetic is not shown anywhere in this UI). */
function splitExplanationByCriterion(explanation: string): Map<string, string> {
  const map = new Map<string, string>();
  if (!explanation) return map;

  const summaryIndex = explanation.search(/deduction summary/i);
  const body = summaryIndex >= 0 ? explanation.slice(0, summaryIndex) : explanation;

  const headingRe = /^\s*\d+\.\s+([^\n:]+):/gm;
  const headings: { name: string; start: number; contentStart: number }[] = [];
  let match: RegExpExecArray | null;
  while ((match = headingRe.exec(body)) !== null) {
    headings.push({ name: match[1], start: match.index, contentStart: match.index + match[0].length });
  }

  headings.forEach((h, i) => {
    const end = i + 1 < headings.length ? headings[i + 1].start : body.length;
    const text = body.slice(h.contentStart, end).trim();
    if (text) map.set(normalizeCriterionName(h.name), text);
  });

  return map;
}

function ToolChips({ tools }: { tools?: string[] }) {
  if (!Array.isArray(tools) || tools.length === 0) return <span className="text-slate-300 dark:text-slate-600">—</span>;
  return (
    <div className="flex flex-wrap gap-1">
      {tools.map((t, i) => (
        <span key={i} className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[10px] text-slate-600 dark:bg-slate-800 dark:text-slate-400">
          {t}
        </span>
      ))}
    </div>
  );
}

function Evidence({ criterion }: { criterion: CriterionVerdict }) {
  const review = typeof criterion.human_review_needed === "string" ? criterion.human_review_needed : "";
  const fix = typeof criterion.suggested_improvement === "string" ? criterion.suggested_improvement : "";
  if (!review && !fix) return <span className="text-slate-300 dark:text-slate-600">—</span>;
  return (
    <div className="flex flex-col gap-1 text-xs">
      {review && <div className="text-slate-700 dark:text-slate-300">{review}</div>}
      {fix && (
        <div className="text-amber-800 dark:text-amber-300">
          <span className="font-medium">Fix: </span>
          {fix}
        </div>
      )}
    </div>
  );
}

/**
 * All tool calls for this variant's agentic pass as one always-visible table
 * (no <details> collapse). check_url_validity calls are cross-referenced
 * against urlChecks so the result column shows a VALID/INVALID badge instead
 * of a raw JSON preview.
 */
function ToolActivity({ variant }: { variant: PolicyVariant }) {
  const toolSummary = computeScenarioToolSummary(variant);
  const urlByNormalizedUrl = new Map(
    variant.agentic.urlChecks.map((u) => [normalizeUrl(u.url), u] as const)
  );

  if (toolSummary.totalToolCalls === 0) {
    return <p className="text-sm text-slate-400 dark:text-slate-500">No tool calls for this variant.</p>;
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap gap-2">
        {toolSummary.toolCounts.map((t) => (
          <span
            key={t.tool}
            className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2.5 py-1 text-xs font-mono text-slate-700 dark:bg-slate-800 dark:text-slate-300"
          >
            {t.tool} <span className="font-semibold">×{t.count}</span>
          </span>
        ))}
      </div>

      {toolSummary.domains.length > 0 && (
        <details className="group w-fit rounded-md border border-sky-200 bg-sky-50/40 dark:border-sky-800 dark:bg-sky-950/10">
          <summary className="flex cursor-pointer list-none items-center gap-1.5 select-none rounded-md px-2.5 py-1 text-xs text-sky-800 hover:bg-sky-100 dark:text-sky-300 dark:hover:bg-sky-950/30">
            <span>
              {toolSummary.domains.length} domains · {toolSummary.totalUrlCount} URL touches · {toolSummary.distinctUrlCount} distinct
            </span>
            <span className="text-sky-500 transition-transform group-open:rotate-180 dark:text-sky-400">▾</span>
          </summary>
          <div className="max-h-56 overflow-y-auto border-t border-sky-200 dark:border-sky-800">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="text-left text-slate-500 dark:text-slate-400">
                  <th className="px-2.5 py-1.5 font-medium">Domain</th>
                  <th className="px-2.5 py-1.5 font-medium text-right">Touches</th>
                </tr>
              </thead>
              <tbody>
                {toolSummary.domains.map((d) => (
                  <tr key={d.domain} className="border-t border-sky-100 dark:border-sky-900">
                    <td className="px-2.5 py-1 font-mono text-slate-700 dark:text-slate-300 break-all">{d.domain}</td>
                    <td className="px-2.5 py-1 text-right tabular-nums text-slate-600 dark:text-slate-400">{d.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}

      <div className="overflow-x-auto rounded-md border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-900">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-left dark:border-slate-700 dark:bg-slate-800">
              <th className="px-3 py-2 font-medium text-slate-600 dark:text-slate-400 w-10">#</th>
              <th className="px-3 py-2 font-medium text-slate-600 dark:text-slate-400">Tool</th>
              <th className="px-3 py-2 font-medium text-slate-600 dark:text-slate-400">Input</th>
              <th className="px-3 py-2 font-medium text-slate-600 dark:text-slate-400">Result</th>
            </tr>
          </thead>
          <tbody>
            {variant.agentic.toolCallLog.map((call, i) => (
              <ToolActivityRow key={i} call={call} urlByNormalizedUrl={urlByNormalizedUrl} />
            ))}
          </tbody>
        </table>
      </div>

      {variant.agentic.claimChecks.length > 0 && (
        <div className="overflow-x-auto rounded-md border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-900">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-left dark:border-slate-700 dark:bg-slate-800">
                <th className="px-3 py-2 font-medium text-slate-600 dark:text-slate-400 w-28">Status</th>
                <th className="px-3 py-2 font-medium text-slate-600 dark:text-slate-400">Claim</th>
              </tr>
            </thead>
            <tbody>
              {variant.agentic.claimChecks.map((c, i) => (
                <tr key={i} className="border-b border-slate-100 last:border-0 align-top dark:border-slate-800">
                  <td className="px-3 py-2 whitespace-nowrap">
                    <ClaimStatusBadge status={c.status} />
                  </td>
                  <td className="px-3 py-2 text-slate-700 dark:text-slate-300">{c.claim}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function normalizeUrl(url?: string): string {
  return (url ?? "").trim().replace(/[*`]+$/, "");
}

function ToolActivityRow({
  call,
  urlByNormalizedUrl,
}: {
  call: ToolCall;
  urlByNormalizedUrl: Map<string, UrlCheck>;
}) {
  const inputUrl = typeof call.input?.url === "string" ? call.input.url : undefined;
  const urlCheck = call.tool === "check_url_validity" && inputUrl ? urlByNormalizedUrl.get(normalizeUrl(inputUrl)) : undefined;

  return (
    <tr className="border-b border-slate-100 last:border-0 align-top dark:border-slate-800">
      <td className="px-3 py-2 text-slate-400 dark:text-slate-500 tabular-nums">{call.call_number ?? "—"}</td>
      <td className="px-3 py-2 font-mono text-xs font-medium text-slate-800 dark:text-slate-200 whitespace-nowrap">
        {call.tool}
        {call.check_purpose && <div className="font-sans italic font-normal text-slate-400 dark:text-slate-500">{call.check_purpose}</div>}
      </td>
      <td className="px-3 py-2 font-mono text-xs text-slate-600 dark:text-slate-400 break-words max-w-xs">
        {call.input ? JSON.stringify(call.input) : "—"}
      </td>
      <td className="px-3 py-2 text-xs max-w-md">
        {urlCheck ? (
          <div className="flex flex-col gap-1">
            <ValidBadge valid={urlCheck.valid ?? null} />
            {urlCheck.status_code !== undefined && urlCheck.status_code !== null && (
              <span className="font-mono text-slate-500 dark:text-slate-400">HTTP {urlCheck.status_code}</span>
            )}
          </div>
        ) : (
          <span className="font-mono text-slate-600 dark:text-slate-400 break-words">
            {String(call.output_preview ?? "").slice(0, 220)}
          </span>
        )}
      </td>
    </tr>
  );
}

const CLAIM_STATUS_COLORS: Record<string, string> = {
  verified: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-300",
  contradicted: "bg-red-100 text-red-800 dark:bg-red-950/50 dark:text-red-300",
  unverifiable: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400",
};

function ClaimStatusBadge({ status }: { status: string }) {
  const cls = CLAIM_STATUS_COLORS[status] ?? "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400";
  return <span className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${cls}`}>{status}</span>;
}
