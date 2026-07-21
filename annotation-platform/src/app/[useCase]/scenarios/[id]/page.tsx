import { notFound } from "next/navigation";
import Link from "next/link";
import { getRecordByIdForDataset, USE_CASES } from "@/lib/adapters";
import { resolveDatasetIdParam } from "@/lib/datasetSelection";
import type { UseCase, PolicyVariant, JudgePass, AgenticPass } from "@/lib/types";
import { VerdictBadge } from "@/lib/ui/badges";
import { UseCaseNav } from "@/lib/ui/UseCaseNav";
import { listAnnotations, listCodebookCodes, listCodeApplications } from "@/lib/db/queries";
import type { Annotation, CodeApplicationWithCode } from "@/lib/db/queries";
import { ScenarioReviewForm } from "./ScenarioReviewForm";
import { SavedReviewCard } from "./SavedReviewCard";

function isUseCase(value: string): value is UseCase {
  return (USE_CASES as string[]).includes(value);
}

export default async function ScenarioDetailPage({
  params,
  searchParams,
}: {
  params: Promise<{ useCase: string; id: string }>;
  searchParams: Promise<{ lang?: string; variant?: string; dataset?: string }>;
}) {
  const { useCase: useCaseParam, id: idParam } = await params;
  const { lang, variant: variantParam, dataset: datasetParam } = await searchParams;
  if (!isUseCase(useCaseParam)) notFound();
  const useCase = useCaseParam;
  const id = decodeURIComponent(idParam);

  const datasetId = await resolveDatasetIdParam(useCase, datasetParam);
  const record = await getRecordByIdForDataset(useCase, datasetId, id, lang);
  if (!record) notFound();

  const variant: PolicyVariant =
    record.policyVariants.find((v) => v.label === variantParam) ?? record.policyVariants[0];

  let annotations: Annotation[] = [];
  let codes: Awaited<ReturnType<typeof listCodebookCodes>> = [];
  let codeApplications: CodeApplicationWithCode[] = [];
  let reviewDbError: string | null = null;
  try {
    [annotations, codes, codeApplications] = await Promise.all([
      listAnnotations(useCase, record.id),
      listCodebookCodes(useCase),
      listCodeApplications(useCase, record.id),
    ]);
  } catch (err) {
    reviewDbError = err instanceof Error ? err.message : "Could not load saved reviews";
  }

  const variantAnnotations = annotations.filter((a) => a.policy_label === variant.label);
  const variantCodeApplications = codeApplications.filter((a) => a.policy_label === variant.label);
  const annotatorNames = [
    ...new Set([
      ...variantAnnotations.map((a) => a.annotator_name),
      ...variantCodeApplications.map((a) => a.annotator_name),
    ]),
  ];
  const reviewGroups = annotatorNames
    .map((name) => {
      const annotation = variantAnnotations.find((a) => a.annotator_name === name) ?? null;
      const apps = variantCodeApplications.filter((a) => a.annotator_name === name);
      const latest = [annotation?.updated_at, ...apps.map((a) => a.updated_at)]
        .filter((d): d is string => Boolean(d))
        .sort()
        .at(-1);
      return { annotatorName: name, annotation, codeApplications: apps, latest: latest ?? "" };
    })
    .sort((a, b) => b.latest.localeCompare(a.latest));

  return (
    <div className="flex flex-col gap-8">
      <UseCaseNav useCase={useCase} datasetId={String(datasetId)} />

      <div>
        <h1 className="text-2xl font-semibold tracking-tight">
          {record.id} <span className="text-slate-400 dark:text-slate-500 font-normal uppercase text-base">{record.language}</span>
        </h1>
        <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
          <Link href={`/${useCase}/scenarios?dataset=${datasetId}`} className="underline">back to scenario list</Link>
        </p>
      </div>

      {record.policyVariants.length > 1 && (
        <div className="flex flex-wrap gap-2">
          {record.policyVariants.map((v) => (
            <Link
              key={v.label}
              href={`/${useCase}/scenarios/${encodeURIComponent(record.id)}?lang=${record.language}&dataset=${datasetId}&variant=${encodeURIComponent(v.label)}`}
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

      <StepSection number={1} title="The case">
        <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
        </section>
      </StepSection>

      <StepSection number={2} title="Judge evaluation" subtitle="What the automated judge already found — read this before writing your review below.">
        <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <JudgePanel title="Non-agentic (full policy, no tools)" pass={variant.nonagentic} />
          <JudgePanel title="Agentic (split-criteria, tool-verified)" pass={variant.agentic} agentic />
        </section>

        {variant.agentic.toolCallLog.length > 0 && (
          <details className="rounded-md border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-900">
            <summary className="cursor-pointer px-4 py-2 text-sm font-medium text-slate-700 dark:text-slate-300">
              Tool call log ({variant.agentic.toolCallLog.length})
            </summary>
            <div className="flex flex-col gap-2 p-4 pt-0">
              {variant.agentic.toolCallLog.map((call, i) => (
                <div key={i} className="rounded border border-slate-200 bg-slate-50 p-3 text-xs dark:border-slate-700 dark:bg-slate-800">
                  <div className="flex items-center justify-between gap-2">
                    <div className="font-mono font-medium text-slate-800 dark:text-slate-200">
                      {call.call_number ? `#${call.call_number} ` : ""}
                      {call.tool}
                    </div>
                    {call.timestamp && (
                      <span className="text-slate-400 dark:text-slate-500 shrink-0">{new Date(call.timestamp).toLocaleTimeString()}</span>
                    )}
                  </div>
                  {call.check_purpose && <div className="mt-1 text-slate-500 dark:text-slate-400 italic">{call.check_purpose}</div>}
                  {call.input && <div className="mt-1 text-slate-600 dark:text-slate-400 font-mono break-words">{JSON.stringify(call.input)}</div>}
                  {call.output_preview && (
                    <div className="mt-1 text-slate-500 dark:text-slate-400 font-mono break-words">{String(call.output_preview).slice(0, 300)}</div>
                  )}
                </div>
              ))}
            </div>
          </details>
        )}

        {variant.agentic.urlChecks.length > 0 && (
          <details className="rounded-md border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-900">
            <summary className="cursor-pointer px-4 py-2 text-sm font-medium text-slate-700 dark:text-slate-300">
              URL checks ({variant.agentic.urlChecks.length})
            </summary>
            <ul className="flex flex-col gap-1 p-4 pt-0 text-sm">
              {variant.agentic.urlChecks.map((u, i) => (
                <li key={i} className="font-mono">
                  {u.valid ? "✓" : "✗"} {u.url} {u.status_code ? `(HTTP ${u.status_code})` : ""}
                </li>
              ))}
            </ul>
          </details>
        )}

        {variant.agentic.claimChecks.length > 0 && (
          <details className="rounded-md border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-900">
            <summary className="cursor-pointer px-4 py-2 text-sm font-medium text-slate-700 dark:text-slate-300">
              Claim checks ({variant.agentic.claimChecks.length})
            </summary>
            <ul className="flex flex-col gap-1 p-4 pt-0 text-sm">
              {variant.agentic.claimChecks.map((c, i) => (
                <li key={i}>
                  <span className="font-medium">{c.status}</span> — {c.claim}
                </li>
              ))}
            </ul>
          </details>
        )}
      </StepSection>

      <StepSection
        number={3}
        title="Your review"
        subtitle="Record your structured judgment and apply qualitative codes together, then save once. Need a refresher on coding? See the help page."
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
            <ScenarioReviewForm useCase={useCase} scenarioId={record.id} language={record.language} policyLabel={variant.label} codes={codes} />
          </>
        )}
      </StepSection>
    </div>
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

function JudgePanel({ title, pass, agentic }: { title: string; pass: JudgePass | AgenticPass; agentic?: boolean }) {
  const ag = agentic ? (pass as AgenticPass) : null;
  const compliantCount = pass.criteriaVerdicts.filter((c) => c.verdict === "COMPLIANT").length;
  const total = pass.criteriaVerdicts.length;
  return (
    <div className="rounded-md border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-900 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-sm dark:text-slate-100">{title}</h3>
        {total > 0 && (
          <span className="text-xs text-slate-500 dark:text-slate-400 tabular-nums">{compliantCount}/{total} compliant</span>
        )}
      </div>
      {ag && (
        <div className="text-xs text-slate-500 dark:text-slate-400">
          {ag.toolCallsMade ?? ag.toolCallLog.length} tool call(s)
          {ag.judgmentTimeS !== null ? ` · ${ag.judgmentTimeS.toFixed(1)}s` : ""}
        </div>
      )}
      {pass.criteriaVerdicts.length > 0 && (
        <div className="flex flex-col gap-1">
          {pass.criteriaVerdicts.map((c, i) => (
            <div key={i} className="flex flex-col gap-0.5 text-xs">
              <div className="flex items-center justify-between gap-2">
                <span className="text-slate-700 dark:text-slate-300">{c.criterion}</span>
                <VerdictBadge verdict={c.verdict} />
              </div>
              {Array.isArray(c.tools_used) && c.tools_used.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {c.tools_used.map((t, j) => (
                    <span
                      key={j}
                      className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[10px] text-slate-600 dark:bg-slate-800 dark:text-slate-400"
                    >
                      {String(t)}
                    </span>
                  ))}
                </div>
              )}
              {typeof c.human_review_needed === "string" && c.human_review_needed && (
                <div className="text-amber-800 dark:text-amber-300">
                  <span className="font-medium">Review: </span>
                  {c.human_review_needed}
                </div>
              )}
              {typeof c.suggested_improvement === "string" && c.suggested_improvement && (
                <div className="text-slate-600 dark:text-slate-400">
                  <span className="font-medium">Suggested fix: </span>
                  {c.suggested_improvement}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
      <p className="text-xs text-slate-600 dark:text-slate-400 whitespace-pre-wrap">{pass.explanation}</p>
    </div>
  );
}
