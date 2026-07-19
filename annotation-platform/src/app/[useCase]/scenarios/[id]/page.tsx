import { notFound } from "next/navigation";
import Link from "next/link";
import { getRecordByIdForDataset, USE_CASES } from "@/lib/adapters";
import { resolveDatasetIdParam } from "@/lib/datasetSelection";
import type { UseCase, PolicyVariant, JudgePass, AgenticPass } from "@/lib/types";
import { ScoreBar, ValidBadge, VerdictBadge } from "@/lib/ui/badges";
import { listAnnotations, listCodebookCodes, listCodeApplications } from "@/lib/db/queries";
import { AnnotationForm } from "./AnnotationForm";
import { QualitativeCodingForm } from "./QualitativeCodingForm";
import { CODING_TARGET_FIELD_LABELS, type CodingTargetField } from "@/lib/annotationOptions";

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

  let annotations: Awaited<ReturnType<typeof listAnnotations>> = [];
  let dbError: string | null = null;
  try {
    annotations = await listAnnotations(useCase, record.id);
  } catch (err) {
    dbError = err instanceof Error ? err.message : "Could not load annotations";
  }

  let codes: Awaited<ReturnType<typeof listCodebookCodes>> = [];
  let codeApplications: Awaited<ReturnType<typeof listCodeApplications>> = [];
  let codingDbError: string | null = null;
  try {
    [codes, codeApplications] = await Promise.all([
      listCodebookCodes(useCase),
      listCodeApplications(useCase, record.id),
    ]);
  } catch (err) {
    codingDbError = err instanceof Error ? err.message : "Could not load qualitative coding data";
  }

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">
          {record.id} <span className="text-slate-400 font-normal uppercase text-base">{record.language}</span>
        </h1>
        <p className="mt-1 text-sm text-slate-600">
          <Link href={`/${useCase}/scenarios?dataset=${datasetId}`} className="underline">back to scenario list</Link>
        </p>
      </div>

      <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="rounded-md border border-slate-200 bg-white p-4">
          <div className="text-xs font-medium uppercase tracking-wide text-slate-500 mb-2">Scenario</div>
          <p className="text-sm whitespace-pre-wrap">{record.scenario}</p>
        </div>
        <div className="rounded-md border border-slate-200 bg-white p-4">
          <div className="text-xs font-medium uppercase tracking-wide text-slate-500 mb-2">
            Assistant response {record.assistantModel && <span className="normal-case font-normal">({record.assistantModel})</span>}
          </div>
          <p className="text-sm whitespace-pre-wrap">{record.assistantResponse}</p>
        </div>
      </section>

      {record.policyVariants.length > 1 && (
        <div className="flex flex-wrap gap-2">
          {record.policyVariants.map((v) => (
            <Link
              key={v.label}
              href={`/${useCase}/scenarios/${encodeURIComponent(record.id)}?lang=${record.language}&dataset=${datasetId}&variant=${encodeURIComponent(v.label)}`}
              className={`rounded-full border px-3 py-1 text-xs ${
                v.label === variant.label
                  ? "border-slate-900 bg-slate-900 text-white"
                  : "border-slate-300 text-slate-600 hover:border-slate-500"
              }`}
            >
              {v.label}
            </Link>
          ))}
        </div>
      )}

      <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <JudgePanel title="Non-agentic (full policy, no tools)" pass={variant.nonagentic} />
        <JudgePanel title="Agentic (split-criteria, tool-verified)" pass={variant.agentic} agentic />
      </section>

      {variant.agentic.toolCallLog.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold mb-2">Tool call log</h2>
          <div className="flex flex-col gap-2">
            {variant.agentic.toolCallLog.map((call, i) => (
              <div key={i} className="rounded border border-slate-200 bg-white p-3 text-xs">
                <div className="flex items-center justify-between gap-2">
                  <div className="font-mono font-medium text-slate-800">
                    {call.call_number ? `#${call.call_number} ` : ""}
                    {call.tool}
                  </div>
                  {call.timestamp && (
                    <span className="text-slate-400 shrink-0">
                      {new Date(call.timestamp).toLocaleTimeString()}
                    </span>
                  )}
                </div>
                {call.check_purpose && (
                  <div className="mt-1 text-slate-500 italic">{call.check_purpose}</div>
                )}
                {call.input && <div className="mt-1 text-slate-600 font-mono break-words">{JSON.stringify(call.input)}</div>}
                {call.output_preview && (
                  <div className="mt-1 text-slate-500 font-mono break-words">{String(call.output_preview).slice(0, 300)}</div>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {variant.agentic.urlChecks.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold mb-2">URL checks</h2>
          <ul className="flex flex-col gap-1 text-sm">
            {variant.agentic.urlChecks.map((u, i) => (
              <li key={i} className="font-mono">
                {u.valid ? "✓" : "✗"} {u.url} {u.status_code ? `(HTTP ${u.status_code})` : ""}
              </li>
            ))}
          </ul>
        </section>
      )}

      {variant.agentic.claimChecks.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold mb-2">Claim checks</h2>
          <ul className="flex flex-col gap-1 text-sm">
            {variant.agentic.claimChecks.map((c, i) => (
              <li key={i}>
                <span className="font-medium">{c.status}</span> — {c.claim}
              </li>
            ))}
          </ul>
        </section>
      )}

      <section>
        <h2 className="text-lg font-semibold mb-1">Qualitative coding</h2>
        <p className="text-xs text-slate-500 mb-3">
          Apply codes from the <Link href={`/${useCase}/codebook`} className="underline">use case&apos;s codebook</Link> to
          specific text in this scenario (thematic-analysis style coding, separate from the structured annotation below).
        </p>
        {codingDbError ? (
          <p className="text-sm text-amber-700 bg-amber-50 border border-amber-300 rounded px-3 py-2">
            Could not load qualitative coding data: {codingDbError}
          </p>
        ) : (
          <>
            {codeApplications.length === 0 ? (
              <p className="text-sm text-slate-500 mb-3">No codes applied to this scenario yet.</p>
            ) : (
              <div className="flex flex-col gap-2 mb-4">
                {codeApplications.map((a) => (
                  <div key={a.id} className="rounded border border-slate-200 bg-white p-3 text-sm">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono font-medium text-slate-800">
                        {a.code_theme ? `${a.code_theme} / ` : ""}{a.code_name}
                      </span>
                      <span className="text-xs text-slate-400 shrink-0">{new Date(a.created_at).toLocaleString()}</span>
                    </div>
                    <div className="mt-1 text-xs text-slate-500">
                      {a.annotator_name} · {CODING_TARGET_FIELD_LABELS[a.target_field as CodingTargetField] ?? a.target_field}
                    </div>
                    {a.quote_text && <p className="mt-1 text-sm italic text-slate-700">“{a.quote_text}”</p>}
                    {a.note && <p className="mt-1 text-sm text-slate-600">{a.note}</p>}
                  </div>
                ))}
              </div>
            )}
            <QualitativeCodingForm
              useCase={useCase}
              scenarioId={record.id}
              language={record.language}
              policyLabel={variant.label}
              codes={codes}
            />
          </>
        )}
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-3">Annotations</h2>
        {dbError ? (
          <p className="text-sm text-amber-700 bg-amber-50 border border-amber-300 rounded px-3 py-2">
            Could not load existing annotations: {dbError}
          </p>
        ) : annotations.length === 0 ? (
          <p className="text-sm text-slate-500">No annotations yet.</p>
        ) : (
          <div className="flex flex-col gap-2 mb-4">
            {annotations.map((a) => (
              <div key={a.id} className="rounded border border-slate-200 bg-white p-3 text-sm">
                <div className="flex items-center justify-between">
                  <span className="font-medium">{a.annotator_name}</span>
                  <span className="text-xs text-slate-400">{new Date(a.created_at).toLocaleString()}</span>
                </div>
                <div className="mt-1 text-xs text-slate-600">
                  {a.agrees_with_verdict === null ? "unsure" : a.agrees_with_verdict ? "agrees" : "disagrees"}
                  {a.confidence ? ` · confidence: ${a.confidence}` : ""}
                  {a.evidence_source_type ? ` · source: ${a.evidence_source_type}` : ""}
                  {a.deduction_reason_category ? ` · reason: ${a.deduction_reason_category}` : ""}
                </div>
                {a.free_text && <p className="mt-1 text-sm">{a.free_text}</p>}
              </div>
            ))}
          </div>
        )}

        <AnnotationForm useCase={useCase} scenarioId={record.id} language={record.language} policyLabel={variant.label} />
      </section>
    </div>
  );
}

function JudgePanel({ title, pass, agentic }: { title: string; pass: JudgePass | AgenticPass; agentic?: boolean }) {
  const ag = agentic ? (pass as AgenticPass) : null;
  return (
    <div className="rounded-md border border-slate-200 bg-white p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-sm">{title}</h3>
        <div className="flex items-center gap-2">
          <ValidBadge valid={pass.valid} />
          <ScoreBar score={pass.score} />
        </div>
      </div>
      {ag && (
        <div className="text-xs text-slate-500">
          {ag.toolCallsMade ?? ag.toolCallLog.length} tool call(s)
          {ag.judgmentTimeS !== null ? ` · ${ag.judgmentTimeS.toFixed(1)}s` : ""}
        </div>
      )}
      {pass.criteriaVerdicts.length > 0 && (
        <div className="flex flex-col gap-1">
          {pass.criteriaVerdicts.map((c, i) => (
            <div key={i} className="flex flex-col gap-0.5 text-xs">
              <div className="flex items-center justify-between gap-2">
                <span className="text-slate-700">{c.criterion}</span>
                <VerdictBadge verdict={c.verdict} />
              </div>
              {Array.isArray(c.tools_used) && c.tools_used.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {c.tools_used.map((t, j) => (
                    <span
                      key={j}
                      className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[10px] text-slate-600"
                    >
                      {String(t)}
                    </span>
                  ))}
                </div>
              )}
              {typeof c.human_review_needed === "string" && c.human_review_needed && (
                <div className="text-amber-800">
                  <span className="font-medium">Review: </span>
                  {c.human_review_needed}
                </div>
              )}
              {typeof c.suggested_improvement === "string" && c.suggested_improvement && (
                <div className="text-slate-600">
                  <span className="font-medium">Suggested fix: </span>
                  {c.suggested_improvement}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
      <p className="text-xs text-slate-600 whitespace-pre-wrap">{pass.explanation}</p>
    </div>
  );
}
