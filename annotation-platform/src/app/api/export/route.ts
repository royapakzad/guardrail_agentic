import { NextRequest, NextResponse } from "next/server";
import { getRecordsForDataset, USE_CASES } from "@/lib/adapters";
import { resolveDatasetIdParam } from "@/lib/datasetSelection";
import { listAnnotationsForUseCase, listCodeApplicationsForUseCase } from "@/lib/db/queries";
import type { AgenticPass, EvaluationRecord, PolicyVariant, UseCase } from "@/lib/types";

function isUseCase(value: unknown): value is UseCase {
  return typeof value === "string" && (USE_CASES as string[]).includes(value);
}

function csvValue(value: unknown): string {
  const s = value === null || value === undefined ? "" : String(value);
  if (/[",\n]/.test(s)) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

function csvRow(values: unknown[]): string {
  return values.map(csvValue).join(",") + "\r\n";
}

/** Compact "CRITERION: VERDICT; CRITERION2: VERDICT2" summary from a policy
 * variant's final (agentic/merged) criteria verdicts -- the same list shown
 * on the scenario detail page. */
function complianceSummary(variant: PolicyVariant): string {
  return variant.agentic.criteriaVerdicts.map((c) => `${c.criterion}: ${c.verdict}`).join("; ");
}

/** "tool×count" per distinct tool name called during the agentic pass, e.g.
 * "check_url_validity×3; web_search×1" -- the same tool names shown as chips
 * on the scenario detail page's Tool activity section. */
function toolsUsedSummary(agentic: AgenticPass): string {
  const counts = new Map<string, number>();
  for (const call of agentic.toolCallLog) {
    counts.set(call.tool, (counts.get(call.tool) ?? 0) + 1);
  }
  return [...counts.entries()].map(([tool, count]) => `${tool}×${count}`).join("; ");
}

/** "url: VALID (200)" / "url: INVALID (404)" per URL the agentic judge
 * checked -- mirrors the ValidBadge shown per tool call on the scenario
 * detail page. */
function urlChecksSummary(agentic: AgenticPass): string {
  return agentic.urlChecks
    .map((u) => {
      const validity = u.valid === true ? "VALID" : u.valid === false ? "INVALID" : "UNKNOWN";
      const status = u.status_code !== undefined && u.status_code !== null ? ` (${u.status_code})` : "";
      return `${u.url ?? ""}: ${validity}${status}`;
    })
    .join(" | ");
}

/** Per-language view of one scenario's case text, judge explanations, and
 * quantitative judge data -- mirrors the en / non-en split the review form and
 * annotations table already use (judgment_alignment_en/non_en etc.), rather
 * than inventing a third bucketing scheme. */
type LanguageView = {
  language: string;
  policyName: string;
  scenarioText: string;
  assistantResponse: string;
  assistantModel: string | null;
  judgeModel: string | null;
  nonagenticExplanation: string;
  agenticExplanation: string;
  criteriaCompliantCount: number | "";
  criteriaTotalCount: number | "";
  complianceSummary: string;
  toolCallsMade: number | null;
  toolsUsed: string;
  sourcesUsed: string;
  urlChecksSummary: string;
  criteriaChangedByTools: string;
  nonagenticTimeS: number | null;
  agenticTimeS: number | null;
  nonagenticTotalTokens: number | null;
  agenticTotalTokens: number | null;
};

function buildLanguageView(record: EvaluationRecord, policyLabel: string): LanguageView | null {
  const variant = record.policyVariants.find((v) => v.label === policyLabel);
  if (!variant) return null;
  return {
    language: record.language,
    policyName: variant.policyName,
    scenarioText: record.scenario,
    assistantResponse: record.assistantResponse,
    assistantModel: record.assistantModel ?? null,
    judgeModel: variant.judgeModel,
    nonagenticExplanation: variant.nonagentic.explanation,
    agenticExplanation: variant.agentic.explanation,
    criteriaCompliantCount: variant.agentic.criteriaVerdicts.filter((c) => c.verdict === "COMPLIANT").length,
    criteriaTotalCount: variant.agentic.criteriaVerdicts.length,
    complianceSummary: complianceSummary(variant),
    toolCallsMade: variant.agentic.toolCallsMade,
    toolsUsed: toolsUsedSummary(variant.agentic),
    sourcesUsed: variant.agentic.sourcesUsed.join("; "),
    urlChecksSummary: urlChecksSummary(variant.agentic),
    criteriaChangedByTools: (variant.agentic.toolChangedVerdictFor ?? []).join("; "),
    nonagenticTimeS: variant.nonagentic.judgmentTimeS,
    agenticTimeS: variant.agentic.judgmentTimeS,
    nonagenticTotalTokens: variant.nonagentic.tokens.totalTokens,
    agenticTotalTokens: variant.agentic.tokens.totalTokens,
  };
}

type ExportRow = {
  scenarioId: string;
  policyVariant: string;
  annotator: string;
  en: LanguageView | null;
  nonEn: LanguageView | null;
  judgmentAlignmentEn: string | null;
  alignmentExplanationEn: string | null;
  judgmentAlignmentNonEn: string | null;
  alignmentExplanationNonEn: string | null;
  evidenceSourceType: string | null;
  deductionReasonCategory: string | null;
  confidence: string | null;
  freeText: string | null;
  qualitativeCodes: { theme: string | null; name: string; targetField: string; quoteText: string | null; note: string | null }[];
  annotationCreatedAt: string | null;
  annotationUpdatedAt: string | null;
};

export async function GET(request: NextRequest) {
  const useCase = request.nextUrl.searchParams.get("useCase");
  const datasetParam = request.nextUrl.searchParams.get("dataset") ?? undefined;
  const format = request.nextUrl.searchParams.get("format") === "json" ? "json" : "csv";
  if (!isUseCase(useCase)) {
    return NextResponse.json({ error: "useCase query param is required" }, { status: 400 });
  }

  let rows: ExportRow[];
  try {
    const datasetId = await resolveDatasetIdParam(useCase, datasetParam);
    const [records, annotations, codeApplications] = await Promise.all([
      getRecordsForDataset(useCase, datasetId),
      listAnnotationsForUseCase(useCase, String(datasetId)),
      listCodeApplicationsForUseCase(useCase, String(datasetId)),
    ]);

    // Every language variant of one scenario id (e.g. "IR01" in en + fa).
    const recordsByScenario = new Map<string, EvaluationRecord[]>();
    for (const record of records) {
      const list = recordsByScenario.get(record.id) ?? [];
      list.push(record);
      recordsByScenario.set(record.id, list);
    }

    // One export row per (scenario, policy_label, annotator) -- an annotation
    // and its qualitative codes are saved once per scenario (covering every
    // language shown on that scenario's page together), not once per
    // language, so this is the row unit that actually matches the data model.
    type RowKey = string;
    const rowKeys = new Set<RowKey>();
    const keyOf = (scenarioId: string, policyLabel: string, annotator: string) => `${scenarioId}|${policyLabel}|${annotator}`;

    for (const a of annotations) rowKeys.add(keyOf(a.scenario_id, a.policy_label, a.annotator_name));
    for (const c of codeApplications) rowKeys.add(keyOf(c.scenario_id, c.policy_label, c.annotator_name));

    rows = [...rowKeys].map((key) => {
      const [scenarioId, policyLabel, annotator] = key.split("|");
      const scenarioRecords = recordsByScenario.get(scenarioId) ?? [];
      const enRecord = scenarioRecords.find((r) => r.language === "en");
      const nonEnRecord = scenarioRecords.find((r) => r.language !== "en");

      const annotation = annotations.find(
        (a) => a.scenario_id === scenarioId && a.policy_label === policyLabel && a.annotator_name === annotator
      );
      const codes = codeApplications.filter(
        (c) => c.scenario_id === scenarioId && c.policy_label === policyLabel && c.annotator_name === annotator
      );

      return {
        scenarioId,
        policyVariant: policyLabel,
        annotator,
        en: enRecord ? buildLanguageView(enRecord, policyLabel) : null,
        nonEn: nonEnRecord ? buildLanguageView(nonEnRecord, policyLabel) : null,
        judgmentAlignmentEn: annotation?.judgment_alignment_en ?? null,
        alignmentExplanationEn: annotation?.alignment_explanation_en ?? null,
        judgmentAlignmentNonEn: annotation?.judgment_alignment_non_en ?? null,
        alignmentExplanationNonEn: annotation?.alignment_explanation_non_en ?? null,
        evidenceSourceType: annotation?.evidence_source_type ?? null,
        deductionReasonCategory: annotation?.deduction_reason_category ?? null,
        confidence: annotation?.confidence ?? null,
        freeText: annotation?.free_text ?? null,
        qualitativeCodes: codes.map((c) => ({
          theme: c.code_theme,
          name: c.code_name,
          targetField: c.target_field,
          quoteText: c.quote_text,
          note: c.note,
        })),
        annotationCreatedAt: annotation?.created_at ?? null,
        annotationUpdatedAt: annotation?.updated_at ?? null,
      };
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected error building export";
    return NextResponse.json({ error: message }, { status: 500 });
  }

  const filename = `${useCase}_annotations_export.${format}`;

  if (format === "json") {
    return new NextResponse(JSON.stringify(rows, null, 2), {
      status: 200,
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        "Content-Disposition": `attachment; filename="${filename}"`,
      },
    });
  }

  const languageColumns = (suffix: string) => [
    `language_${suffix}`,
    `policy_file_${suffix}`,
    `scenario_${suffix}`,
    `assistant_response_${suffix}`,
    `assistant_model_${suffix}`,
    `judge_model_${suffix}`,
    `nonagentic_judge_explanation_${suffix}`,
    `agentic_judge_explanation_${suffix}`,
    `criteria_compliant_count_${suffix}`,
    `criteria_total_count_${suffix}`,
    `criteria_compliance_summary_${suffix}`,
    `agentic_tool_calls_made_${suffix}`,
    `agentic_tools_used_${suffix}`,
    `agentic_sources_used_${suffix}`,
    `agentic_url_checks_${suffix}`,
    `criteria_changed_by_tools_${suffix}`,
    `nonagentic_time_s_${suffix}`,
    `agentic_time_s_${suffix}`,
    `nonagentic_total_tokens_${suffix}`,
    `agentic_total_tokens_${suffix}`,
  ];

  const header = [
    "scenario_id", "policy_variant", "annotator",
    ...languageColumns("en"),
    ...languageColumns("non_en"),
    "judgment_alignment_en", "alignment_explanation_en",
    "judgment_alignment_non_en", "alignment_explanation_non_en",
    "evidence_source_type", "deduction_reason_category", "confidence", "free_text",
    "qualitative_codes",
    "annotation_created_at", "annotation_updated_at",
  ];

  const languageValues = (view: LanguageView | null) => [
    view?.language ?? "",
    view?.policyName ?? "",
    view?.scenarioText ?? "",
    view?.assistantResponse ?? "",
    view?.assistantModel ?? "",
    view?.judgeModel ?? "",
    view?.nonagenticExplanation ?? "",
    view?.agenticExplanation ?? "",
    view?.criteriaCompliantCount ?? "",
    view?.criteriaTotalCount ?? "",
    view?.complianceSummary ?? "",
    view?.toolCallsMade ?? "",
    view?.toolsUsed ?? "",
    view?.sourcesUsed ?? "",
    view?.urlChecksSummary ?? "",
    view?.criteriaChangedByTools ?? "",
    view?.nonagenticTimeS ?? "",
    view?.agenticTimeS ?? "",
    view?.nonagenticTotalTokens ?? "",
    view?.agenticTotalTokens ?? "",
  ];

  let csv = csvRow(header);
  for (const row of rows) {
    const codesSummary = row.qualitativeCodes
      .map((c) => `${c.theme ? `${c.theme}/` : ""}${c.name} [${c.targetField}]${c.quoteText ? `: "${c.quoteText}"` : ""}${c.note ? ` (${c.note})` : ""}`)
      .join(" | ");

    csv += csvRow([
      row.scenarioId, row.policyVariant, row.annotator,
      ...languageValues(row.en),
      ...languageValues(row.nonEn),
      row.judgmentAlignmentEn ?? "",
      row.alignmentExplanationEn ?? "",
      row.judgmentAlignmentNonEn ?? "",
      row.alignmentExplanationNonEn ?? "",
      row.evidenceSourceType ?? "",
      row.deductionReasonCategory ?? "",
      row.confidence ?? "",
      row.freeText ?? "",
      codesSummary,
      row.annotationCreatedAt ?? "",
      row.annotationUpdatedAt ?? "",
    ]);
  }

  return new NextResponse(csv, {
    status: 200,
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": `attachment; filename="${filename}"`,
    },
  });
}
