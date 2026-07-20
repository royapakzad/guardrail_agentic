import { NextRequest, NextResponse } from "next/server";
import { getRecordsForDataset, USE_CASES } from "@/lib/adapters";
import { resolveDatasetIdParam } from "@/lib/datasetSelection";
import { listAnnotationsForUseCase, listCodeApplicationsForUseCase } from "@/lib/db/queries";
import type { UseCase, PolicyVariant } from "@/lib/types";

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

export async function GET(request: NextRequest) {
  const useCase = request.nextUrl.searchParams.get("useCase");
  const datasetParam = request.nextUrl.searchParams.get("dataset") ?? undefined;
  if (!isUseCase(useCase)) {
    return NextResponse.json({ error: "useCase query param is required" }, { status: 400 });
  }

  let csv: string;
  try {
    const datasetId = await resolveDatasetIdParam(useCase, datasetParam);
    const [records, annotations, codeApplications] = await Promise.all([
      getRecordsForDataset(useCase, datasetId),
      listAnnotationsForUseCase(useCase),
      listCodeApplicationsForUseCase(useCase),
    ]);

    // scenario_id|language|policy_label -> {record, variant}, for pulling
    // quantitative data (criteria compliance, tool counts, times, tokens)
    // live from the dataset rather than duplicating it into Postgres.
    const variantLookup = new Map<string, { record: (typeof records)[number]; variant: PolicyVariant }>();
    for (const record of records) {
      for (const variant of record.policyVariants) {
        variantLookup.set(`${record.id}|${record.language}|${variant.label}`, { record, variant });
      }
    }

    // One export "unit" per (scenario, language, policy_label, annotator) --
    // an annotator may have filled in the structured annotation, applied
    // codes, or both; every combination that has either shows up as one row.
    type RowKey = string;
    const rowKeys = new Set<RowKey>();
    const keyOf = (scenarioId: string, language: string, policyLabel: string, annotator: string) =>
      `${scenarioId}|${language}|${policyLabel}|${annotator}`;

    for (const a of annotations) rowKeys.add(keyOf(a.scenario_id, a.language, a.policy_label, a.annotator_name));
    for (const c of codeApplications) rowKeys.add(keyOf(c.scenario_id, c.language, c.policy_label, c.annotator_name));

    const header = [
      "scenario_id", "language", "policy_variant", "annotator",
      "criteria_compliant_count", "criteria_total_count", "criteria_compliance_summary",
      "agentic_tool_calls_made", "nonagentic_time_s", "agentic_time_s",
      "nonagentic_total_tokens", "agentic_total_tokens",
      "agrees_with_verdict", "disagreement_reason", "evidence_source_type",
      "deduction_reason_category", "evidentiary_attribution_present", "confidence", "free_text",
      "qualitative_codes",
      "annotation_created_at", "annotation_updated_at",
    ];

    let csvOut = csvRow(header);

    for (const key of rowKeys) {
      const [scenarioId, language, policyLabel, annotator] = key.split("|");
      const lookup = variantLookup.get(`${scenarioId}|${language}|${policyLabel}`);
      const variant = lookup?.variant;

      const compliantCount = variant ? variant.agentic.criteriaVerdicts.filter((c) => c.verdict === "COMPLIANT").length : "";
      const totalCount = variant ? variant.agentic.criteriaVerdicts.length : "";

      const annotation = annotations.find(
        (a) => a.scenario_id === scenarioId && a.language === language && a.policy_label === policyLabel && a.annotator_name === annotator
      );
      const codes = codeApplications.filter(
        (c) => c.scenario_id === scenarioId && c.language === language && c.policy_label === policyLabel && c.annotator_name === annotator
      );
      const codesSummary = codes
        .map((c) => `${c.code_theme ? `${c.code_theme}/` : ""}${c.code_name} [${c.target_field}]${c.quote_text ? `: "${c.quote_text}"` : ""}${c.note ? ` (${c.note})` : ""}`)
        .join(" | ");

      csvOut += csvRow([
        scenarioId, language, policyLabel, annotator,
        compliantCount, totalCount, variant ? complianceSummary(variant) : "",
        variant?.agentic.toolCallsMade ?? "",
        variant?.nonagentic.judgmentTimeS ?? "",
        variant?.agentic.judgmentTimeS ?? "",
        variant?.nonagentic.tokens.totalTokens ?? "",
        variant?.agentic.tokens.totalTokens ?? "",
        annotation?.agrees_with_verdict ?? "",
        annotation?.disagreement_reason ?? "",
        annotation?.evidence_source_type ?? "",
        annotation?.deduction_reason_category ?? "",
        annotation?.evidentiary_attribution_present ?? "",
        annotation?.confidence ?? "",
        annotation?.free_text ?? "",
        codesSummary,
        annotation?.created_at ?? "",
        annotation?.updated_at ?? "",
      ]);
    }

    csv = csvOut;
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected error building export";
    return NextResponse.json({ error: message }, { status: 500 });
  }

  return new NextResponse(csv, {
    status: 200,
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": `attachment; filename="${useCase}_annotations_export.csv"`,
    },
  });
}
