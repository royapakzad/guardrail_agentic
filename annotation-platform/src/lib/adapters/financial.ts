import type { AgenticPass, EvaluationRecord, JudgePass, PolicyVariant } from "@/lib/types";
import {
  ensureList,
  readClaimChecks,
  readCriteriaVerdicts,
  readToolCallLog,
  readTokenUsage,
  readUrlChecks,
  scoreToValid,
  toBoolOrNull,
  toNumberOrNull,
} from "./shared";

// Financial's field-prefix pattern is the plain policy label — no judge model
// embedded (unlike humanitarian). The judge model is instead a single
// per-record field (guardrail_model), applied uniformly across all of that
// record's policy variants.
function findPrefixes(record: Record<string, unknown>): string[] {
  const prefixes = new Set<string>();
  for (const key of Object.keys(record)) {
    const m = key.match(/^(.*)_nonagentic_score$/);
    if (m) prefixes.add(m[1]);
  }
  return [...prefixes];
}

function buildNonagentic(record: Record<string, unknown>, prefix: string): JudgePass {
  const score = toNumberOrNull(record[`${prefix}_nonagentic_score`]);
  return {
    score,
    valid: scoreToValid(score),
    explanation: String(record[`${prefix}_nonagentic_explanation`] ?? ""),
    overallVerdict: record[`${prefix}_nonagentic_overall_verdict`] as string | undefined,
    confidence: record[`${prefix}_nonagentic_confidence`] as string | undefined,
    criteriaVerdicts: readCriteriaVerdicts(record, `${prefix}_nonagentic_criteria_verdicts`),
    tokens: readTokenUsage(record, `${prefix}_nonagentic`),
    judgmentTimeS: toNumberOrNull(record[`${prefix}_nonagentic_judgment_time_s`]),
  };
}

function buildAgentic(record: Record<string, unknown>, prefix: string): AgenticPass {
  const score = toNumberOrNull(record[`${prefix}_agentic_score`]);
  return {
    score,
    valid: scoreToValid(score),
    explanation: String(record[`${prefix}_agentic_explanation`] ?? ""),
    overallVerdict: record[`${prefix}_agentic_overall_verdict`] as string | undefined,
    confidence: record[`${prefix}_agentic_confidence`] as string | undefined,
    criteriaVerdicts: readCriteriaVerdicts(record, `${prefix}_agentic_criteria_verdicts`),
    tokens: readTokenUsage(record, `${prefix}_agentic`),
    judgmentTimeS: toNumberOrNull(record[`${prefix}_agentic_judgment_time_s`]),
    toolCallLog: readToolCallLog(record, `${prefix}_agentic_tool_call_log`),
    urlChecks: readUrlChecks(record, `${prefix}_agentic_url_checks`),
    claimChecks: readClaimChecks(record, `${prefix}_agentic_claim_checks`),
    toolCallsMade: toNumberOrNull(record[`${prefix}_agentic_tool_calls_made`]),
    sourcesUsed: ensureList<string>(record[`${prefix}_agentic_sources_used`]),
    toolChangedVerdictFor: ensureList<string>(record[`${prefix}_agentic_tool_changed_verdict_for`]),
  };
}

export function adaptFinancial(raw: unknown): EvaluationRecord[] {
  const rows = ensureList<Record<string, unknown>>(raw);
  return rows.map((row) => {
    const judgeModel = (row.guardrail_model as string) ?? null;
    const prefixes = findPrefixes(row);
    const policyVariants: PolicyVariant[] = prefixes.map((prefix) => ({
      label: prefix,
      policyName: prefix,
      judgeModel,
      nonagentic: buildNonagentic(row, prefix),
      agentic: buildAgentic(row, prefix),
      scoreDelta: toNumberOrNull(row[`${prefix}_score_delta`]),
      judgmentChanged: toBoolOrNull(row[`${prefix}_judgment_changed`]),
    }));

    return {
      id: String(row.id ?? ""),
      useCase: "financial",
      language: String(row.language ?? "en"),
      scenario: String(row.scenario ?? ""),
      assistantResponse: String(row.assistant_response ?? ""),
      assistantModel: (row.model as string) ?? null,
      scenarioType: null,
      testFocus: null,
      policyVariants,
    };
  });
}
