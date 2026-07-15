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

// Humanitarian's field-prefix pattern embeds the assistant judge model at the
// end of the prefix (e.g. "humanitarian_policy_explicit_fa_claude_sonnet_4_6"),
// unlike financial's prefixes. Strip a known model suffix to recover the plain
// policy name; anything not matching one of these is treated as an unknown
// model rather than silently mis-splitting the policy name.
const KNOWN_MODEL_SUFFIXES = ["claude_sonnet_4_6", "gpt_5_nano"];

function splitPolicyAndModel(prefix: string): { policyName: string; judgeModel: string | null } {
  for (const suffix of KNOWN_MODEL_SUFFIXES) {
    if (prefix.endsWith(`_${suffix}`)) {
      return { policyName: prefix.slice(0, -(suffix.length + 1)), judgeModel: suffix };
    }
  }
  return { policyName: prefix, judgeModel: null };
}

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

export function adaptHumanitarian(raw: unknown): EvaluationRecord[] {
  const rows = ensureList<Record<string, unknown>>(raw);
  return rows.map((row) => {
    const prefixes = findPrefixes(row);
    const policyVariants: PolicyVariant[] = prefixes.map((prefix) => {
      const { policyName, judgeModel } = splitPolicyAndModel(prefix);
      return {
        label: prefix,
        policyName,
        judgeModel,
        nonagentic: buildNonagentic(row, prefix),
        agentic: buildAgentic(row, prefix),
        scoreDelta: toNumberOrNull(row[`${prefix}_score_delta`]),
        judgmentChanged: toBoolOrNull(row[`${prefix}_judgment_changed`]),
      };
    });

    return {
      id: String(row.id ?? ""),
      useCase: "humanitarian",
      language: String(row.language ?? "en"),
      scenario: String(row.scenario ?? ""),
      assistantResponse: String(row.assistant_response ?? ""),
      assistantModel: (row.model as string) ?? null,
      scenarioType: (row.scenario_type as string) ?? null,
      testFocus: (row.test_focus as string) ?? null,
      policyVariants,
    };
  });
}
