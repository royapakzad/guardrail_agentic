import type { AgenticPass, EvaluationRecord, JudgePass, PolicyVariant, TokenUsage } from "@/lib/types";
import { ensureList, readClaimChecks, readCriteriaVerdicts, readUrlChecks, scoreToValid, toNumberOrNull } from "./shared";

// Cybersecurity's output format is structurally different from the other two
// use cases: one dict per scenario file (not a flat-keyed row in a list), with
// a bracket-notation key embedding both policy name and judge model
// (`policy_policy_cybersecurity[anthropic:claude-sonnet-4-6]`), and no
// criteria_verdicts/claim_checks/judgment_time_s fields at all — this is an
// earlier-generation log format than the other two use cases produce. Curation
// already merges the per-file dicts into a JSON array; this adapter reads that.
const BRACKET_KEY_RE = /^(.+)\[(.+):(.+)\]$/; // policy_name[provider:model]

function findPolicyKey(record: Record<string, unknown>): string | undefined {
  return Object.keys(record).find((k) => k.startsWith("policy_"));
}

function parseBracketKey(key: string): { policyName: string; judgeModel: string | null } {
  const m = key.match(BRACKET_KEY_RE);
  if (!m) return { policyName: key, judgeModel: null };
  return { policyName: m[1], judgeModel: `${m[2]}:${m[3]}` };
}

function tokens(obj: Record<string, unknown> | undefined, isAgentic: boolean): TokenUsage {
  if (!obj) return { promptTokens: null, completionTokens: null, totalTokens: null };
  return isAgentic
    ? {
        promptTokens: toNumberOrNull(obj.prompt_tokens_total),
        completionTokens: toNumberOrNull(obj.completion_tokens_total),
        totalTokens: toNumberOrNull(obj.total_tokens_used),
        peakPromptTokens: toNumberOrNull(obj.peak_prompt_tokens),
      }
    : {
        promptTokens: toNumberOrNull(obj.prompt_tokens),
        completionTokens: toNumberOrNull(obj.completion_tokens),
        totalTokens: toNumberOrNull(obj.total_tokens),
      };
}

function buildNonagentic(obj: Record<string, unknown> | undefined): JudgePass {
  const score = toNumberOrNull(obj?.score);
  return {
    score,
    valid: scoreToValid(score),
    explanation: String(obj?.explanation ?? ""),
    criteriaVerdicts: readCriteriaVerdicts(obj ?? {}, "criteria_verdicts"),
    tokens: tokens(obj, false),
    judgmentTimeS: null,
  };
}

function buildAgentic(obj: Record<string, unknown> | undefined): AgenticPass {
  const score = toNumberOrNull(obj?.score);
  const toolCalls = ensureList<{ tool: string; input?: Record<string, unknown> }>(obj?.tool_calls);
  return {
    score,
    valid: scoreToValid(score),
    explanation: String(obj?.explanation ?? ""),
    criteriaVerdicts: readCriteriaVerdicts(obj ?? {}, "criteria_verdicts"),
    tokens: tokens(obj, true),
    judgmentTimeS: null,
    toolCallLog: toolCalls.map((t) => ({ tool: t.tool, input: t.input })),
    urlChecks: readUrlChecks(obj ?? {}, "url_checks"),
    claimChecks: readClaimChecks(obj ?? {}, "claim_checks"),
    toolCallsMade: toNumberOrNull(obj?.tool_calls_made) ?? toolCalls.length,
    sourcesUsed: ensureList<string>(obj?.sources_used),
    toolChangedVerdictFor: [],
  };
}

export function adaptCybersecurity(raw: unknown): EvaluationRecord[] {
  const rows = ensureList<Record<string, unknown>>(raw);
  return rows.map((row) => {
    const responseGen = (row.response_generation as Record<string, unknown>) ?? {};
    const policyKey = findPolicyKey(row);
    const policyVariants: PolicyVariant[] = [];

    if (policyKey) {
      const { policyName, judgeModel } = parseBracketKey(policyKey);
      const inner = row[policyKey] as Record<string, unknown>;
      const nonagenticObj = inner?.nonagentic as Record<string, unknown> | undefined;
      const agenticObj = inner?.agentic as Record<string, unknown> | undefined;
      const nonagentic = buildNonagentic(nonagenticObj);
      const agentic = buildAgentic(agenticObj);
      policyVariants.push({
        label: policyKey,
        policyName,
        judgeModel,
        nonagentic,
        agentic,
        scoreDelta: agentic.score !== null && nonagentic.score !== null ? agentic.score - nonagentic.score : null,
        judgmentChanged: agentic.valid !== null && nonagentic.valid !== null ? agentic.valid !== nonagentic.valid : null,
      });
    }

    return {
      id: String(row.scenario_id ?? ""),
      useCase: "cybersecurity",
      language: String(row.language ?? "en"),
      scenario: String(row.scenario_text ?? responseGen.user_message ?? ""),
      assistantResponse: String(responseGen.assistant_response ?? ""),
      assistantModel: (responseGen.model as string) ?? null,
      scenarioType: null,
      testFocus: null,
      policyVariants,
    };
  });
}
