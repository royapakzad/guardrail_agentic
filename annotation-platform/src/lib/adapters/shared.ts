import type {
  AgenticPass,
  ClaimCheck,
  CriterionVerdict,
  JudgePass,
  TokenUsage,
  ToolCall,
  UrlCheck,
} from "@/lib/types";

/** Matches visualize_results.py's _VALID_THRESHOLD — a pipeline quirk where the
 * stored `valid` field is sometimes wrong, so downstream consumers recompute it
 * from score instead of trusting the stored value. */
export const VALID_THRESHOLD = 0.6;

export function scoreToValid(score: number | null): boolean | null {
  if (score === null || Number.isNaN(score)) return null;
  return score > VALID_THRESHOLD;
}

/** Some historical output files store list-shaped fields (criteria_verdicts,
 * tool_call_log, url_checks, claim_checks) as JSON-encoded strings instead of
 * actual arrays — ported from visualize_results.py's ensure_list(). */
export function ensureList<T>(value: unknown): T[] {
  if (Array.isArray(value)) return value as T[];
  if (typeof value === "string" && value.trim().length > 0) {
    try {
      const parsed = JSON.parse(value);
      return Array.isArray(parsed) ? (parsed as T[]) : [];
    } catch {
      return [];
    }
  }
  return [];
}

export function toNumberOrNull(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  const n = typeof value === "number" ? value : Number(value);
  return Number.isNaN(n) ? null : n;
}

export function toBoolOrNull(value: unknown): boolean | null {
  if (value === null || value === undefined) return null;
  return Boolean(value);
}

export function readTokenUsage(
  record: Record<string, unknown>,
  prefix: string
): TokenUsage {
  return {
    promptTokens: toNumberOrNull(record[`${prefix}_prompt_tokens`] ?? record[`${prefix}_prompt_tokens_total`]),
    completionTokens: toNumberOrNull(record[`${prefix}_completion_tokens`] ?? record[`${prefix}_completion_tokens_total`]),
    totalTokens: toNumberOrNull(record[`${prefix}_total_tokens`]),
    peakPromptTokens: toNumberOrNull(record[`${prefix}_peak_prompt_tokens`]),
  };
}

export function readCriteriaVerdicts(record: Record<string, unknown>, key: string): CriterionVerdict[] {
  return ensureList<CriterionVerdict>(record[key]);
}

export function readToolCallLog(record: Record<string, unknown>, key: string): ToolCall[] {
  return ensureList<ToolCall>(record[key]);
}

export function readUrlChecks(record: Record<string, unknown>, key: string): UrlCheck[] {
  return ensureList<UrlCheck>(record[key]);
}

export function readClaimChecks(record: Record<string, unknown>, key: string): ClaimCheck[] {
  return ensureList<ClaimCheck>(record[key]);
}

// Shared by every "flat-row" use case adapter (humanitarian, financial,
// cybersecurity) -- run_agentic_comparison.py's --guardrail-judges output
// writes one flat-keyed row per scenario with a `{prefix}_nonagentic_*` /
// `{prefix}_agentic_*` column family per policy variant, so parsing it is
// identical regardless of domain.

/** run_agentic_comparison.py's multi-judge column prefix embeds the judge
 * model at the end (e.g. "policy_cybersecurity_explicit_claude_sonnet_4_6")
 * -- strip a known model suffix to recover the plain policy name. Anything
 * not matching one of these is treated as an unknown model rather than
 * silently mis-splitting the policy name. */
export const KNOWN_MODEL_SUFFIXES = ["claude_sonnet_4_6", "gpt_5_nano"];

function splitPolicyAndModel(prefix: string): { policyName: string; judgeModel: string | null } {
  for (const suffix of KNOWN_MODEL_SUFFIXES) {
    if (prefix.endsWith(`_${suffix}`)) {
      return { policyName: prefix.slice(0, -(suffix.length + 1)), judgeModel: suffix };
    }
  }
  return { policyName: prefix, judgeModel: null };
}

/** Two output shapes exist in the wild for run_agentic_comparison.py's
 * --guardrail-judges mode: the per-judge extraction (`_extract_judge_rows`)
 * writes a flat `guardrail_model` field per record with plain policy-name
 * prefixes; the combined "mega" file instead embeds the judge model at the
 * end of the prefix itself with no separate field. Prefer the flat field
 * when present, falling back to suffix-splitting the prefix otherwise, so
 * either file shape resolves the same policy name + judge model. */
export function resolveJudgeModel(row: Record<string, unknown>, prefix: string): { policyName: string; judgeModel: string | null } {
  if (typeof row.guardrail_model === "string") {
    return { policyName: prefix, judgeModel: row.guardrail_model };
  }
  return splitPolicyAndModel(prefix);
}

export function findPrefixes(record: Record<string, unknown>): string[] {
  const prefixes = new Set<string>();
  for (const key of Object.keys(record)) {
    const m = key.match(/^(.*)_nonagentic_score$/);
    if (m) prefixes.add(m[1]);
  }
  return [...prefixes];
}

export function buildNonagentic(record: Record<string, unknown>, prefix: string): JudgePass {
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

export function buildAgentic(record: Record<string, unknown>, prefix: string): AgenticPass {
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
