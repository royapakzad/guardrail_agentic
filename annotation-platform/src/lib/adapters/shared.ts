import type {
  ClaimCheck,
  CriterionVerdict,
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
