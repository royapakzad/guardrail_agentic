import type { EvaluationRecord, PolicyVariant } from "@/lib/types";
import { buildAgentic, buildNonagentic, ensureList, findPrefixes, resolveJudgeModel, toBoolOrNull, toNumberOrNull } from "./shared";

// Same flat-row shape as humanitarian/financial (run_agentic_comparison.py's
// --guardrail-judges output) -- earlier cybersecurity runs used a bespoke
// per-scenario log format instead, but that predates criteria_verdicts,
// judgment_time_s, and claim_checks, so new runs should use --guardrail-judges
// like the other two use cases rather than reintroducing that format.
export function adaptCybersecurity(raw: unknown): EvaluationRecord[] {
  const rows = ensureList<Record<string, unknown>>(raw);
  return rows.map((row) => {
    const prefixes = findPrefixes(row);
    const policyVariants: PolicyVariant[] = prefixes.map((prefix) => {
      const { policyName, judgeModel } = resolveJudgeModel(row, prefix);
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
      useCase: "cybersecurity",
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
