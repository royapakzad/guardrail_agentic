import type { EvaluationRecord, PolicyVariant } from "@/lib/types";
import {
  buildAgentic,
  buildNonagentic,
  ensureList,
  findPrefixes,
  resolveJudgeModel,
  toBoolOrNull,
  toNumberOrNull,
} from "./shared";

export function adaptHumanitarian(raw: unknown): EvaluationRecord[] {
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
