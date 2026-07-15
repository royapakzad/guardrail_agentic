import type { EvaluationRecord } from "@/lib/types";
import { flattenVariants, groupByLabel, mean, round } from "./flatten";

export type AcronymCheckSummary = {
  label: string;
  totalChecks: number;
  meanMatchScore: number | null;
  verdictHintDistribution: { hint: string; count: number }[];
};

type ParsedAcronymOutput = { match_score?: number; verdict_hint?: string };

function parseOutputPreview(preview: string | undefined): ParsedAcronymOutput | null {
  if (!preview) return null;
  try {
    return JSON.parse(preview) as ParsedAcronymOutput;
  } catch {
    return null; // output_preview can be truncated for long tool results
  }
}

/** Descriptive only — this is the check_acronym tool's own heuristic hint, not
 * a verified-correct/incorrect judgment (no gold label exists for this batch
 * data). See analysis_improvemnets doc §3.4 for why verdict_hint should never
 * be read as ground truth on its own. */
export function computeAcronymChecks(records: EvaluationRecord[]): AcronymCheckSummary[] {
  const groups = groupByLabel(flattenVariants(records));
  const summaries: AcronymCheckSummary[] = [];

  for (const [label, items] of groups) {
    const scores: number[] = [];
    const hintCounts = new Map<string, number>();

    for (const { variant } of items) {
      for (const call of variant.agentic.toolCallLog) {
        if (call.tool !== "check_acronym") continue;
        const parsed = parseOutputPreview(call.output_preview);
        if (!parsed) continue;
        if (typeof parsed.match_score === "number") scores.push(parsed.match_score);
        if (parsed.verdict_hint) hintCounts.set(parsed.verdict_hint, (hintCounts.get(parsed.verdict_hint) ?? 0) + 1);
      }
    }

    summaries.push({
      label,
      totalChecks: scores.length,
      meanMatchScore: round(mean(scores)),
      verdictHintDistribution: [...hintCounts.entries()]
        .map(([hint, count]) => ({ hint, count }))
        .sort((a, b) => b.count - a.count),
    });
  }

  return summaries.sort((a, b) => a.label.localeCompare(b.label));
}
