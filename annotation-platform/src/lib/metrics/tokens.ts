import type { EvaluationRecord } from "@/lib/types";
import { flattenVariants, groupByLabel, mean, round } from "./flatten";

export type TokenSummary = {
  label: string;
  n: number;
  meanNonagenticTotal: number | null;
  meanAgenticTotal: number | null;
  meanAgenticPeakPrompt: number | null;
};

export function computeTokenUsage(records: EvaluationRecord[]): TokenSummary[] {
  const groups = groupByLabel(flattenVariants(records));
  const summaries: TokenSummary[] = [];

  for (const [label, items] of groups) {
    const naTotal = items.map((i) => i.variant.nonagentic.tokens.totalTokens).filter((v): v is number => v !== null);
    const agTotal = items.map((i) => i.variant.agentic.tokens.totalTokens).filter((v): v is number => v !== null);
    const agPeak = items
      .map((i) => i.variant.agentic.tokens.peakPromptTokens ?? null)
      .filter((v): v is number => v !== null);

    summaries.push({
      label,
      n: items.length,
      meanNonagenticTotal: round(mean(naTotal), 0),
      meanAgenticTotal: round(mean(agTotal), 0),
      meanAgenticPeakPrompt: round(mean(agPeak), 0),
    });
  }

  return summaries.sort((a, b) => a.label.localeCompare(b.label));
}
