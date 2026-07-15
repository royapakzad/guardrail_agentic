import type { EvaluationRecord } from "@/lib/types";
import { flattenVariants, groupByLabel, mean, median, round } from "./flatten";

export type LatencySummary = {
  label: string;
  n: number;
  meanNonagenticS: number | null;
  medianNonagenticS: number | null;
  meanAgenticS: number | null;
  medianAgenticS: number | null;
};

export function computeLatency(records: EvaluationRecord[]): LatencySummary[] {
  const groups = groupByLabel(flattenVariants(records));
  const summaries: LatencySummary[] = [];

  for (const [label, items] of groups) {
    const na = items.map((i) => i.variant.nonagentic.judgmentTimeS).filter((v): v is number => v !== null);
    const ag = items.map((i) => i.variant.agentic.judgmentTimeS).filter((v): v is number => v !== null);

    summaries.push({
      label,
      n: items.length,
      meanNonagenticS: round(mean(na)),
      medianNonagenticS: median(na),
      meanAgenticS: round(mean(ag)),
      medianAgenticS: median(ag),
    });
  }

  return summaries.sort((a, b) => a.label.localeCompare(b.label));
}
