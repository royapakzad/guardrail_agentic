import type { EvaluationRecord } from "@/lib/types";
import { flattenVariants, groupByLabel, mean, median, round } from "./flatten";

export type ScoreDeltaSummary = {
  label: string;
  n: number;
  meanDelta: number | null;
  medianDelta: number | null;
  meanAbsDelta: number | null;
  harsherCount: number; // delta < 0: tool evidence made the verdict stricter
  lenientCount: number; // delta > 0: tool evidence made the verdict more lenient
  unchangedCount: number; // delta === 0
};

export function computeScoreDeltas(records: EvaluationRecord[]): ScoreDeltaSummary[] {
  const groups = groupByLabel(flattenVariants(records));
  const summaries: ScoreDeltaSummary[] = [];

  for (const [label, items] of groups) {
    const deltas = items.map((i) => i.variant.scoreDelta).filter((d): d is number => d !== null);
    summaries.push({
      label,
      n: deltas.length,
      meanDelta: round(mean(deltas)),
      medianDelta: round(median(deltas)),
      meanAbsDelta: round(mean(deltas.map((d) => Math.abs(d)))),
      harsherCount: deltas.filter((d) => d < 0).length,
      lenientCount: deltas.filter((d) => d > 0).length,
      unchangedCount: deltas.filter((d) => d === 0).length,
    });
  }

  return summaries.sort((a, b) => a.label.localeCompare(b.label));
}
