import type { EvaluationRecord } from "@/lib/types";
import { flattenVariants, groupByLabel, round } from "./flatten";

export type FlipRateSummary = {
  label: string;
  n: number; // records with at least one comparable criterion (non-empty on both sides)
  flippedCount: number;
  flipRate: number | null;
  topFlippedCriteria: { criterion: string; count: number }[];
};

/** A criterion "flips" when the agentic pass's verdict for it differs from the
 * non-agentic pass's verdict. Criteria the agentic pass never evaluated (not
 * present in its criteriaVerdicts — e.g. historical records where the agentic
 * pass predates the split-criteria merge) are not counted as flips or as
 * comparable — they're simply excluded, not treated as agreement. */
export function computeFlipRate(records: EvaluationRecord[]): FlipRateSummary[] {
  const groups = groupByLabel(flattenVariants(records));
  const summaries: FlipRateSummary[] = [];

  for (const [label, items] of groups) {
    let comparableRecordCount = 0;
    let flippedRecordCount = 0;
    const criterionFlipCounts = new Map<string, number>();

    for (const { variant } of items) {
      const naByName = new Map(variant.nonagentic.criteriaVerdicts.map((c) => [c.criterion, c.verdict]));
      const agByName = new Map(variant.agentic.criteriaVerdicts.map((c) => [c.criterion, c.verdict]));

      if (agByName.size === 0) continue; // no agentic criteria data to compare — exclude, not "no flip"
      comparableRecordCount += 1;

      let recordFlipped = false;
      for (const [criterion, agVerdict] of agByName) {
        const naVerdict = naByName.get(criterion);
        if (naVerdict !== undefined && naVerdict !== agVerdict) {
          recordFlipped = true;
          criterionFlipCounts.set(criterion, (criterionFlipCounts.get(criterion) ?? 0) + 1);
        }
      }
      if (recordFlipped) flippedRecordCount += 1;
    }

    const topFlippedCriteria = [...criterionFlipCounts.entries()]
      .map(([criterion, count]) => ({ criterion, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);

    summaries.push({
      label,
      n: comparableRecordCount,
      flippedCount: flippedRecordCount,
      flipRate: comparableRecordCount > 0 ? round(flippedRecordCount / comparableRecordCount) : null,
      topFlippedCriteria,
    });
  }

  return summaries.sort((a, b) => a.label.localeCompare(b.label));
}
