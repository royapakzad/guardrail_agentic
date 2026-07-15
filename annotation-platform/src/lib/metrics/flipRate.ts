import type { EvaluationRecord } from "@/lib/types";
import { normalizeCriterionName } from "@/lib/policyCriteria";
import { flattenVariants, groupByLabel, round } from "./flatten";

export type FlipRateSummary = {
  label: string;
  n: number; // records with at least one comparable criterion (non-empty on both sides)
  flippedCount: number;
  flipRate: number | null;
};

/** A criterion "flips" when the agentic pass's verdict for it differs from the
 * non-agentic pass's verdict. Criteria the agentic pass never evaluated (not
 * present in its criteriaVerdicts — e.g. historical records where the agentic
 * pass predates the split-criteria merge) are not counted as flips or as
 * comparable — they're simply excluded, not treated as agreement.
 *
 * This is the aggregate, record-level summary ("did *any* criterion flip for
 * this scenario") — see lib/metrics/criterionFlips.ts for the per-criterion
 * breakdown restricted to tool-tagged criteria, which is the more
 * informative view: non-tool criteria can never flip by construction (the
 * agentic pass only re-evaluates the tool-tagged subset), so an aggregate
 * number here mixes "structurally guaranteed same" with "independently
 * re-evaluated and happened to agree." */
export function computeFlipRate(records: EvaluationRecord[]): FlipRateSummary[] {
  const groups = groupByLabel(flattenVariants(records));
  const summaries: FlipRateSummary[] = [];

  for (const [label, items] of groups) {
    let comparableRecordCount = 0;
    let flippedRecordCount = 0;

    for (const { record, variant } of items) {
      const naByName = new Map(
        variant.nonagentic.criteriaVerdicts.map((c) => [normalizeCriterionName(record.useCase, c.criterion), c.verdict])
      );
      const agByName = new Map(
        variant.agentic.criteriaVerdicts.map((c) => [normalizeCriterionName(record.useCase, c.criterion), c.verdict])
      );

      if (agByName.size === 0) continue; // no agentic criteria data to compare — exclude, not "no flip"
      comparableRecordCount += 1;

      let recordFlipped = false;
      for (const [criterion, agVerdict] of agByName) {
        const naVerdict = naByName.get(criterion);
        if (naVerdict !== undefined && naVerdict !== agVerdict) {
          recordFlipped = true;
        }
      }
      if (recordFlipped) flippedRecordCount += 1;
    }

    summaries.push({
      label,
      n: comparableRecordCount,
      flippedCount: flippedRecordCount,
      flipRate: comparableRecordCount > 0 ? round(flippedRecordCount / comparableRecordCount) : null,
    });
  }

  return summaries.sort((a, b) => a.label.localeCompare(b.label));
}
