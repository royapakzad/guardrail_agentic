import type { EvaluationRecord } from "@/lib/types";
import { normalizeCriterionName, toolTaggedCriteriaNames } from "@/lib/policyCriteria";
import { flattenVariants, groupByLabel, round } from "./flatten";

export type CriterionFlipRow = {
  criterion: string;
  n: number; // scenarios where this criterion was evaluated on both passes
  flippedCount: number;
  flipRate: number | null;
  transitions: { from: string; to: string; count: number }[]; // e.g. MINOR_ISSUE -> COMPLIANT: 3
};

export type CriterionFlipSummary = {
  label: string;
  rows: CriterionFlipRow[];
};

/** Per-criterion agentic-vs-non-agentic verdict comparison, restricted to
 * criteria tagged "(potentially needs tool calls)" in the policy — see
 * lib/policyCriteria.ts. Non-tool criteria are deliberately excluded here:
 * by construction of the split-criteria pipeline, the agentic pass only
 * re-evaluates the tool-tagged subset and carries every other criterion's
 * verdict over unchanged, so a flip rate for them would always be ~0% and
 * would misleadingly read as "measured and found stable" rather than
 * "guaranteed identical by design." Use computeComplianceByCriterion for
 * those criteria's own compliance rates instead. */
export function computeCriterionFlips(records: EvaluationRecord[]): CriterionFlipSummary[] {
  const groups = groupByLabel(flattenVariants(records));
  const summaries: CriterionFlipSummary[] = [];

  for (const [label, items] of groups) {
    const useCase = items[0]?.record.useCase;
    if (!useCase) {
      summaries.push({ label, rows: [] });
      continue;
    }
    const toolCriteria = new Set(toolTaggedCriteriaNames(useCase));

    const perCriterion = new Map<
      string,
      { n: number; flipped: number; transitions: Map<string, number> }
    >();
    for (const name of toolCriteria) {
      perCriterion.set(name, { n: 0, flipped: 0, transitions: new Map() });
    }

    for (const { record, variant } of items) {
      const naByName = new Map(
        variant.nonagentic.criteriaVerdicts.map((c) => [normalizeCriterionName(record.useCase, c.criterion), c.verdict])
      );
      const agByName = new Map(
        variant.agentic.criteriaVerdicts.map((c) => [normalizeCriterionName(record.useCase, c.criterion), c.verdict])
      );

      for (const criterion of toolCriteria) {
        const naVerdict = naByName.get(criterion);
        const agVerdict = agByName.get(criterion);
        if (naVerdict === undefined || agVerdict === undefined) continue; // not comparable for this record

        const entry = perCriterion.get(criterion)!;
        entry.n += 1;
        if (naVerdict !== agVerdict) {
          entry.flipped += 1;
          const key = `${naVerdict} -> ${agVerdict}`;
          entry.transitions.set(key, (entry.transitions.get(key) ?? 0) + 1);
        }
      }
    }

    const rows: CriterionFlipRow[] = [...perCriterion.entries()].map(([criterion, stats]) => ({
      criterion,
      n: stats.n,
      flippedCount: stats.flipped,
      flipRate: stats.n > 0 ? round(stats.flipped / stats.n) : null,
      transitions: [...stats.transitions.entries()]
        .map(([key, count]) => {
          const [from, to] = key.split(" -> ");
          return { from, to, count };
        })
        .sort((a, b) => b.count - a.count),
    }));

    rows.sort((a, b) => a.criterion.localeCompare(b.criterion));
    summaries.push({ label, rows });
  }

  return summaries.sort((a, b) => a.label.localeCompare(b.label));
}
