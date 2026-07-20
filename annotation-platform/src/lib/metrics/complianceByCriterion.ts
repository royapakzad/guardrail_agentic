import type { EvaluationRecord } from "@/lib/types";
import { normalizeCriterionName, isToolTaggedCriterion } from "@/lib/policyCriteria";
import { flattenVariants, groupByLabel, round } from "./flatten";

// Binary compliance scale (Issue #54 follow-up). Older uploaded datasets from
// before that change may still carry MINOR_ISSUE/MAJOR_ISSUE/CRITICAL -- those
// fall into the `other` bucket below rather than being silently miscounted.
const VERDICTS = ["COMPLIANT", "NOT_FULLY_COMPLIANT"] as const;
type Verdict = (typeof VERDICTS)[number];

export type VerdictCounts = Record<Verdict, number> & { other: number };

export type CriterionComplianceRow = {
  criterion: string;
  toolTagged: boolean;
  nonagentic: VerdictCounts;
  agentic: VerdictCounts;
  nonagenticComplianceRate: number | null; // COMPLIANT / total
  agenticComplianceRate: number | null;
};

export type ComplianceByCriterionSummary = {
  label: string;
  rows: CriterionComplianceRow[];
};

function emptyCounts(): VerdictCounts {
  return { COMPLIANT: 0, NOT_FULLY_COMPLIANT: 0, other: 0 };
}

function tally(counts: VerdictCounts, verdict: string) {
  if ((VERDICTS as readonly string[]).includes(verdict)) {
    counts[verdict as Verdict] += 1;
  } else {
    counts.other += 1;
  }
}

function complianceRate(counts: VerdictCounts): number | null {
  const total = VERDICTS.reduce((sum, v) => sum + counts[v], 0) + counts.other;
  return total > 0 ? round(counts.COMPLIANT / total) : null;
}

/** Compliance/non-compliance distribution per policy criterion — every
 * criterion in the policy, not just an aggregate valid/invalid score. Shows
 * both passes side by side so a criterion's compliance rate can be compared
 * before and after tool verification, in addition to the flip-specific view
 * in criterionFlips.ts. */
export function computeComplianceByCriterion(records: EvaluationRecord[]): ComplianceByCriterionSummary[] {
  const groups = groupByLabel(flattenVariants(records));
  const summaries: ComplianceByCriterionSummary[] = [];

  for (const [label, items] of groups) {
    const useCase = items[0]?.record.useCase;
    const naCounts = new Map<string, VerdictCounts>();
    const agCounts = new Map<string, VerdictCounts>();

    for (const { record, variant } of items) {
      for (const c of variant.nonagentic.criteriaVerdicts) {
        const name = normalizeCriterionName(record.useCase, c.criterion);
        if (!naCounts.has(name)) naCounts.set(name, emptyCounts());
        tally(naCounts.get(name)!, c.verdict);
      }
      for (const c of variant.agentic.criteriaVerdicts) {
        const name = normalizeCriterionName(record.useCase, c.criterion);
        if (!agCounts.has(name)) agCounts.set(name, emptyCounts());
        tally(agCounts.get(name)!, c.verdict);
      }
    }

    const criterionNames = new Set([...naCounts.keys(), ...agCounts.keys()]);
    const rows: CriterionComplianceRow[] = [...criterionNames].map((criterion) => {
      const nonagentic = naCounts.get(criterion) ?? emptyCounts();
      const agentic = agCounts.get(criterion) ?? emptyCounts();
      return {
        criterion,
        toolTagged: useCase ? isToolTaggedCriterion(useCase, criterion) : false,
        nonagentic,
        agentic,
        nonagenticComplianceRate: complianceRate(nonagentic),
        agenticComplianceRate: complianceRate(agentic),
      };
    });

    rows.sort((a, b) => a.criterion.localeCompare(b.criterion));
    summaries.push({ label, rows });
  }

  return summaries.sort((a, b) => a.label.localeCompare(b.label));
}
