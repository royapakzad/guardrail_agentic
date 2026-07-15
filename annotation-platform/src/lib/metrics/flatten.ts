import type { EvaluationRecord, PolicyVariant } from "@/lib/types";

export type FlatVariant = {
  record: EvaluationRecord;
  variant: PolicyVariant;
};

export function flattenVariants(records: EvaluationRecord[]): FlatVariant[] {
  return records.flatMap((record) => record.policyVariants.map((variant) => ({ record, variant })));
}

export function groupByLabel(flat: FlatVariant[]): Map<string, FlatVariant[]> {
  const groups = new Map<string, FlatVariant[]>();
  for (const item of flat) {
    const list = groups.get(item.variant.label);
    if (list) list.push(item);
    else groups.set(item.variant.label, [item]);
  }
  return groups;
}

export function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

export function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

export function round(value: number | null, digits = 3): number | null {
  if (value === null) return null;
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}
