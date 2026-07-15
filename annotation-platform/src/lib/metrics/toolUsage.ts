import type { EvaluationRecord } from "@/lib/types";
import { flattenVariants, groupByLabel, mean, median, round } from "./flatten";

export type ToolUsageSummary = {
  label: string;
  n: number;
  meanCallsPerScenario: number | null;
  medianCallsPerScenario: number | null;
  toolFrequency: { tool: string; count: number }[];
};

export function computeToolUsage(records: EvaluationRecord[]): ToolUsageSummary[] {
  const groups = groupByLabel(flattenVariants(records));
  const summaries: ToolUsageSummary[] = [];

  for (const [label, items] of groups) {
    const callCounts = items.map((i) => i.variant.agentic.toolCallLog.length);
    const toolFreq = new Map<string, number>();
    for (const { variant } of items) {
      for (const call of variant.agentic.toolCallLog) {
        toolFreq.set(call.tool, (toolFreq.get(call.tool) ?? 0) + 1);
      }
    }

    summaries.push({
      label,
      n: items.length,
      meanCallsPerScenario: round(mean(callCounts)),
      medianCallsPerScenario: median(callCounts),
      toolFrequency: [...toolFreq.entries()]
        .map(([tool, count]) => ({ tool, count }))
        .sort((a, b) => b.count - a.count),
    });
  }

  return summaries.sort((a, b) => a.label.localeCompare(b.label));
}
