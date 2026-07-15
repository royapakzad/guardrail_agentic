import type { EvaluationRecord } from "@/lib/types";
import { flattenVariants, groupByLabel, round } from "./flatten";

export type ToolSelectionSummary = {
  label: string;
  acronymOpportunities: number;
  acronymToolUsed: number;
  acronymConsistency: number | null;
  urlOpportunities: number;
  urlToolUsed: number;
  urlConsistency: number | null;
};

// Best-effort heuristics for "did the assistant response contain a claim type
// that has a dedicated verification tool" — there is no gold-labeled ground
// truth for this batch data (unlike the hand-crafted 9-scenario pilot), so
// this is a text-pattern proxy, not a precise measurement. It mirrors the
// extraction pattern already used in agentic_runner.py's acronym detection.
const ACRONYM_PATTERN = /\b[A-Z]{2,6}\b\s*\(/; // e.g. "UNHCR (" — acronym followed by a parenthetical
const URL_PATTERN = /https?:\/\/\S+/;
const URL_TOOLS = new Set(["check_url_validity", "urlscan_check", "fetch_url"]);

export function computeToolSelectionConsistency(records: EvaluationRecord[]): ToolSelectionSummary[] {
  const groups = groupByLabel(flattenVariants(records));
  const summaries: ToolSelectionSummary[] = [];

  for (const [label, items] of groups) {
    let acronymOpportunities = 0;
    let acronymToolUsed = 0;
    let urlOpportunities = 0;
    let urlToolUsed = 0;

    for (const { record, variant } of items) {
      const toolNames = new Set(variant.agentic.toolCallLog.map((t) => t.tool));

      if (ACRONYM_PATTERN.test(record.assistantResponse)) {
        acronymOpportunities += 1;
        if (toolNames.has("check_acronym")) acronymToolUsed += 1;
      }
      if (URL_PATTERN.test(record.assistantResponse)) {
        urlOpportunities += 1;
        if ([...toolNames].some((t) => URL_TOOLS.has(t))) urlToolUsed += 1;
      }
    }

    summaries.push({
      label,
      acronymOpportunities,
      acronymToolUsed,
      acronymConsistency: acronymOpportunities > 0 ? round(acronymToolUsed / acronymOpportunities) : null,
      urlOpportunities,
      urlToolUsed,
      urlConsistency: urlOpportunities > 0 ? round(urlToolUsed / urlOpportunities) : null,
    });
  }

  return summaries.sort((a, b) => a.label.localeCompare(b.label));
}
