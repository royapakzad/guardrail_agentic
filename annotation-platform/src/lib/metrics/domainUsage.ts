import type { EvaluationRecord, PolicyVariant, ToolCall } from "@/lib/types";
import { flattenVariants, groupByLabel } from "./flatten";

export type DomainCount = { domain: string; count: number };

export type DomainUsageSummary = {
  label: string;
  n: number;
  totalUrlCount: number;
  distinctUrlCount: number;
  distinctDomainCount: number;
  /** Domain frequency from search_web results only. */
  searchDomains: DomainCount[];
  /** Domain frequency across every tool call (search results, fetch/check
   * inputs, and any URL embedded in a domain-specific tool's own output). */
  allDomains: DomainCount[];
};

const URL_PATTERN = /https?:\/\/[^\s"')>\]]+/g;

function getHostname(url: string): string | null {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return null;
  }
}

/**
 * Recursively collect every http(s) URL found in a tool call's input and
 * parsed output_preview. Tool-agnostic by design — a new tool that returns or
 * accepts URLs needs no changes here, since this walks the raw JSON shape
 * rather than reading known field paths per tool.
 */
function extractUrls(call: ToolCall): string[] {
  const urls: string[] = [];
  const visit = (value: unknown): void => {
    if (typeof value === "string") {
      const matches = value.match(URL_PATTERN);
      if (matches) urls.push(...matches);
      return;
    }
    if (Array.isArray(value)) {
      value.forEach(visit);
      return;
    }
    if (value && typeof value === "object") {
      Object.values(value).forEach(visit);
    }
  };

  visit(call.input);

  if (typeof call.output_preview === "string") {
    try {
      visit(JSON.parse(call.output_preview));
    } catch {
      // Not JSON (e.g. a raw error string) -- still scan the text itself.
      visit(call.output_preview);
    }
  } else {
    visit(call.output_preview);
  }

  return urls;
}

function tally(urls: string[]): { counts: DomainCount[]; distinctUrls: Set<string> } {
  const domainCounts = new Map<string, number>();
  const distinctUrls = new Set<string>();
  for (const url of urls) {
    distinctUrls.add(url);
    const domain = getHostname(url);
    if (!domain) continue;
    domainCounts.set(domain, (domainCounts.get(domain) ?? 0) + 1);
  }
  const counts = [...domainCounts.entries()]
    .map(([domain, count]) => ({ domain, count }))
    .sort((a, b) => b.count - a.count);
  return { counts, distinctUrls };
}

export function computeDomainUsage(records: EvaluationRecord[]): DomainUsageSummary[] {
  const groups = groupByLabel(flattenVariants(records));
  const summaries: DomainUsageSummary[] = [];

  for (const [label, items] of groups) {
    const allUrls: string[] = [];
    const searchUrls: string[] = [];

    for (const { variant } of items) {
      for (const call of variant.agentic.toolCallLog) {
        const urls = extractUrls(call);
        allUrls.push(...urls);
        if (call.tool === "search_web") searchUrls.push(...urls);
      }
    }

    const all = tally(allUrls);
    const search = tally(searchUrls);

    summaries.push({
      label,
      n: items.length,
      totalUrlCount: allUrls.length,
      distinctUrlCount: all.distinctUrls.size,
      distinctDomainCount: all.counts.length,
      searchDomains: search.counts,
      allDomains: all.counts,
    });
  }

  return summaries.sort((a, b) => a.label.localeCompare(b.label));
}

export type ToolCount = { tool: string; count: number };

export type ScenarioToolSummary = {
  /** How many times each tool was called for this one scenario/policy variant. */
  toolCounts: ToolCount[];
  totalToolCalls: number;
  /** Every domain touched by this variant's agentic pass (search results, fetch/check
   * inputs, and any URL embedded in a domain-specific tool's own output). */
  domains: DomainCount[];
  totalUrlCount: number;
  distinctUrlCount: number;
};

/** Per-scenario counterpart to computeDomainUsage/computeToolUsage — both of
 * those aggregate across an entire dataset grouped by policy label; this is
 * for the scenario detail page, which needs the same tool-call and
 * domain-touch counts for just one record's one policy variant. */
export function computeScenarioToolSummary(variant: PolicyVariant): ScenarioToolSummary {
  const toolCounts = new Map<string, number>();
  const allUrls: string[] = [];

  for (const call of variant.agentic.toolCallLog) {
    toolCounts.set(call.tool, (toolCounts.get(call.tool) ?? 0) + 1);
    allUrls.push(...extractUrls(call));
  }

  const { counts, distinctUrls } = tally(allUrls);

  return {
    toolCounts: [...toolCounts.entries()]
      .map(([tool, count]) => ({ tool, count }))
      .sort((a, b) => b.count - a.count),
    totalToolCalls: variant.agentic.toolCallLog.length,
    domains: counts,
    totalUrlCount: allUrls.length,
    distinctUrlCount: distinctUrls.size,
  };
}
