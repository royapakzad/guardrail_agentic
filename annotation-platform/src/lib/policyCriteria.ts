import type { UseCase } from "@/lib/types";

/**
 * The canonical criterion list per use case, straight from the policy text
 * files (config/*.txt in the parent repo) — including which criteria are
 * tagged "(potentially needs tool calls)". This is the source of truth for
 * "does this criterion require agentic tool use", not the judge's own
 * self-reported `tool_influenced` field (scenario-dependent, not always
 * populated) and not "is this criterion present in the agentic pass's
 * criteriaVerdicts" (many historical records have an empty agentic array
 * even for tool-tagged criteria — see adapters/humanitarian.ts notes).
 *
 * Re-verify against config/*.txt if a policy is ever edited — this list is
 * intentionally not derived at runtime so it stays stable across uploaded
 * datasets that may carry judge-output quirks (see normalizeCriterionName).
 */
export const POLICY_CRITERIA: Record<UseCase, { name: string; toolTagged: boolean }[]> = {
  humanitarian: [
    { name: "ACTIONABILITY AND PRACTICALITY", toolTagged: true },
    { name: "FACTUALITY AND ACCURACY", toolTagged: true },
    { name: "SAFETY, SECURITY, AND PRIVACY", toolTagged: false },
    { name: "TONE, DIGNITY, AND EMPATHY", toolTagged: false },
    { name: "NON-DISCRIMINATION AND FAIRNESS", toolTagged: false },
    { name: "FREEDOM OF ACCESS TO INFORMATION, CENSORSHIP, AND REFUSAL", toolTagged: false },
  ],
  financial: [
    { name: "EXPLAINABILITY AND ALGORITHMIC TRANSPARENCY", toolTagged: false },
    { name: "REGULATORY CONTEXT AND DISCLAIMERS", toolTagged: true },
    { name: "DATA RIGHTS AND SURVEILLANCE ARCHITECTURE", toolTagged: false },
    { name: "FIDUCIARY BOUNDARIES", toolTagged: false },
    { name: "POWER ASYMMETRIES AND STRUCTURAL CONSTRAINTS", toolTagged: false },
    { name: "ALTERNATIVE FINANCIAL SYSTEMS", toolTagged: false },
    { name: "ACCURACY AND CURRENCY", toolTagged: true },
    { name: "HARM VECTORS SPECIFIC TO FINANCIAL CONTEXTS", toolTagged: false },
  ],
  cybersecurity: [
    { name: "THREAT INDICATOR RECOGNITION", toolTagged: true },
    { name: "AUTHENTICITY, DOMAIN, AND SOURCE VERIFICATION", toolTagged: true },
    { name: "CREDENTIALS, SECRETS, AND SENSITIVE DATA PROTECTION", toolTagged: false },
    { name: "SAFE RESPONSE AND INCIDENT HANDLING", toolTagged: false },
    { name: "EVIDENCE CALIBRATION AND UNCERTAINTY", toolTagged: true },
    { name: "ABUSE-ENABLEMENT BOUNDARIES", toolTagged: false },
    { name: "LEGITIMATE SECURITY, EDUCATIONAL, AND DEFENSIVE USES", toolTagged: false },
    { name: "PRIVACY, DIGNITY, AND NON-BLAMING USER SUPPORT", toolTagged: false },
    { name: "URLSCAN.IO AND URL REPUTATION EVIDENCE HANDLING", toolTagged: true },
  ],
};

/**
 * Maps a raw criterion name from judge output to its canonical form. Handles
 * the judge sometimes emitting a truncated name (observed in real data: e.g.
 * "FREEDOM OF ACCESS TO INFORMATION" for the full
 * "FREEDOM OF ACCESS TO INFORMATION, CENSORSHIP, AND REFUSAL") by matching on
 * prefix. Falls back to the trimmed raw name (visible as its own row) rather
 * than silently dropping anything unrecognized.
 */
export function normalizeCriterionName(useCase: UseCase, raw: string): string {
  const trimmed = raw.trim();
  const canonical = POLICY_CRITERIA[useCase];

  const exact = canonical.find((c) => c.name === trimmed);
  if (exact) return exact.name;

  const upper = trimmed.toUpperCase();
  const prefixMatch = canonical.find(
    (c) => c.name.toUpperCase().startsWith(upper) || upper.startsWith(c.name.toUpperCase())
  );
  if (prefixMatch) return prefixMatch.name;

  return trimmed;
}

export function isToolTaggedCriterion(useCase: UseCase, canonicalName: string): boolean {
  return POLICY_CRITERIA[useCase].some((c) => c.name === canonicalName && c.toolTagged);
}

export function toolTaggedCriteriaNames(useCase: UseCase): string[] {
  return POLICY_CRITERIA[useCase].filter((c) => c.toolTagged).map((c) => c.name);
}
