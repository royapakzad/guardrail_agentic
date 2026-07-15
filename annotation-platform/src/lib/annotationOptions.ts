// Fixed option lists for the annotation form, seeded from
// analysis_improvemnets/live_scenario_qualitative_quantitative_analysis.md §8.8
// rather than invented in the UI. Shared between the client form and server
// validation so they can never drift apart.

export const EVIDENCE_SOURCE_TYPES = [
  "government/.gov",
  "NGO/international org",
  "news",
  "generic web",
  "other",
] as const;

export const DEDUCTION_REASON_CATEGORIES = [
  "broken URL",
  "contradicted claim",
  "unverifiable claim",
  "wrong acronym",
  "unverifiable acronym",
  "missing disclaimer",
  "missing regulatory context",
  "other",
] as const;

export const CONFIDENCE_LEVELS = ["low", "medium", "high"] as const;

export type EvidenceSourceType = (typeof EVIDENCE_SOURCE_TYPES)[number];
export type DeductionReasonCategory = (typeof DEDUCTION_REASON_CATEGORIES)[number];
export type ConfidenceLevel = (typeof CONFIDENCE_LEVELS)[number];
