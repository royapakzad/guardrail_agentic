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

// Part 4 "Your review" -- which guardrail's judgment the annotator finds more
// reliable and trustworthy.
export const JUDGMENT_ALIGNMENT_OPTIONS = ["agentic", "non-agentic", "no_difference"] as const;

export type JudgmentAlignment = (typeof JUDGMENT_ALIGNMENT_OPTIONS)[number];

export const JUDGMENT_ALIGNMENT_LABELS: Record<JudgmentAlignment, string> = {
  agentic: "Agentic guardrail",
  "non-agentic": "Non-agentic guardrail",
  no_difference: "Not a meaningful difference between them",
};

// Qualitative coding target fields (Issue #57) -- v1 codes at the field level
// rather than an in-text highlighted span (see the issue's fast-follow note).
// For the two per-criterion options, the annotator names the specific
// criterion in the quote/note text rather than a dedicated schema column --
// keeps the schema simple for v1; a fast-follow could add a target_criterion
// column if that turns out to matter for analysis.
export const CODING_TARGET_FIELDS = [
  "nonagentic_explanation",
  "agentic_explanation",
  "other",
] as const;

export const CODING_TARGET_FIELD_LABELS: Record<(typeof CODING_TARGET_FIELDS)[number], string> = {
  nonagentic_explanation: "Non-agentic judge explanation",
  agentic_explanation: "Agentic judge explanation",
  other: "Other (specify in note)",
};

export type CodingTargetField = (typeof CODING_TARGET_FIELDS)[number];
