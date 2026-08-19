export type UseCase = "humanitarian" | "financial" | "cybersecurity";

export type CriterionVerdict = {
  criterion: string;
  verdict: string; // COMPLIANT | NOT_FULLY_COMPLIANT (older runs may still have MINOR_ISSUE/MAJOR_ISSUE/CRITICAL)
  /** The specific claim/URL/phrase/omission a human reviewer should check. Empty when compliant. */
  human_review_needed?: string;
  /** Concrete, actionable fix. Empty when compliant. */
  suggested_improvement?: string;
  /** @deprecated superseded by human_review_needed/suggested_improvement; kept for older runs */
  issues?: string[];
  /** @deprecated superseded by suggested_improvement; kept for older runs */
  improvements?: string[];
  tool_influenced?: boolean;
  /** Tool names the judge SELF-REPORTS it called for this criterion (its own
   * claim, from criteria_verdicts JSON). */
  tools_used?: string[];
  /** Ground truth: tool names actually tagged to this criterion in
   * tool_call_log, computed server-side by
   * agentic_runner._verify_tool_criterion_links -- independent of what the
   * judge claims in tools_used. */
  tools_actually_tagged?: string[];
  /** True only if every tool named in tools_used has a matching tagged call
   * in tool_call_log for this criterion; false on any overclaim (including
   * a non-empty tools_used with nothing actually tagged). Absent on records
   * from before this verification existed. */
  tools_used_verified?: boolean;
  [key: string]: unknown;
};

export type ToolCall = {
  tool: string;
  /** 1-indexed position in this judge's tool-call sequence. */
  call_number?: number;
  /** ISO 8601 UTC timestamp of when the call was made. */
  timestamp?: string;
  /** Short human-readable description of what this tool checks, e.g.
   * "URL malicious/benign reputation check (URLScan.io)". */
  check_purpose?: string;
  input?: Record<string, unknown>;
  output_preview?: string;
  [key: string]: unknown;
};

export type UrlCheck = {
  url?: string;
  valid?: boolean;
  status_code?: number | null;
  final_url?: string;
  [key: string]: unknown;
};

export type ClaimCheck = {
  claim: string;
  status: string; // verified | contradicted | unverifiable
  [key: string]: unknown;
};

export type TokenUsage = {
  promptTokens: number | null;
  completionTokens: number | null;
  totalTokens: number | null;
  peakPromptTokens?: number | null;
};

export type JudgePass = {
  score: number | null;
  /** Recomputed from score, not trusted from the raw field — see adapters/shared.ts */
  valid: boolean | null;
  explanation: string;
  criteriaVerdicts: CriterionVerdict[];
  tokens: TokenUsage;
  judgmentTimeS: number | null;
};

export type AgenticPass = JudgePass & {
  toolCallLog: ToolCall[];
  urlChecks: UrlCheck[];
  claimChecks: ClaimCheck[];
  toolCallsMade: number | null;
  sourcesUsed: string[];
  toolChangedVerdictFor?: string[];
};

export type PolicyVariant = {
  /** Unique within a record: encodes policy + judge model, e.g. "humanitarian_policy_explicit_fa_claude_sonnet_4_6" */
  label: string;
  policyName: string;
  judgeModel: string | null;
  nonagentic: JudgePass;
  agentic: AgenticPass;
  scoreDelta: number | null;
  judgmentChanged: boolean | null;
};

export type EvaluationRecord = {
  id: string;
  useCase: UseCase;
  language: string;
  scenario: string;
  assistantResponse: string;
  assistantModel?: string | null;
  scenarioType?: string | null;
  testFocus?: string | null;
  policyVariants: PolicyVariant[];
};
