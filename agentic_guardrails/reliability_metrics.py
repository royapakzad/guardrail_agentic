"""
reliability_metrics.py
-----------------------
Domain-agnostic reliability metrics for repeated-judge diagnostic runs
(Issue #50).

These functions operate purely on criterion-name -> verdict-string
dictionaries. They have no knowledge of which policy (humanitarian,
financial, cybersecurity, or any future domain) produced the verdicts, so
the same two functions work unmodified across every use case in this repo --
whatever criteria and verdict labels a policy happens to use.

Input shape: a list of N "runs", where each run is a dict mapping
criterion name -> verdict string (e.g. "COMPLIANT", "MINOR_ISSUE",
"MAJOR_ISSUE", "CRITICAL"). N runs = N repeated judge calls on
byte-identical input (same frozen scenario + assistant response, same
judge model); each criterion is one "item" being rated by each run.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class CriterionFlipStats:
    criterion: str
    verdicts: list           # one verdict string per run, in run order
    modal_verdict: str
    flip_rate: float         # fraction of runs disagreeing with the mode


def compute_flip_rates(runs: list) -> list:
    """
    Per-criterion flip rate across N repeated runs on identical input.

    flip_rate = (runs disagreeing with the modal verdict) / total runs.
    0.0 = perfectly stable across all N runs; higher = less reliable.
    A criterion that splits evenly across two verdicts (e.g. 3-of-6 vs
    3-of-6) scores 0.5 -- the practical ceiling for a two-way disagreement.

    Returns one CriterionFlipStats per criterion observed in any run,
    sorted by criterion name for stable output ordering.
    """
    if not runs:
        return []
    criteria = sorted({c for run in runs for c in run})
    stats = []
    for criterion in criteria:
        verdicts = [run.get(criterion, "MISSING") for run in runs]
        counts = Counter(verdicts)
        modal_verdict, modal_count = counts.most_common(1)[0]
        flip_rate = 1 - (modal_count / len(verdicts))
        stats.append(CriterionFlipStats(criterion, verdicts, modal_verdict, round(flip_rate, 4)))
    return stats


def fleiss_kappa(runs: list) -> "float | None":
    """
    Fleiss' kappa treating each run as a "rater" and each criterion as an
    "item", corrected for chance agreement implied by the observed verdict
    base rates (e.g. if COMPLIANT is the verdict 90% of the time, raw
    percent-agreement looks deceptively high purely from that skew --
    kappa subtracts out exactly that chance component).

    Standard interpretation (Landis & Koch 1977):
      <0.00 poor, 0.01-0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate,
      0.61-0.80 substantial, 0.81-1.00 almost perfect.

    Returns None if there are fewer than 2 runs, fewer than 2 distinct
    verdict categories were observed, or chance-expected agreement is 1.0
    (no variance for kappa to explain) -- kappa is undefined in all three
    cases rather than being a misleading 0.0 or 1.0.
    """
    if len(runs) < 2:
        return None
    criteria = sorted({c for run in runs for c in run})
    if not criteria:
        return None
    n = len(runs)         # raters per item
    N = len(criteria)      # items
    categories = sorted({run.get(c, "MISSING") for run in runs for c in criteria})
    if len(categories) < 2:
        return None

    # table[i][j] = number of raters who assigned item i to category j
    table = []
    for criterion in criteria:
        counts = Counter(run.get(criterion, "MISSING") for run in runs)
        table.append([counts.get(cat, 0) for cat in categories])

    p_j = [sum(row[j] for row in table) / (N * n) for j in range(len(categories))]
    P_i = [(sum(v * v for v in row) - n) / (n * (n - 1)) for row in table]
    P_bar = sum(P_i) / N
    P_e_bar = sum(p * p for p in p_j)

    if P_e_bar >= 1.0:
        return None

    return round((P_bar - P_e_bar) / (1 - P_e_bar), 4)


@dataclass
class ReliabilityReport:
    label: str                                   # e.g. "non-agentic" or "agentic"
    n_runs: int
    criterion_stats: list = field(default_factory=list)
    kappa: "float | None" = None

    def unstable_criteria(self) -> list:
        """Criteria with any disagreement across runs (flip_rate > 0)."""
        return [c for c in self.criterion_stats if c.flip_rate > 0]

    def format_report(self) -> str:
        lines = [f"=== {self.label} (n={self.n_runs} runs) ==="]
        kappa_str = f"{self.kappa:.4f}" if self.kappa is not None else "undefined (insufficient variance/data)"
        lines.append(f"Fleiss' kappa: {kappa_str}")
        lines.append("Per-criterion flip rate:")
        for c in self.criterion_stats:
            flag = "  <<< UNSTABLE" if c.flip_rate > 0 else ""
            lines.append(f"  {c.criterion[:55]:55s} flip_rate={c.flip_rate:.3f} verdicts={c.verdicts}{flag}")
        return "\n".join(lines)


def build_reliability_report(runs: list, label: str) -> ReliabilityReport:
    """Convenience wrapper: compute both metrics and package as one report."""
    return ReliabilityReport(
        label=label,
        n_runs=len(runs),
        criterion_stats=compute_flip_rates(runs),
        kappa=fleiss_kappa(runs),
    )


# ══ Category collapse (Issue #54) ═══════════════════════════════════════════
# Recompute flip-rate/kappa under a coarser verdict scale from data we've
# already collected, so we can empirically check which collapse (if any)
# actually reduces measured instability -- rather than assuming and changing
# the live judge's output schema before knowing whether it would help.
#
# Schemes are named for the question they answer, and reuse the exact
# vocabulary ("COMPLIANT", "violation") already used in guardrails_runner.py's
# SHARED_SEVERITY_ANCHORS rubric text, rather than inventing new labels:
#
#   severity_tier: keeps COMPLIANT vs MINOR_ISSUE separate (useful for the
#     "improvements_required" feedback loop) and merges only MAJOR_ISSUE +
#     CRITICAL -> VIOLATION. Defensible because CRITICAL is explicitly meant
#     to be rare ("reserve this tier for categorically prohibited content"),
#     and MAJOR_ISSUE/CRITICAL was never the boundary that flipped in any
#     diagnostic run collected so far.
#
#   harm_based: directly re-derives the rubric's own stated decision test
#     ("would the user be meaningfully worse off because of this gap?") by
#     merging COMPLIANT + MINOR_ISSUE -> COMPLIANT (that test's "no" answer)
#     and MAJOR_ISSUE + CRITICAL -> VIOLATION (its "yes" answer).
#
# Empirical grounding (see Issue #54): every flip observed so far in
# repeated-run diagnostics, across all three domains, was COMPLIANT <->
# MINOR_ISSUE. None touched MAJOR_ISSUE/CRITICAL. That means severity_tier
# is expected to leave kappa/flip-rate unchanged on data seen so far, while
# harm_based would eliminate those flips -- but only by merging away the
# exact distinction that was flip-flopping, not by making the underlying
# reasoning more consistent. Both are provided so that tradeoff is visible
# in the numbers rather than assumed.
COLLAPSE_SCHEMES: dict = {
    "full": {},  # identity -- no collapse; unmapped verdicts pass through unchanged
    "severity_tier": {
        "MINOR_ISSUE": "MINOR",
        "MAJOR_ISSUE": "VIOLATION",
        "CRITICAL": "VIOLATION",
    },
    "harm_based": {
        "COMPLIANT": "COMPLIANT",
        "MINOR_ISSUE": "COMPLIANT",
        "MAJOR_ISSUE": "VIOLATION",
        "CRITICAL": "VIOLATION",
    },
}


def collapse_runs(runs: list, scheme: str) -> list:
    """
    Remap verdict labels in every run according to a named entry in
    COLLAPSE_SCHEMES. Verdicts not mentioned in the scheme pass through
    unchanged (so this works regardless of which verdict vocabulary a
    given policy happens to use -- domain-agnostic, same as the rest of
    this module).
    """
    if scheme not in COLLAPSE_SCHEMES:
        raise ValueError(f"Unknown collapse scheme {scheme!r}. Known: {sorted(COLLAPSE_SCHEMES)}")
    mapping = COLLAPSE_SCHEMES[scheme]
    return [{c: mapping.get(v, v) for c, v in run.items()} for run in runs]


def compare_collapse_schemes(runs: list, label: str) -> dict:
    """
    Recompute kappa (and count of unstable criteria) under every known
    collapse scheme, so the effect of coarsening the verdict scale is a
    measured number, not an assumption.
    """
    report = {}
    for scheme in COLLAPSE_SCHEMES:
        collapsed = collapse_runs(runs, scheme)
        r = build_reliability_report(collapsed, label=f"{label} [{scheme}]")
        report[scheme] = {
            "kappa": r.kappa,
            "n_unstable_criteria": len(r.unstable_criteria()),
            "n_criteria": len(r.criterion_stats),
        }
    return report


# ══ Evidence reproducibility (Issue #54) ════════════════════════════════════
# Separates "is the evidence itself reproducible" from "is the verdict built
# on top of that evidence reproducible" -- the two things a single kappa
# number conflates. Tool calls are deterministic API/database lookups
# (unlike the judge's categorical verdict), so this is expected to show
# near-perfect reproducibility, isolating the noise to the verdict-mapping
# layer rather than the evidence-gathering layer.


def compare_tool_call_outputs(runs_tool_logs: list) -> list:
    """
    runs_tool_logs: one list of tool-call-log entries per run, each entry a
    dict with at least {"tool": str, "input": dict, "output_preview": str}
    (the same shape agentic_runner.py's tool_call_log already uses).

    For every (tool, input) pair that was called in 2+ runs, check whether
    the returned output was byte-identical every time. Returns one entry
    per such repeated (tool, input) pair, sorted by tool name then input,
    for stable output ordering.
    """
    calls_by_key: dict = defaultdict(list)
    for run_idx, log in enumerate(runs_tool_logs):
        for entry in log:
            key = (entry.get("tool", ""), json.dumps(entry.get("input", {}), sort_keys=True))
            calls_by_key[key].append((run_idx, entry.get("output_preview", "")))

    report = []
    for (tool, input_json), calls in sorted(calls_by_key.items()):
        if len(calls) < 2:
            continue
        outputs = [o for _, o in calls]
        distinct = sorted(set(outputs))
        report.append(
            {
                "tool": tool,
                "input": json.loads(input_json),
                "n_calls": len(calls),
                "identical_output": len(distinct) == 1,
                "distinct_outputs": len(distinct),
            }
        )
    return report


def format_evidence_reproducibility_report(entries: list) -> str:
    if not entries:
        return "No (tool, input) pair was repeated across 2+ runs -- nothing to compare."
    lines = ["Evidence reproducibility (same tool + same input, across repeated runs):"]
    for e in entries:
        flag = "" if e["identical_output"] else "  <<< EVIDENCE ITSELF CHANGED"
        lines.append(
            f"  {e['tool']}({e['input']}) -- {e['n_calls']} calls, "
            f"{e['distinct_outputs']} distinct output(s){flag}"
        )
    return "\n".join(lines)
