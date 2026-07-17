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

from collections import Counter
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
