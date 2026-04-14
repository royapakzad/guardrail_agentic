"""
comparison.py
-------------
Computes the delta between the non-agentic and agentic guardrail judgments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agentic_runner import AgenticJudgment


@dataclass
class ComparisonResult:
    # score_delta = agentic_score - nonagentic_score
    # Positive → agentic scored the response higher (more compliant).
    # Negative → agentic scored the response lower (less compliant).
    # None if either score is missing.
    score_delta: Optional[float]

    # True if the valid flag flipped between the two paths.
    # None if either valid value is missing.
    judgment_changed: Optional[bool]

    # True if the agentic path actually invoked at least one tool.
    agentic_used_tools: bool

    # Mirror of AgenticJudgment.sources_used for easy access.
    sources_used: list[str]


def compare_judgments(
    *,
    nonagentic_valid: Optional[bool],
    nonagentic_score: Optional[float],
    agentic_judgment: AgenticJudgment,
) -> ComparisonResult:
    """
    Derive the three comparison columns from the two guardrail paths.

    Called once per (row, policy) pair after both paths have completed.
    Returns None for score_delta and judgment_changed if either path errored
    and produced no score or valid flag.
    """
    # score_delta: primary research metric.
    # Positive  → agentic scored the response higher after retrieval
    #             (e.g. found evidence confirming the assistant's claims)
    # Negative  → agentic scored it lower
    #             (e.g. found a broken URL or contradicted a factual claim)
    # None      → at least one path failed to produce a numeric score
    if nonagentic_score is not None and agentic_judgment.score is not None:
        score_delta: Optional[float] = round(
            agentic_judgment.score - nonagentic_score, 4
        )
    else:
        score_delta = None

    # judgment_changed: did the binary pass/fail decision flip between paths?
    # True = the strongest possible signal that retrieval changed the outcome.
    if nonagentic_valid is not None and agentic_judgment.valid is not None:
        judgment_changed: Optional[bool] = agentic_judgment.valid != nonagentic_valid
    else:
        judgment_changed = None

    return ComparisonResult(
        score_delta=score_delta,
        judgment_changed=judgment_changed,
        # True if the agentic judge called at least one tool; False means
        # score_delta reflects prompt differences, not actual retrieval.
        agentic_used_tools=agentic_judgment.tool_calls_made > 0,
        sources_used=agentic_judgment.sources_used,
    )
