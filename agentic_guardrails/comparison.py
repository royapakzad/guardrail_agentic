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
    Derive comparison fields from the two guardrail paths.
    """
    # Score delta
    if nonagentic_score is not None and agentic_judgment.score is not None:
        score_delta: Optional[float] = round(
            agentic_judgment.score - nonagentic_score, 4
        )
    else:
        score_delta = None

    # Did the binary judgment flip?
    if nonagentic_valid is not None and agentic_judgment.valid is not None:
        judgment_changed: Optional[bool] = agentic_judgment.valid != nonagentic_valid
    else:
        judgment_changed = None

    return ComparisonResult(
        score_delta=score_delta,
        judgment_changed=judgment_changed,
        agentic_used_tools=agentic_judgment.tool_calls_made > 0,
        sources_used=agentic_judgment.sources_used,
    )
