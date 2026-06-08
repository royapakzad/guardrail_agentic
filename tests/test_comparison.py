"""Tests for comparison.compare_judgments — the core research metric."""

from agentic_runner import AgenticJudgment
from comparison import compare_judgments


def _agentic(score=None, valid=None, tool_calls=0, sources=None):
    return AgenticJudgment(
        valid=valid,
        score=score,
        explanation="",
        tool_calls_made=tool_calls,
        sources_used=sources or [],
    )


def test_score_delta_is_agentic_minus_nonagentic_rounded():
    result = compare_judgments(
        nonagentic_valid=True,
        nonagentic_score=0.80,
        agentic_judgment=_agentic(score=0.65, valid=True, tool_calls=2),
    )
    assert result.score_delta == -0.15


def test_score_delta_none_when_either_score_missing():
    assert (
        compare_judgments(
            nonagentic_valid=True,
            nonagentic_score=None,
            agentic_judgment=_agentic(score=0.65, valid=True),
        ).score_delta
        is None
    )

    assert (
        compare_judgments(
            nonagentic_valid=True,
            nonagentic_score=0.65,
            agentic_judgment=_agentic(score=None, valid=None),
        ).score_delta
        is None
    )


def test_judgment_changed_detects_flip():
    flipped = compare_judgments(
        nonagentic_valid=True,
        nonagentic_score=0.80,
        agentic_judgment=_agentic(score=0.40, valid=False, tool_calls=1),
    )
    assert flipped.judgment_changed is True

    same = compare_judgments(
        nonagentic_valid=True,
        nonagentic_score=0.80,
        agentic_judgment=_agentic(score=0.75, valid=True, tool_calls=1),
    )
    assert same.judgment_changed is False


def test_judgment_changed_none_when_either_valid_missing():
    assert (
        compare_judgments(
            nonagentic_valid=None,
            nonagentic_score=0.80,
            agentic_judgment=_agentic(score=0.40, valid=False),
        ).judgment_changed
        is None
    )


def test_agentic_used_tools_reflects_tool_calls_made():
    assert (
        compare_judgments(
            nonagentic_valid=True,
            nonagentic_score=0.8,
            agentic_judgment=_agentic(score=0.8, valid=True, tool_calls=0),
        ).agentic_used_tools
        is False
    )

    assert (
        compare_judgments(
            nonagentic_valid=True,
            nonagentic_score=0.8,
            agentic_judgment=_agentic(score=0.8, valid=True, tool_calls=3),
        ).agentic_used_tools
        is True
    )


def test_sources_used_is_mirrored():
    sources = ["https://ofac.treasury.gov", "https://unhcr.org"]
    result = compare_judgments(
        nonagentic_valid=True,
        nonagentic_score=0.8,
        agentic_judgment=_agentic(score=0.8, valid=True, tool_calls=1, sources=sources),
    )
    assert result.sources_used == sources
