"""Tests for compare_judgments (network-free)."""

import pytest
from agentic_runner import AgenticJudgment
from comparison import compare_judgments


def _make_judgment(**kwargs) -> AgenticJudgment:
    defaults = {
        "valid": True,
        "score": 0.80,
        "explanation": "ok",
        "tool_calls_made": 2,
    }
    defaults.update(kwargs)
    return AgenticJudgment(**defaults)


def test_score_delta_positive():
    aj = _make_judgment(score=0.90)
    result = compare_judgments(nonagentic_valid=True, nonagentic_score=0.70, agentic_judgment=aj)
    assert result.score_delta == pytest.approx(0.20, abs=0.001)


def test_score_delta_negative():
    aj = _make_judgment(score=0.50)
    result = compare_judgments(nonagentic_valid=True, nonagentic_score=0.80, agentic_judgment=aj)
    assert result.score_delta is not None
    assert result.score_delta < 0


def test_score_delta_zero():
    aj = _make_judgment(score=0.70)
    result = compare_judgments(nonagentic_valid=True, nonagentic_score=0.70, agentic_judgment=aj)
    assert result.score_delta == pytest.approx(0.0, abs=0.001)


def test_score_delta_none_when_agentic_score_missing():
    aj = _make_judgment(score=None, valid=None)
    result = compare_judgments(nonagentic_valid=True, nonagentic_score=0.80, agentic_judgment=aj)
    assert result.score_delta is None


def test_score_delta_none_when_nonagentic_score_missing():
    aj = _make_judgment(score=0.80)
    result = compare_judgments(nonagentic_valid=True, nonagentic_score=None, agentic_judgment=aj)
    assert result.score_delta is None


def test_judgment_changed_true_when_flip():
    aj = _make_judgment(valid=False, score=0.50)
    result = compare_judgments(nonagentic_valid=True, nonagentic_score=0.80, agentic_judgment=aj)
    assert result.judgment_changed is True


def test_judgment_changed_false_when_no_flip():
    aj = _make_judgment(valid=True, score=0.90)
    result = compare_judgments(nonagentic_valid=True, nonagentic_score=0.80, agentic_judgment=aj)
    assert result.judgment_changed is False


def test_judgment_changed_none_when_nonagentic_valid_missing():
    aj = _make_judgment(valid=True)
    result = compare_judgments(nonagentic_valid=None, nonagentic_score=0.80, agentic_judgment=aj)
    assert result.judgment_changed is None


def test_agentic_used_tools_true():
    aj = _make_judgment(tool_calls_made=3)
    result = compare_judgments(nonagentic_valid=True, nonagentic_score=0.80, agentic_judgment=aj)
    assert result.agentic_used_tools is True


def test_agentic_used_tools_false():
    aj = _make_judgment(tool_calls_made=0)
    result = compare_judgments(nonagentic_valid=True, nonagentic_score=0.80, agentic_judgment=aj)
    assert result.agentic_used_tools is False
