"""Tests for the explicit-tag split-criteria path in agentic_runner.py.

Covers _merge_split_criteria (presence-based merge, no keyword classifier)
and run_split_criteria_guardrail (parallel dispatch + merge), with both
sub-calls monkeypatched so no real LLM call happens.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agentic_guardrails"))

import agentic_runner
from agentic_runner import (
    _merge_split_criteria,
    _recompute_score_from_criteria,
    run_split_criteria_guardrail,
    AgenticJudgment,
)
from guardrails_runner import NonAgenticJudgment


# ── _merge_split_criteria ──────────────────────────────────────────────────────

def test_merge_keeps_agentic_verdict_for_criteria_it_evaluated():
    agentic = [{"criterion": "FACTUALITY AND ACCURACY", "verdict": "COMPLIANT"}]
    na = [
        {"criterion": "FACTUALITY AND ACCURACY", "verdict": "MINOR_ISSUE"},
        {"criterion": "SAFETY, SECURITY, AND PRIVACY", "verdict": "COMPLIANT"},
    ]
    merged, changed = _merge_split_criteria(agentic, na)

    assert len(merged) == 2
    fact = next(c for c in merged if c["criterion"] == "FACTUALITY AND ACCURACY")
    assert fact["verdict"] == "COMPLIANT"
    assert changed == ["FACTUALITY AND ACCURACY"]


def test_merge_falls_back_to_nonagentic_for_criteria_agentic_never_saw():
    agentic = [{"criterion": "FACTUALITY AND ACCURACY", "verdict": "COMPLIANT"}]
    na = [
        {"criterion": "FACTUALITY AND ACCURACY", "verdict": "COMPLIANT"},
        {"criterion": "TONE, DIGNITY, AND EMPATHY", "verdict": "COMPLIANT"},
    ]
    merged, changed = _merge_split_criteria(agentic, na)

    tone = next(c for c in merged if c["criterion"] == "TONE, DIGNITY, AND EMPATHY")
    assert tone["verdict"] == "COMPLIANT"
    assert tone["tool_influenced"] is False


def test_merge_no_change_when_verdicts_match():
    agentic = [{"criterion": "FACTUALITY AND ACCURACY", "verdict": "COMPLIANT"}]
    na = [{"criterion": "FACTUALITY AND ACCURACY", "verdict": "COMPLIANT"}]
    _, changed = _merge_split_criteria(agentic, na)
    assert changed == []


# ── run_split_criteria_guardrail ────────────────────────────────────────────────

TAGGED_POLICY = """POLICY

1. ACTIONABILITY AND PRACTICALITY (potentially needs tool calls)
- Must be actionable.

2. FACTUALITY AND ACCURACY (potentially needs tool calls)
- Must be correct.

3. SAFETY, SECURITY, AND PRIVACY
- Must not cause harm.

4. TONE, DIGNITY, AND EMPATHY
- Must be respectful.
"""

ALL_NONTOOL_POLICY = """POLICY

1. TONE, DIGNITY, AND EMPATHY
- Must be respectful.
"""


def test_split_criteria_guardrail_merges_parallel_results(monkeypatch):
    def fake_nonagentic(**kwargs):
        return NonAgenticJudgment(
            valid=True,
            score=0.9,
            explanation="na",
            criteria_verdicts=[
                {"criterion": "ACTIONABILITY AND PRACTICALITY", "verdict": "COMPLIANT"},
                {"criterion": "FACTUALITY AND ACCURACY", "verdict": "MINOR_ISSUE"},
                {"criterion": "SAFETY, SECURITY, AND PRIVACY", "verdict": "COMPLIANT"},
                {"criterion": "TONE, DIGNITY, AND EMPATHY", "verdict": "COMPLIANT"},
            ],
        )

    def fake_agentic(**kwargs):
        # Only evaluates the tool-requiring subset it was given — and finds
        # the factuality claim actually checks out once verified.
        assert "SAFETY" not in kwargs["policy_text"]
        assert "TONE" not in kwargs["policy_text"]
        return AgenticJudgment(
            valid=True,
            score=1.0,
            explanation="agentic",
            tool_calls_made=3,
            criteria_verdicts=[
                {"criterion": "ACTIONABILITY AND PRACTICALITY", "verdict": "COMPLIANT"},
                {"criterion": "FACTUALITY AND ACCURACY", "verdict": "COMPLIANT"},
            ],
        )

    monkeypatch.setattr(agentic_runner, "run_guardrail_for_policy", fake_nonagentic)
    monkeypatch.setattr(agentic_runner, "run_agentic_guardrail", fake_agentic)

    gr, aj = run_split_criteria_guardrail(
        guardrail=object(),
        provider="anthropic",
        guardrail_model="claude-sonnet-5",
        model_id="anthropic:claude-sonnet-5",
        policy_text=TAGGED_POLICY,
        rubric="",
        system_prompt="",
        user_message="scenario text",
        assistant_response="response text",
    )

    assert gr.score == 0.9  # non-agentic's own full-policy baseline, untouched
    assert aj.tool_changed_verdict_for == ["FACTUALITY AND ACCURACY"]
    assert aj.score == 1.0  # all 4 criteria compliant after merge
    verdict_by_criterion = {c["criterion"]: c["verdict"] for c in aj.criteria_verdicts}
    assert verdict_by_criterion["SAFETY, SECURITY, AND PRIVACY"] == "COMPLIANT"
    assert verdict_by_criterion["TONE, DIGNITY, AND EMPATHY"] == "COMPLIANT"


def test_split_criteria_guardrail_skips_agentic_when_no_tool_criteria(monkeypatch):
    def fake_nonagentic(**kwargs):
        return NonAgenticJudgment(
            valid=True,
            score=1.0,
            explanation="na",
            criteria_verdicts=[{"criterion": "TONE, DIGNITY, AND EMPATHY", "verdict": "COMPLIANT"}],
        )

    def fake_agentic(**kwargs):
        raise AssertionError("agentic judge should never be called when there are no tool-requiring criteria")

    monkeypatch.setattr(agentic_runner, "run_guardrail_for_policy", fake_nonagentic)
    monkeypatch.setattr(agentic_runner, "run_agentic_guardrail", fake_agentic)

    gr, aj = run_split_criteria_guardrail(
        guardrail=object(),
        provider="anthropic",
        guardrail_model="claude-sonnet-5",
        model_id="anthropic:claude-sonnet-5",
        policy_text=ALL_NONTOOL_POLICY,
        rubric="",
        system_prompt="",
        user_message="scenario text",
        assistant_response="response text",
    )

    assert aj.score == gr.score == 1.0
    assert aj.criteria_verdicts == gr.criteria_verdicts


# ── _recompute_score_from_criteria ──────────────────────────────────────────────

def _cv(criterion, verdict):
    return {"criterion": criterion, "verdict": verdict, "explanation": "test"}


def test_score_all_compliant():
    verdicts = [_cv("A", "COMPLIANT"), _cv("B", "COMPLIANT")]
    assert _recompute_score_from_criteria(verdicts) == 1.0


def test_score_one_major_issue():
    verdicts = [_cv("A", "MAJOR_ISSUE"), _cv("B", "COMPLIANT")]
    # 1.0 - 0.25 = 0.75
    assert _recompute_score_from_criteria(verdicts) == 0.75


def test_score_clamps_at_minimum():
    verdicts = [_cv(str(i), "CRITICAL") for i in range(5)]
    # 1.0 - 5*0.50 = -1.5 → clamped to 0.05
    assert _recompute_score_from_criteria(verdicts) == 0.05
