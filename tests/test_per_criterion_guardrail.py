"""Tests for the per-criterion parallel judge mode in agentic_runner.py
(Issue #74) -- one independent LLM call per criterion, instead of the
two-call split (non-agentic full policy + agentic tool-tagged subset) in
run_split_criteria_guardrail.

Both sub-calls are monkeypatched per-criterion, keyed by which single
criterion's policy_text was passed in, so no real LLM call happens.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agentic_guardrails"))

import agentic_runner
from agentic_runner import run_per_criterion_guardrail, AgenticJudgment
from guardrails_runner import NonAgenticJudgment

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

# One non-agentic verdict per criterion, keyed by criterion name -- the fake
# call below picks the matching entry based on which single-criterion
# policy_text it was given.
_NA_VERDICTS = {
    "ACTIONABILITY AND PRACTICALITY": "COMPLIANT",
    "FACTUALITY AND ACCURACY": "NOT_FULLY_COMPLIANT",
    "SAFETY, SECURITY, AND PRIVACY": "COMPLIANT",
    "TONE, DIGNITY, AND EMPATHY": "COMPLIANT",
}

# Agentic (tool-enabled) verdict for just the two tool-tagged criteria --
# factuality flips to COMPLIANT once verified, actionability doesn't change.
_AG_VERDICTS = {
    "ACTIONABILITY AND PRACTICALITY": "COMPLIANT",
    "FACTUALITY AND ACCURACY": "COMPLIANT",
}


def _criterion_in(policy_text: str, candidates: dict) -> str:
    for name in candidates:
        if name in policy_text:
            return name
    raise AssertionError(f"no known criterion name found in policy_text: {policy_text!r}")


def _fake_nonagentic(**kwargs):
    name = _criterion_in(kwargs["policy_text"], _NA_VERDICTS)
    return NonAgenticJudgment(
        valid=True,
        score=1.0 if _NA_VERDICTS[name] == "COMPLIANT" else 0.8,
        explanation=f"na for {name}",
        criteria_verdicts=[{"criterion": name, "verdict": _NA_VERDICTS[name]}],
    )


def _fake_agentic(**kwargs):
    name = _criterion_in(kwargs["policy_text"], _AG_VERDICTS)
    # Each call must be scoped to exactly one criterion -- not the other
    # tool-tagged one, and not any non-tool criterion.
    other_tool_criteria = [n for n in _AG_VERDICTS if n != name]
    assert all(other not in kwargs["policy_text"] for other in other_tool_criteria)
    assert "SAFETY" not in kwargs["policy_text"]
    assert "TONE" not in kwargs["policy_text"]
    return AgenticJudgment(
        valid=True,
        score=1.0,
        explanation=f"agentic for {name}",
        tool_calls_made=2,
        criteria_verdicts=[{"criterion": name, "verdict": _AG_VERDICTS[name]}],
    )


def test_per_criterion_guardrail_fires_one_call_per_criterion_plus_one_per_tool_criterion(monkeypatch):
    na_calls = []
    ag_calls = []

    def counting_nonagentic(**kwargs):
        na_calls.append(kwargs["policy_text"])
        return _fake_nonagentic(**kwargs)

    def counting_agentic(**kwargs):
        ag_calls.append(kwargs["policy_text"])
        return _fake_agentic(**kwargs)

    monkeypatch.setattr(agentic_runner, "run_guardrail_for_policy", counting_nonagentic)
    monkeypatch.setattr(agentic_runner, "run_agentic_guardrail", counting_agentic)

    run_per_criterion_guardrail(
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

    # 4 criteria -> 4 non-agentic calls; 2 tool-tagged -> 2 additional agentic calls.
    assert len(na_calls) == 4
    assert len(ag_calls) == 2


def test_per_criterion_guardrail_merges_tool_criteria_from_own_agentic_call(monkeypatch):
    monkeypatch.setattr(agentic_runner, "run_guardrail_for_policy", _fake_nonagentic)
    monkeypatch.setattr(agentic_runner, "run_agentic_guardrail", _fake_agentic)

    gr, aj = run_per_criterion_guardrail(
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

    na_by_criterion = {c["criterion"]: c["verdict"] for c in gr.criteria_verdicts}
    assert na_by_criterion == _NA_VERDICTS

    ag_by_criterion = {c["criterion"]: c["verdict"] for c in aj.criteria_verdicts}
    # Tool criteria: agentic's own verdict.
    assert ag_by_criterion["FACTUALITY AND ACCURACY"] == "COMPLIANT"
    assert ag_by_criterion["ACTIONABILITY AND PRACTICALITY"] == "COMPLIANT"
    # Non-tool criteria: non-agentic verdict carried forward unchanged.
    assert ag_by_criterion["SAFETY, SECURITY, AND PRIVACY"] == "COMPLIANT"
    assert ag_by_criterion["TONE, DIGNITY, AND EMPATHY"] == "COMPLIANT"

    # Factuality flipped NOT_FULLY_COMPLIANT -> COMPLIANT once tool-verified.
    assert aj.tool_changed_verdict_for == ["FACTUALITY AND ACCURACY"]


def test_per_criterion_guardrail_aggregates_tool_usage_across_tool_criteria(monkeypatch):
    monkeypatch.setattr(agentic_runner, "run_guardrail_for_policy", _fake_nonagentic)
    monkeypatch.setattr(agentic_runner, "run_agentic_guardrail", _fake_agentic)

    _, aj = run_per_criterion_guardrail(
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

    # 2 tool-tagged criteria, 2 tool_calls_made each (see _fake_agentic).
    assert aj.tool_calls_made == 4


def test_per_criterion_guardrail_skips_agentic_calls_when_no_tool_criteria(monkeypatch):
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

    gr, aj = run_per_criterion_guardrail(
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
    # Same criterion/verdict pairs carried forward -- aj additionally tags
    # each as tool_influenced=False (consistent regardless of whether *any*
    # criterion in the policy needs tools), which gr's own verdicts don't carry.
    assert {c["criterion"]: c["verdict"] for c in aj.criteria_verdicts} == {
        c["criterion"]: c["verdict"] for c in gr.criteria_verdicts
    }
    assert all(c["tool_influenced"] is False for c in aj.criteria_verdicts)
    assert aj.tool_calls_made == 0
