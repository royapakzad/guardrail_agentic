"""Tests for the full-policy judge mode in agentic_runner.py (Issue #91).

Both judge calls see the identical, full, untouched policy text -- the only
difference between them is tool access, not policy scope. Contrast with
run_split_criteria_guardrail (tests/test_split_criteria_guardrail.py), where
the agentic call only ever sees the tool-tagged subset.

Both sub-calls are monkeypatched, keyed by kwargs the real call received, so
no real LLM call happens.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agentic_guardrails"))

import agentic_runner
from agentic_runner import (
    build_agentic_guardrail_system_prompt_full_policy,
    run_full_policy_guardrail,
    AgenticJudgment,
)
from guardrails_runner import NonAgenticJudgment


FULL_POLICY = """POLICY

1. ACTIONABILITY AND PRACTICALITY (potentially needs tool calls)
- Must be actionable.

2. FACTUALITY AND ACCURACY (potentially needs tool calls)
- Must be correct.

3. SAFETY, SECURITY, AND PRIVACY
- Must not cause harm.

4. TONE, DIGNITY, AND EMPATHY
- Must be respectful.
"""


# ── run_full_policy_guardrail ───────────────────────────────────────────────

def test_both_calls_receive_the_identical_full_policy_text(monkeypatch):
    seen: dict = {}

    def fake_nonagentic(**kwargs):
        seen["nonagentic_policy_text"] = kwargs["policy_text"]
        return NonAgenticJudgment(
            valid=True,
            score=0.8,
            explanation="na",
            criteria_verdicts=[
                {"criterion": "ACTIONABILITY AND PRACTICALITY", "verdict": "COMPLIANT"},
                {"criterion": "FACTUALITY AND ACCURACY", "verdict": "NOT_FULLY_COMPLIANT"},
                {"criterion": "SAFETY, SECURITY, AND PRIVACY", "verdict": "COMPLIANT"},
                {"criterion": "TONE, DIGNITY, AND EMPATHY", "verdict": "COMPLIANT"},
            ],
        )

    def fake_agentic(**kwargs):
        seen["agentic_policy_text"] = kwargs["policy_text"]
        seen["full_policy_flag"] = kwargs.get("full_policy")
        # Unlike the split mode, the agentic call here judges ALL 4 criteria
        # in one shot, tool-verifying only the two tagged ones.
        return AgenticJudgment(
            valid=True,
            score=1.0,
            explanation="agentic",
            tool_calls_made=2,
            criteria_verdicts=[
                {"criterion": "ACTIONABILITY AND PRACTICALITY", "verdict": "COMPLIANT",
                 "tool_influenced": True, "tools_used": ["check_url_validity"]},
                {"criterion": "FACTUALITY AND ACCURACY", "verdict": "COMPLIANT",
                 "tool_influenced": True, "tools_used": ["search_web"]},
                {"criterion": "SAFETY, SECURITY, AND PRIVACY", "verdict": "COMPLIANT",
                 "tool_influenced": False, "tools_used": []},
                {"criterion": "TONE, DIGNITY, AND EMPATHY", "verdict": "COMPLIANT",
                 "tool_influenced": False, "tools_used": []},
            ],
        )

    monkeypatch.setattr(agentic_runner, "run_guardrail_for_policy", fake_nonagentic)
    monkeypatch.setattr(agentic_runner, "run_agentic_guardrail", fake_agentic)

    gr, aj = run_full_policy_guardrail(
        guardrail=object(),
        provider="anthropic",
        guardrail_model="claude-opus-4-6",
        model_id="anthropic:claude-opus-4-6",
        policy_text=FULL_POLICY,
        rubric="",
        system_prompt="",
        user_message="scenario text",
        assistant_response="response text",
    )

    # Both calls got the exact same, full, un-split policy text -- including
    # criteria 3 and 4, which run_split_criteria_guardrail would have stripped
    # out of the agentic call entirely.
    assert seen["nonagentic_policy_text"] == FULL_POLICY
    assert seen["agentic_policy_text"] == FULL_POLICY
    assert seen["full_policy_flag"] is True
    assert "SAFETY" in seen["agentic_policy_text"]
    assert "TONE" in seen["agentic_policy_text"]


def test_agentic_own_verdict_kept_for_untagged_criteria_no_substitution(monkeypatch):
    """
    Issue #91 decision: the agentic call's own (text-only) verdict for an
    untagged criterion is the recorded verdict -- it is NOT replaced by the
    non-agentic call's verdict, even though they disagree here.
    """
    def fake_nonagentic(**kwargs):
        return NonAgenticJudgment(
            valid=True,
            score=0.8,
            explanation="na",
            criteria_verdicts=[
                {"criterion": "SAFETY, SECURITY, AND PRIVACY", "verdict": "NOT_FULLY_COMPLIANT"},
            ],
        )

    def fake_agentic(**kwargs):
        return AgenticJudgment(
            valid=True,
            score=1.0,
            explanation="agentic",
            tool_calls_made=0,
            criteria_verdicts=[
                {"criterion": "SAFETY, SECURITY, AND PRIVACY", "verdict": "COMPLIANT",
                 "tool_influenced": False, "tools_used": []},
            ],
        )

    monkeypatch.setattr(agentic_runner, "run_guardrail_for_policy", fake_nonagentic)
    monkeypatch.setattr(agentic_runner, "run_agentic_guardrail", fake_agentic)

    gr, aj = run_full_policy_guardrail(
        guardrail=object(),
        provider="anthropic",
        guardrail_model="claude-opus-4-6",
        model_id="anthropic:claude-opus-4-6",
        policy_text="1. SAFETY, SECURITY, AND PRIVACY\n- Must not cause harm.\n",
        rubric="",
        system_prompt="",
        user_message="scenario text",
        assistant_response="response text",
    )

    assert gr.score == 0.8  # non-agentic baseline untouched
    assert aj.criteria_verdicts[0]["verdict"] == "COMPLIANT"  # agentic's own verdict wins
    assert aj.tool_changed_verdict_for == ["SAFETY, SECURITY, AND PRIVACY"]


def test_tool_changed_verdict_for_spans_the_whole_policy(monkeypatch):
    """Unlike split mode, tool_changed_verdict_for can now include untagged
    criteria too -- since both sides judge every criterion, a disagreement on
    an untagged one is itself a signal worth surfacing, even though no tool
    caused it."""
    def fake_nonagentic(**kwargs):
        return NonAgenticJudgment(
            valid=True, score=0.6, explanation="na",
            criteria_verdicts=[
                {"criterion": "A", "verdict": "COMPLIANT"},
                {"criterion": "B", "verdict": "COMPLIANT"},
            ],
        )

    def fake_agentic(**kwargs):
        return AgenticJudgment(
            valid=True, score=0.6, explanation="agentic", tool_calls_made=1,
            criteria_verdicts=[
                {"criterion": "A", "verdict": "NOT_FULLY_COMPLIANT"},  # tagged, tool-driven flip
                {"criterion": "B", "verdict": "NOT_FULLY_COMPLIANT"},  # untagged, unexpected flip
            ],
        )

    monkeypatch.setattr(agentic_runner, "run_guardrail_for_policy", fake_nonagentic)
    monkeypatch.setattr(agentic_runner, "run_agentic_guardrail", fake_agentic)

    gr, aj = run_full_policy_guardrail(
        guardrail=object(), provider="anthropic", guardrail_model="claude-opus-4-6",
        model_id="anthropic:claude-opus-4-6",
        policy_text="1. A (potentially needs tool calls)\n- x.\n\n2. B\n- y.\n",
        rubric="", system_prompt="", user_message="s", assistant_response="r",
    )

    assert set(aj.tool_changed_verdict_for) == {"A", "B"}


def test_frozen_nonagentic_skips_the_nonagentic_call(monkeypatch):
    def fail_if_called(**kwargs):
        raise AssertionError("non-agentic judge should not be called when frozen_nonagentic is supplied")

    def fake_agentic(**kwargs):
        return AgenticJudgment(
            valid=True, score=1.0, explanation="agentic", tool_calls_made=1,
            criteria_verdicts=[{"criterion": "A", "verdict": "COMPLIANT"}],
        )

    monkeypatch.setattr(agentic_runner, "run_guardrail_for_policy", fail_if_called)
    monkeypatch.setattr(agentic_runner, "run_agentic_guardrail", fake_agentic)

    frozen = NonAgenticJudgment(
        valid=True, score=0.9, explanation="frozen na",
        criteria_verdicts=[{"criterion": "A", "verdict": "NOT_FULLY_COMPLIANT"}],
    )

    gr, aj = run_full_policy_guardrail(
        guardrail=object(), provider="anthropic", guardrail_model="claude-opus-4-6",
        model_id="anthropic:claude-opus-4-6",
        policy_text="1. A\n- x.\n",
        rubric="", system_prompt="", user_message="s", assistant_response="r",
        frozen_nonagentic=frozen,
    )

    assert gr is frozen
    assert aj.tool_changed_verdict_for == ["A"]  # diffed against the frozen baseline


# ── build_agentic_guardrail_system_prompt_full_policy ───────────────────────

def test_full_policy_prompt_distinguishes_marked_and_unmarked():
    prompt = build_agentic_guardrail_system_prompt_full_policy(
        policy="1. TEST (potentially needs tool calls)\n- desc\n\n2. OTHER\n- desc",
        rubric="r",
        tool_group="humanitarian",
    )
    assert "MARKED" in prompt
    assert "UNMARKED" in prompt
    assert "Do NOT call any tool for an UNMARKED criterion" in prompt


def test_full_policy_prompt_lists_actual_tools_for_domain():
    prompt = build_agentic_guardrail_system_prompt_full_policy(
        policy="1. TEST\n- desc", rubric="r", tool_group="financial"
    )
    assert "entity_registration" in prompt
    assert "sanctions_screen" in prompt
    assert "reliefweb_situation" not in prompt  # humanitarian-only tool


def test_full_policy_prompt_embeds_policy_with_tags_intact():
    policy = "1. ACTIONABILITY (potentially needs tool calls)\n- x.\n\n2. TONE\n- y.\n"
    prompt = build_agentic_guardrail_system_prompt_full_policy(policy=policy, rubric="r")
    # Unlike the split-mode prompt, the tag is NOT stripped -- it's the only
    # signal telling the model which criteria to verify with tools.
    assert "(potentially needs tool calls)" in prompt
