"""Tests for criterion-tagged tool calls and the code-computed verification
that cross-checks each criterion's self-reported `tools_used` against what
was actually tagged in tool_call_log (agentic_runner._verify_tool_criterion_links).

Context: the agentic judge's criteria_verdicts JSON always let the model
*claim* which tools it used for a given criterion (tools_used, tool_influenced,
human_review_needed) -- but nothing checked that claim against the tool calls
it actually made. Every tool schema now carries a required `criterion`
argument (tools.py's _register) so tool_call_log entries can be grouped by
criterion, and _verify_tool_criterion_links() diffs that ground truth against
the judge's own tools_used claim.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agentic_guardrails"))

from types import SimpleNamespace

import agentic_runner
import tools
from tools import REGISTRY, get_tool_schemas


# ── Schema: every registered tool requires `criterion` ─────────────────────


def test_every_registered_tool_requires_criterion():
    for name, tool in REGISTRY.items():
        params = tool.schema["function"]["parameters"]
        assert "criterion" in params["properties"], f"{name} missing criterion property"
        assert "criterion" in params["required"], f"{name} does not require criterion"


def test_criterion_param_present_in_every_group():
    for group in ("default", "humanitarian", "financial", "cybersecurity"):
        for schema in get_tool_schemas(group):
            assert "criterion" in schema["function"]["parameters"]["properties"]


# ── _verify_tool_criterion_links ────────────────────────────────────────────


def test_verified_when_claimed_tool_matches_a_tagged_call():
    criteria_verdicts = [
        {"criterion": "URL SAFETY", "verdict": "COMPLIANT", "tools_used": ["check_url_validity"]},
    ]
    tool_call_log = [
        {"call_number": 1, "tool": "check_url_validity", "criterion": "URL SAFETY"},
    ]
    out = agentic_runner._verify_tool_criterion_links(criteria_verdicts, tool_call_log)
    assert out[0]["tools_actually_tagged"] == ["check_url_validity"]
    assert out[0]["tools_used_verified"] is True


def test_unverified_when_judge_claims_a_tool_it_never_tagged_to_this_criterion():
    # The judge's final JSON claims search_web supported this criterion, but
    # the only tool call actually tagged to it was check_url_validity --
    # search_web was tagged to a different criterion (or not tagged at all).
    criteria_verdicts = [
        {"criterion": "FACTUAL ACCURACY", "verdict": "NOT_FULLY_COMPLIANT", "tools_used": ["search_web"]},
    ]
    tool_call_log = [
        {"call_number": 1, "tool": "check_url_validity", "criterion": "FACTUAL ACCURACY"},
        {"call_number": 2, "tool": "search_web", "criterion": "SOMETHING ELSE"},
    ]
    out = agentic_runner._verify_tool_criterion_links(criteria_verdicts, tool_call_log)
    assert out[0]["tools_actually_tagged"] == ["check_url_validity"]
    assert out[0]["tools_used_verified"] is False


def test_unclaimed_criterion_with_no_tools_used_is_not_verified_but_not_flagged_wrong():
    # tools_used empty (judge said no tool applied) -- tools_used_verified is
    # False (nothing to confirm), but this is a different case from a false
    # claim: distinguish by checking tools_used itself is also empty.
    criteria_verdicts = [
        {"criterion": "TONE", "verdict": "COMPLIANT", "tools_used": []},
    ]
    out = agentic_runner._verify_tool_criterion_links(criteria_verdicts, tool_call_log=[])
    assert out[0]["tools_actually_tagged"] == []
    assert out[0]["tools_used_verified"] is False
    assert out[0]["tools_used"] == []  # unchanged -- verified flag alone disambiguates


def test_fuzzy_matches_slightly_different_criterion_text():
    # The tool call's free-text criterion tag and the final criteria_verdicts
    # criterion string are both the model's own writing and can drift
    # slightly (e.g. a trailing annotation _normalize_criterion_name doesn't
    # catch, or minor rewording) -- fuzzy matching should still link them.
    criteria_verdicts = [
        {"criterion": "REGULATORY DISCLOSURE REQUIREMENTS", "verdict": "COMPLIANT",
         "tools_used": ["sanctions_screen"]},
    ]
    tool_call_log = [
        {"call_number": 1, "tool": "sanctions_screen", "criterion": "REGULATORY DISCLOSURE REQUIREMENT"},
    ]
    out = agentic_runner._verify_tool_criterion_links(criteria_verdicts, tool_call_log)
    assert out[0]["tools_used_verified"] is True


def test_unrelated_criterion_tag_does_not_fuzzy_match():
    criteria_verdicts = [
        {"criterion": "ACRONYM ACCURACY", "verdict": "COMPLIANT", "tools_used": ["search_web"]},
    ]
    tool_call_log = [
        {"call_number": 1, "tool": "search_web", "criterion": "SCAM GUIDANCE FOR WIRE TRANSFERS"},
    ]
    out = agentic_runner._verify_tool_criterion_links(criteria_verdicts, tool_call_log)
    assert out[0]["tools_actually_tagged"] == []
    assert out[0]["tools_used_verified"] is False


# ── End-to-end through run_agentic_guardrail ────────────────────────────────


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)


def _fake_response(*, tool_calls=None, content=""):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def test_end_to_end_tool_call_log_and_criteria_verdicts_carry_criterion_links(monkeypatch):
    """
    The model calls check_url_validity tagged to "LINK SAFETY", then finishes
    with a criteria_verdicts entry honestly claiming check_url_validity for
    that same criterion. tool_call_log should carry the criterion tag, and
    the returned AgenticJudgment's criteria_verdicts should confirm the link.
    """
    calls_seen: list[list[str]] = []

    def fake_completion(**kwargs):
        calls_seen.append(kwargs)
        turn = len(calls_seen)
        if turn == 1:
            return _fake_response(tool_calls=[
                _FakeToolCall(
                    "call_1", "check_url_validity",
                    '{"url": "https://example.org", "criterion": "LINK SAFETY"}',
                )
            ])
        return _fake_response(content=(
            '{"score": 1.0, "explanation": "1. LINK SAFETY: ok -> Verdict: compliant", '
            '"criteria_verdicts": [{"criterion": "LINK SAFETY", "verdict": "COMPLIANT", '
            '"human_review_needed": "check_url_validity confirmed https://example.org", '
            '"suggested_improvement": "", "tool_influenced": true, '
            '"tools_used": ["check_url_validity"]}]}'
        ))

    monkeypatch.setattr(agentic_runner, "_completion_with_retry", fake_completion)
    monkeypatch.setattr(
        tools, "dispatch_tool_call",
        lambda name, args: '{"url": "https://example.org", "valid": true, "status_code": 200}',
    )

    result = agentic_runner.run_agentic_guardrail(
        provider="openai",
        guardrail_model="gpt-5-mini",
        policy_text="1. LINK SAFETY\n- desc",
        rubric="r",
        system_prompt="",
        user_message="scenario",
        assistant_response="Visit https://example.org for details.",
        max_tool_calls=4,
        tool_group="default",
    )

    assert result.tool_call_log[0]["criterion"] == "LINK SAFETY"
    cv = result.criteria_verdicts[0]
    assert cv["tools_actually_tagged"] == ["check_url_validity"]
    assert cv["tools_used_verified"] is True


def test_end_to_end_flags_unverified_when_judge_overclaims(monkeypatch):
    """
    The model never calls any tool, but the final JSON dishonestly claims
    check_url_validity supported the verdict. tools_used_verified must be
    False since tool_call_log has nothing tagged to this criterion.
    """
    def fake_completion(**kwargs):
        return _fake_response(content=(
            '{"score": 1.0, "explanation": "1. LINK SAFETY: ok -> Verdict: compliant", '
            '"criteria_verdicts": [{"criterion": "LINK SAFETY", "verdict": "COMPLIANT", '
            '"human_review_needed": "check_url_validity confirmed the link", '
            '"suggested_improvement": "", "tool_influenced": true, '
            '"tools_used": ["check_url_validity"]}]}'
        ))

    monkeypatch.setattr(agentic_runner, "_completion_with_retry", fake_completion)

    result = agentic_runner.run_agentic_guardrail(
        provider="openai",
        guardrail_model="gpt-5-mini",
        policy_text="1. LINK SAFETY\n- desc",
        rubric="r",
        system_prompt="",
        user_message="scenario",
        assistant_response="No links here.",
        max_tool_calls=4,
        tool_group="default",
    )

    assert result.tool_call_log == []
    cv = result.criteria_verdicts[0]
    assert cv["tools_actually_tagged"] == []
    assert cv["tools_used_verified"] is False
