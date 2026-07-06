"""Tests for Issue #23 — criteria split by tool-need.

Verifies that:
1. criterion_needs_tools() correctly classifies criteria by name.
2. _merge_with_nonagentic() enforces agentic == non-agentic for non-tool criteria.
3. _merge_with_nonagentic() keeps agentic verdicts for tool-requiring criteria and
   populates tool_changed_verdict_for only for those criteria where verdicts differ.
4. _recompute_score_from_criteria() applies the shared deduction table correctly.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agentic_guardrails"))

from agentic_runner import (
    criterion_needs_tools,
    _merge_with_nonagentic,
    _recompute_score_from_criteria,
)


# ── criterion_needs_tools ──────────────────────────────────────────────────────

def test_factuality_needs_tools():
    assert criterion_needs_tools("Factuality Check") is True

def test_accuracy_needs_tools():
    assert criterion_needs_tools("Information Accuracy") is True

def test_actionability_needs_tools():
    assert criterion_needs_tools("Actionability") is True

def test_url_needs_tools():
    assert criterion_needs_tools("URL Validity") is True

def test_claim_needs_tools():
    assert criterion_needs_tools("Claim Verification") is True

def test_tone_does_not_need_tools():
    assert criterion_needs_tools("Tone and Empathy") is False

def test_safety_does_not_need_tools():
    assert criterion_needs_tools("Safety") is False

def test_dignity_does_not_need_tools():
    assert criterion_needs_tools("Human Dignity") is False

def test_privacy_does_not_need_tools():
    assert criterion_needs_tools("Privacy Protection") is False

def test_multilingual_does_not_need_tools():
    assert criterion_needs_tools("Multilingual Quality") is False

def test_freedom_of_information_does_not_need_tools():
    assert criterion_needs_tools("Freedom of Information") is False


# ── _merge_with_nonagentic ─────────────────────────────────────────────────────

def _cv(criterion, verdict):
    return {"criterion": criterion, "verdict": verdict, "explanation": "test"}


def test_non_tool_criterion_gets_nonagentic_verdict():
    agentic = [_cv("Tone and Empathy", "MAJOR_ISSUE")]
    nonagentic = [_cv("Tone and Empathy", "COMPLIANT")]

    merged, tool_changed = _merge_with_nonagentic(agentic, nonagentic)

    assert len(merged) == 1
    assert merged[0]["verdict"] == "COMPLIANT"  # non-agentic wins
    assert merged[0]["tool_influenced"] is False
    assert tool_changed == []  # not a tool criterion — excluded from this list


def test_tool_criterion_keeps_agentic_verdict():
    agentic = [_cv("Factuality Check", "CRITICAL")]
    nonagentic = [_cv("Factuality Check", "COMPLIANT")]

    merged, tool_changed = _merge_with_nonagentic(agentic, nonagentic)

    assert merged[0]["verdict"] == "CRITICAL"  # agentic wins for tool criteria
    assert "Factuality Check" in tool_changed


def test_tool_criterion_not_in_changed_when_same_verdict():
    agentic = [_cv("Factuality Check", "COMPLIANT")]
    nonagentic = [_cv("Factuality Check", "COMPLIANT")]

    merged, tool_changed = _merge_with_nonagentic(agentic, nonagentic)

    assert merged[0]["verdict"] == "COMPLIANT"
    assert tool_changed == []  # no change, so not listed


def test_mixed_criteria_splits_correctly():
    agentic = [
        _cv("Factuality Check", "CRITICAL"),   # tool — keep agentic
        _cv("Tone and Empathy", "MAJOR_ISSUE"),  # non-tool — use non-agentic
        _cv("Actionability", "MINOR_ISSUE"),   # tool — keep agentic
        _cv("Safety", "CRITICAL"),              # non-tool — use non-agentic
    ]
    nonagentic = [
        _cv("Factuality Check", "COMPLIANT"),
        _cv("Tone and Empathy", "COMPLIANT"),
        _cv("Actionability", "COMPLIANT"),
        _cv("Safety", "COMPLIANT"),
    ]

    merged, tool_changed = _merge_with_nonagentic(agentic, nonagentic)

    verdicts = {cv["criterion"]: cv["verdict"] for cv in merged}
    assert verdicts["Factuality Check"] == "CRITICAL"    # agentic kept
    assert verdicts["Tone and Empathy"] == "COMPLIANT"   # non-agentic override
    assert verdicts["Actionability"] == "MINOR_ISSUE"    # agentic kept
    assert verdicts["Safety"] == "COMPLIANT"             # non-agentic override

    # Only tool criteria with changed verdicts appear here
    assert set(tool_changed) == {"Factuality Check", "Actionability"}


# ── _recompute_score_from_criteria ─────────────────────────────────────────────

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
