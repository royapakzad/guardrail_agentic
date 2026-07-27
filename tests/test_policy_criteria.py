"""Tests for policy_criteria.py — explicit, author-driven tool-need tagging.

Replaces the keyword-guessing classifier (criterion_needs_tools) that
mis-sorted criteria on coincidental substring matches (e.g. "claim" inside
"disclaimers", or cybersecurity's "Recognition of Threat Indicators" never
matching any keyword at all).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agentic_guardrails"))

import policy_criteria as pc


SAMPLE_POLICY = """POLICY

1. ACTIONABILITY AND PRACTICALITY (potentially needs tool calls)
- The response must provide practical, effective, and directly helpful information.
- Must include specific steps.

2. FACTUALITY AND ACCURACY (Potentially needs tool calls)
- Any factual claims must be correct.

3. SAFETY, SECURITY, AND PRIVACY
- The response must not expose the user to harm.

4. TONE, DIGNITY, AND EMPATHY
- The response must be respectful.
"""

UNTAGGED_POLICY = """POLICY

1. EXPLAINABILITY AND ALGORITHMIC TRANSPARENCY
- Some requirement.

2. ACCURACY AND CURRENCY
- Some other requirement.
"""


def test_has_explicit_tool_tags_true_when_marker_present():
    assert pc.has_explicit_tool_tags(SAMPLE_POLICY) is True


def test_has_explicit_tool_tags_false_when_no_marker():
    assert pc.has_explicit_tool_tags(UNTAGGED_POLICY) is False


def test_parse_tagged_criteria_classifies_correctly():
    criteria = pc.parse_tagged_criteria(SAMPLE_POLICY)
    by_name = {c["name"]: c["needs_tools"] for c in criteria}
    assert by_name["ACTIONABILITY AND PRACTICALITY"] is True
    assert by_name["FACTUALITY AND ACCURACY"] is True
    assert by_name["SAFETY, SECURITY, AND PRIVACY"] is False
    assert by_name["TONE, DIGNITY, AND EMPATHY"] is False


def test_parse_tagged_criteria_strips_tag_from_name():
    criteria = pc.parse_tagged_criteria(SAMPLE_POLICY)
    names = [c["name"] for c in criteria]
    assert all("tool" not in n.lower() for n in names)


def test_parse_tagged_criteria_preserves_body_text():
    criteria = pc.parse_tagged_criteria(SAMPLE_POLICY)
    actionability = next(c for c in criteria if c["name"] == "ACTIONABILITY AND PRACTICALITY")
    assert "practical, effective" in actionability["body"]
    assert "specific steps" in actionability["body"]


def test_parse_tagged_criteria_raises_on_no_criteria():
    try:
        pc.parse_tagged_criteria("No numbered criteria here at all.")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_split_tagged_policy_separates_correctly():
    tool_text, nontool_text = pc.split_tagged_policy(SAMPLE_POLICY)
    assert "ACTIONABILITY AND PRACTICALITY" in tool_text
    assert "FACTUALITY AND ACCURACY" in tool_text
    assert "SAFETY, SECURITY, AND PRIVACY" not in tool_text
    assert "SAFETY, SECURITY, AND PRIVACY" in nontool_text
    assert "TONE, DIGNITY, AND EMPATHY" in nontool_text
    assert "ACTIONABILITY AND PRACTICALITY" not in nontool_text


def test_split_tagged_policy_preserves_preamble_in_both_halves():
    tool_text, nontool_text = pc.split_tagged_policy(SAMPLE_POLICY)
    assert tool_text.startswith("POLICY")
    assert nontool_text.startswith("POLICY")


def test_split_tagged_policy_renumbers_consecutively():
    # Tool-requiring criteria are #1 and #4 here — non-consecutive. A judge
    # shown "1." then "4." with nothing between tends to self-annotate its
    # output to disambiguate (e.g. "NAME (Policy 4)"), breaking merge
    # matching — so the split subset must renumber to 1, 2 instead of 1, 4.
    policy = """POLICY

1. ACTIONABILITY AND PRACTICALITY (potentially needs tool calls)
- Must be actionable.

2. SAFETY, SECURITY, AND PRIVACY
- Must not cause harm.

3. TONE, DIGNITY, AND EMPATHY
- Must be respectful.

4. FACTUALITY AND ACCURACY (potentially needs tool calls)
- Must be correct.
"""
    tool_text, nontool_text = pc.split_tagged_policy(policy)
    assert "1. ACTIONABILITY AND PRACTICALITY" in tool_text
    assert "2. FACTUALITY AND ACCURACY" in tool_text
    assert "4. FACTUALITY AND ACCURACY" not in tool_text
    assert "1. SAFETY, SECURITY, AND PRIVACY" in nontool_text
    assert "2. TONE, DIGNITY, AND EMPATHY" in nontool_text


def test_split_tagged_policy_empty_half_when_nothing_matches():
    all_nontool = """POLICY

1. TONE, DIGNITY, AND EMPATHY
- Be respectful.
"""
    tool_text, nontool_text = pc.split_tagged_policy(all_nontool)
    assert tool_text == ""
    assert "TONE, DIGNITY, AND EMPATHY" in nontool_text


def test_real_humanitarian_explicit_tool_selection_file_parses_correctly():
    path = os.path.join(
        os.path.dirname(__file__), "..", "config", "humanitarian_policy_explicit_tool_selection.txt"
    )
    with open(path, encoding="utf-8") as f:
        text = f.read()

    assert pc.has_explicit_tool_tags(text) is True
    criteria = pc.parse_tagged_criteria(text)
    by_name = {c["name"]: c["needs_tools"] for c in criteria}

    assert by_name["ACTIONABILITY AND PRACTICALITY"] is True
    assert by_name["FACTUALITY AND ACCURACY"] is True
    assert by_name["SAFETY, SECURITY, AND PRIVACY"] is False
    assert by_name["TONE, DIGNITY, AND EMPATHY"] is False
    assert by_name["NON-DISCRIMINATION AND FAIRNESS"] is False
    assert by_name["FREEDOM OF ACCESS TO INFORMATION, CENSORSHIP, AND REFUSAL"] is False
