"""Tests for get_specialized_tool_names / get_tool_schemas(only=...) and the
reserved-window mechanism in run_agentic_guardrail (specialized_tool_reserve).

Together these exist so the agentic judge's small tool-call budget isn't
entirely spent on generic tools (search_web, fetch_url, check_url_validity)
before a domain's specialized tools (sanctions_screen, urlscan_check, ...)
ever get a turn -- see financial-run logs referenced in tools.py's TOOL_GROUPS
comment: 0/93 specialized calls, 60% of runs budget-exhausted on generic
tools alone.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agentic_guardrails"))

from types import SimpleNamespace

import agentic_runner
import tools
from tools import get_specialized_tool_names, get_tool_schemas


# ── get_specialized_tool_names / get_tool_schemas(only=...) ────────────────


def test_default_group_has_no_specialized_tools():
    assert get_specialized_tool_names("default") == []


def test_financial_specialized_names_exclude_generic_tools():
    names = get_specialized_tool_names("financial")
    assert names == [
        "entity_registration",
        "sanctions_screen",
        "broker_license_check",
        "filing_search",
        "crypto_address_screen",
    ]
    assert "search_web" not in names
    assert "check_url_validity" not in names


def test_check_acronym_not_in_any_live_group():
    # check_acronym stays registered (dispatchable) but is deliberately not
    # offered to the model in any group -- pre-run acronym checks cover it
    # at zero tool-call-budget cost instead (see tools.py's TOOL_GROUPS comment).
    for group in ("default", "humanitarian", "financial", "cybersecurity"):
        names = [s["function"]["name"] for s in get_tool_schemas(group)]
        assert "check_acronym" not in names
    assert "check_acronym" in tools.REGISTRY


def test_get_tool_schemas_only_narrows_to_the_given_names():
    schemas = get_tool_schemas("financial", only=["sanctions_screen", "search_web"])
    names = [s["function"]["name"] for s in schemas]
    # Order follows the group's own ordering, not the `only` list's order.
    assert names == ["search_web", "sanctions_screen"]


def test_get_tool_schemas_only_ignores_names_outside_the_group():
    # "urlscan_check" isn't in the financial group, so it's dropped even
    # though it's a real, registered tool name.
    schemas = get_tool_schemas("financial", only=["urlscan_check", "sanctions_screen"])
    names = [s["function"]["name"] for s in schemas]
    assert names == ["sanctions_screen"]


# ── Reserved-window behavior inside run_agentic_guardrail ──────────────────


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str = "{}"):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)


def _fake_response(*, tool_calls=None, content=""):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def test_reserved_window_narrows_tools_to_specialized_only(monkeypatch):
    """
    max_tool_calls=4, specialized_tool_reserve=2, tool_group="financial":
    turns 1-2 (calls_remaining 4, 3) should see the full financial tool list;
    turns 3-4 (calls_remaining 2, 1) should see only the specialized subset.
    """
    calls_seen: list[list[str]] = []

    def fake_completion(**kwargs):
        tools_offered = [t["function"]["name"] for t in kwargs.get("tools", [])]
        calls_seen.append(tools_offered)
        turn = len(calls_seen)
        if turn < 4:
            # Keep making generic/specialized calls to burn through the budget
            # without ever finishing early.
            name = "search_web" if turn <= 2 else "sanctions_screen"
            return _fake_response(tool_calls=[_FakeToolCall(f"call_{turn}", name)])
        # Final turn: no more tools offered (budget exhausted) -- emit judgment.
        return _fake_response(content='{"score": 1.0, "explanation": "done", "criteria_verdicts": []}')

    monkeypatch.setattr(agentic_runner, "_completion_with_retry", fake_completion)
    monkeypatch.setattr(tools, "dispatch_tool_call", lambda name, args: "{}")

    agentic_runner.run_agentic_guardrail(
        provider="openai",
        guardrail_model="gpt-5-mini",
        policy_text="1. TEST\n- desc",
        rubric="r",
        system_prompt="",
        user_message="hello",
        assistant_response="no urls or acronyms here",
        max_tool_calls=4,
        tool_group="financial",
        specialized_tool_reserve=2,
    )

    # Turns 1-2: full financial group (generic + specialized tools) offered.
    assert "search_web" in calls_seen[0]
    assert "sanctions_screen" in calls_seen[0]
    assert "search_web" in calls_seen[1]
    assert "sanctions_screen" in calls_seen[1]

    # Turns 3-4: reserved window -- generic tools dropped, only specialized left.
    assert "search_web" not in calls_seen[2]
    assert set(calls_seen[2]) == set(get_specialized_tool_names("financial"))
    assert "search_web" not in calls_seen[3]
    assert set(calls_seen[3]) == set(get_specialized_tool_names("financial"))


def test_reserve_never_eats_the_first_call(monkeypatch):
    """With max_tool_calls=1, the reserve must not block Phase 1's
    check_url_validity from ever being offered on the only available turn."""
    calls_seen: list[list[str]] = []

    def fake_completion(**kwargs):
        calls_seen.append([t["function"]["name"] for t in kwargs.get("tools", [])])
        return _fake_response(content='{"score": 1.0, "explanation": "done", "criteria_verdicts": []}')

    monkeypatch.setattr(agentic_runner, "_completion_with_retry", fake_completion)
    monkeypatch.setattr(tools, "dispatch_tool_call", lambda name, args: "{}")

    agentic_runner.run_agentic_guardrail(
        provider="openai",
        guardrail_model="gpt-5-mini",
        policy_text="1. TEST\n- desc",
        rubric="r",
        system_prompt="",
        user_message="hello",
        assistant_response="no urls or acronyms here",
        max_tool_calls=1,
        tool_group="financial",
        specialized_tool_reserve=2,
    )

    assert "check_url_validity" in calls_seen[0]


def test_reserve_is_a_noop_for_the_default_group(monkeypatch):
    """The default tool group has no specialized tools, so the reserved
    window must never kick in -- the full (generic-only) list stays offered
    right up to budget exhaustion."""
    calls_seen: list[list[str]] = []

    def fake_completion(**kwargs):
        tools_offered = [t["function"]["name"] for t in kwargs.get("tools", [])]
        calls_seen.append(tools_offered)
        turn = len(calls_seen)
        if turn < 3:
            return _fake_response(tool_calls=[_FakeToolCall(f"call_{turn}", "search_web")])
        return _fake_response(content='{"score": 1.0, "explanation": "done", "criteria_verdicts": []}')

    monkeypatch.setattr(agentic_runner, "_completion_with_retry", fake_completion)
    monkeypatch.setattr(tools, "dispatch_tool_call", lambda name, args: "{}")

    agentic_runner.run_agentic_guardrail(
        provider="openai",
        guardrail_model="gpt-5-mini",
        policy_text="1. TEST\n- desc",
        rubric="r",
        system_prompt="",
        user_message="hello",
        assistant_response="no urls or acronyms here",
        max_tool_calls=3,
        tool_group="default",
        specialized_tool_reserve=2,
    )

    for offered in calls_seen:
        assert set(offered) == {"search_web", "fetch_url", "check_url_validity"}
