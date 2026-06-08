"""Tests for tools.py dispatch error handling and the check_acronym heuristic.

No network is touched: search_web is monkeypatched.
"""

import json

import tools
from tools import ToolError, check_acronym, dispatch_tool_call


def test_dispatch_bad_json_arguments_returns_error():
    out = json.loads(dispatch_tool_call("search_web", "{not valid json"))
    assert "error" in out
    assert "parse arguments" in out["error"].lower()


def test_dispatch_unknown_tool_returns_error():
    out = json.loads(dispatch_tool_call("does_not_exist", "{}"))
    assert "Unknown tool" in out["error"]


def test_dispatch_empty_arguments_is_allowed():
    # Empty args parse to {}; an unknown tool still reports cleanly.
    out = json.loads(dispatch_tool_call("does_not_exist", ""))
    assert "Unknown tool" in out["error"]


def test_check_acronym_confirms_correct_expansion(monkeypatch):
    def fake_search(query, max_results=3):
        return [
            {
                "title": "OFPRA",
                "url": "https://www.ofpra.gouv.fr",
                "snippet": "Office Francais de Protection des Refugies et Apatrides",
            }
        ]

    monkeypatch.setattr(tools, "search_web", fake_search)
    result = check_acronym("OFPRA", "Office Francais de Protection des Refugies", "fr")
    assert result["match_score"] >= 0.6
    assert result["verdict_hint"] == "likely_correct"


def test_check_acronym_flags_wrong_expansion(monkeypatch):
    def fake_search(query, max_results=3):
        return [{"title": "Unrelated", "url": "https://example.com", "snippet": "nothing relevant here"}]

    monkeypatch.setattr(tools, "search_web", fake_search)
    result = check_acronym("WHO", "World Health Organization", "en")
    assert result["match_score"] < 0.25
    assert result["verdict_hint"] == "likely_wrong"


def test_check_acronym_handles_search_failure(monkeypatch):
    def failing_search(query, max_results=3):
        raise ToolError("search backend down")

    monkeypatch.setattr(tools, "search_web", failing_search)
    result = check_acronym("NATO", "North Atlantic Treaty Organization")
    assert result["verdict_hint"] == "unclear"
    assert result["match_score"] == 0.0
    assert "error" in result
