"""Tests for tools.py error handling and check_acronym heuristic (network-free)."""
import json
from unittest.mock import patch

import pytest

from tools import ToolError, check_acronym, dispatch_tool_call


# ── dispatch_tool_call error handling ─────────────────────────────────────────


def test_dispatch_unknown_tool_returns_error_json():
    result_str = dispatch_tool_call("nonexistent_tool_xyz", "{}")
    result = json.loads(result_str)
    assert "error" in result
    assert "nonexistent_tool_xyz" in result["error"]


def test_dispatch_bad_arguments_json_returns_error():
    result_str = dispatch_tool_call("search_web", "not-valid-json!!!")
    result = json.loads(result_str)
    assert "error" in result


def test_dispatch_empty_arguments_for_unknown_tool():
    result_str = dispatch_tool_call("mystery_tool", "")
    result = json.loads(result_str)
    assert "error" in result


# ── check_acronym heuristic ───────────────────────────────────────────────────


def test_check_acronym_likely_correct_when_words_match():
    mock_results = [
        {
            "title": "WHO - World Health Organization",
            "url": "https://who.int",
            "snippet": "World Health Organization is the United Nations agency.",
        }
    ]
    with patch("tools.search_web", return_value=mock_results):
        result = check_acronym("WHO", "World Health Organization", "en")
    assert result["verdict_hint"] == "likely_correct"
    assert result["match_score"] >= 0.5


def test_check_acronym_likely_wrong_when_words_absent():
    mock_results = [
        {
            "title": "NATO - North Atlantic Treaty Organization",
            "url": "https://nato.int",
            "snippet": "NATO is a military alliance in the North Atlantic.",
        }
    ]
    with patch("tools.search_web", return_value=mock_results):
        result = check_acronym("NATO", "Nonsense Acronym That Obfuscates", "en")
    assert result["match_score"] < 0.3


def test_check_acronym_error_on_search_failure():
    with patch("tools.search_web", side_effect=ToolError("search down")):
        result = check_acronym("TEST", "Totally Erroneous Silly Thing", "en")
    assert "error" in result
    assert result["verdict_hint"] == "unclear"
    assert result["match_score"] == 0.0


def test_check_acronym_returns_search_results():
    mock_results = [
        {"title": "UN", "url": "https://un.org", "snippet": "United Nations organization."}
    ]
    with patch("tools.search_web", return_value=mock_results):
        result = check_acronym("UN", "United Nations", "en")
    assert "search_results" in result
    assert len(result["search_results"]) == 1


def test_check_acronym_includes_note_field():
    with patch("tools.search_web", return_value=[]):
        result = check_acronym("XYZ", "Some Expansion", "en")
    # Even with empty results it should return a structured dict (no error)
    # or a dict with error key — in either case 'acronym' is present
    assert "acronym" in result
    assert result["acronym"] == "XYZ"
