"""Tests for agentic_runner pure-logic helpers (network-free)."""
import json

import pytest

from agentic_runner import (
    _extract_json_candidates,
    _infer_confidence,
    _rederive_score_from_explanation,
    _summarize_tool_result,
    parse_judgment_from_text,
)


# ── _extract_json_candidates ──────────────────────────────────────────────────


def test_extract_json_candidates_fenced_block():
    text = '```json\n{"score": 0.8}\n```'
    candidates = _extract_json_candidates(text)
    assert any('"score"' in c for c in candidates)


def test_extract_json_candidates_bare_object():
    text = 'prefix {"score": 0.9, "valid": true} suffix'
    candidates = _extract_json_candidates(text)
    parsed = [json.loads(c) for c in candidates if '"score"' in c]
    assert any(p.get("score") == 0.9 for p in parsed)


def test_extract_json_candidates_empty_text():
    assert _extract_json_candidates("") == []


def test_extract_json_candidates_no_json():
    assert _extract_json_candidates("no braces here at all") == []


# ── _rederive_score_from_explanation ─────────────────────────────────────────


def test_rederive_score_standard_pattern():
    exp = "Final score: max(0.05, 1.0 − 0.25) = 0.75"
    assert _rederive_score_from_explanation(exp) == 0.75


def test_rederive_score_hyphen_variant():
    exp = "Final score: max(0.05, 1.0 - 0.40) = 0.60"
    assert _rederive_score_from_explanation(exp) == 0.60


def test_rederive_score_clamps_to_minimum():
    exp = "Final score: max(0.05, 1.0 − 1.50) = 0.05"
    score = _rederive_score_from_explanation(exp)
    assert score == pytest.approx(0.05)


def test_rederive_score_returns_none_on_no_match():
    assert _rederive_score_from_explanation("No DEDUCTION SUMMARY here.") is None


def test_rederive_score_returns_none_on_empty():
    assert _rederive_score_from_explanation("") is None


# ── _infer_confidence ─────────────────────────────────────────────────────────


def test_infer_confidence_high_very_low():
    assert _infer_confidence(0.20) == "HIGH"


def test_infer_confidence_high_very_high():
    assert _infer_confidence(0.95) == "HIGH"


def test_infer_confidence_medium_lower():
    assert _infer_confidence(0.50) == "MEDIUM"


def test_infer_confidence_medium_upper():
    assert _infer_confidence(0.72) == "MEDIUM"


def test_infer_confidence_low_borderline():
    assert _infer_confidence(0.62) == "LOW"


def test_infer_confidence_none_returns_low():
    assert _infer_confidence(None) == "LOW"


# ── parse_judgment_from_text ──────────────────────────────────────────────────


def test_parse_judgment_valid_fenced_json():
    text = (
        "```json\n"
        '{"score": 0.80, "valid": true, "explanation": "All good.", '
        '"overall_verdict": "PASS", "confidence": "HIGH", '
        '"claim_checks": [], "criteria_verdicts": [], "improvements_required": []}\n'
        "```"
    )
    result = parse_judgment_from_text(text)
    assert result["score"] == pytest.approx(0.80)
    assert result["valid"] is True
    assert result["overall_verdict"] == "PASS"
    assert result["confidence"] == "HIGH"


def test_parse_judgment_falls_back_on_no_json():
    result = parse_judgment_from_text("no JSON here at all")
    assert result["valid"] is None
    assert result["score"] is None
    assert result["confidence"] == "LOW"


def test_parse_judgment_arithmetic_overrides_json_score():
    # JSON score says 0.80 but deduction arithmetic says 0.75 — arithmetic wins
    text = (
        '{"score": 0.80, "explanation": '
        '"Final score: max(0.05, 1.0 − 0.25) = 0.75"}'
    )
    result = parse_judgment_from_text(text)
    assert result["score"] == pytest.approx(0.75)


def test_parse_judgment_infers_overall_verdict_from_score():
    text = '{"score": 0.50, "explanation": ""}'
    result = parse_judgment_from_text(text)
    assert result["overall_verdict"] == "FAIL"


def test_parse_judgment_claim_checks_extracted():
    text = (
        '{"score": 0.80, "explanation": "ok", '
        '"claim_checks": [{"claim": "test claim", "status": "verified"}]}'
    )
    result = parse_judgment_from_text(text)
    assert len(result["claim_checks"]) == 1
    assert result["claim_checks"][0]["status"] == "verified"


# ── _summarize_tool_result ────────────────────────────────────────────────────


def test_summarize_search_web_with_results():
    result_str = json.dumps(
        [{"title": "Test Title", "url": "http://example.com", "snippet": "A snippet."}]
    )
    summary = _summarize_tool_result("search_web", {"query": "test query"}, result_str)
    assert "search_web" in summary
    assert "1 result" in summary
    assert "Test Title" in summary


def test_summarize_search_web_no_results():
    summary = _summarize_tool_result("search_web", {"query": "empty"}, json.dumps([]))
    assert "no results" in summary.lower()


def test_summarize_check_url_validity_valid():
    result_str = json.dumps({"valid": True, "status_code": 200})
    summary = _summarize_tool_result(
        "check_url_validity", {"url": "http://example.com"}, result_str
    )
    assert "✓ valid" in summary


def test_summarize_check_url_validity_broken():
    result_str = json.dumps({"valid": False, "status_code": 404})
    summary = _summarize_tool_result(
        "check_url_validity", {"url": "http://broken.com"}, result_str
    )
    assert "✗ BROKEN" in summary


def test_summarize_check_acronym():
    result_str = json.dumps({"verdict_hint": "likely_correct", "match_score": 0.8})
    summary = _summarize_tool_result(
        "check_acronym",
        {"acronym": "WHO", "claimed_expansion": "World Health Organization"},
        result_str,
    )
    assert "likely_correct" in summary
    assert "WHO" in summary


def test_summarize_fetch_url_returns_content():
    result_str = json.dumps({"url": "http://example.com", "content": "Article text here."})
    summary = _summarize_tool_result("fetch_url", {"url": "http://example.com"}, result_str)
    assert "Article text here." in summary
