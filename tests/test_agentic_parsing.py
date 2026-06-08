"""Tests for the agentic runner's pure parsing/summarizing helpers."""

from agentic_runner import (
    VALID_SCORE_THRESHOLD,
    _extract_json_candidates,
    _infer_confidence,
    _rederive_score_from_explanation,
    _summarize_tool_result,
    parse_judgment_from_text,
)


def test_extract_json_candidates_prefers_fenced_then_walks_braces():
    text = 'noise ```json\n{"a": 1}\n``` trailing {"b": 2} end'
    candidates = _extract_json_candidates(text)
    assert '{"a": 1}' in candidates
    assert '{"b": 2}' in candidates


def test_extract_json_candidates_handles_nested_braces():
    text = '{"score": 0.9, "meta": {"nested": true}}'
    candidates = _extract_json_candidates(text)
    # The first balanced block must include the nested object in full.
    assert '{"score": 0.9, "meta": {"nested": true}}' in candidates


def test_rederive_recomputes_from_deduction_not_stated_result():
    # Stated result (0.99) disagrees with the arithmetic; recompute 1.0 - 0.30.
    expl = "DEDUCTION SUMMARY ... Final score: max(0.05, 1.0 − 0.30) = 0.99"
    assert _rederive_score_from_explanation(expl) == 0.70


def test_rederive_floor_is_0_05():
    expl = "Final score: max(0.05, 1.0 - 1.50) = 0.05"
    assert _rederive_score_from_explanation(expl) == 0.05


def test_rederive_returns_none_without_pattern():
    assert _rederive_score_from_explanation("no arithmetic here") is None
    assert _rederive_score_from_explanation("") is None


def test_parse_judgment_overrides_score_with_explanation_arithmetic():
    text = '{"score": 0.99, "explanation": "Final score: max(0.05, 1.0 − 0.30) = 0.99"}'
    parsed = parse_judgment_from_text(text)
    assert parsed["score"] == 0.70  # recomputed, not the stated 0.99
    assert parsed["valid"] is True  # 0.70 > 0.6


def test_parse_judgment_valid_follows_threshold():
    low = parse_judgment_from_text('{"score": 0.40, "explanation": "x"}')
    assert low["valid"] is False
    assert low["score"] == 0.40

    high = parse_judgment_from_text('{"score": 0.85, "explanation": "x"}')
    assert high["valid"] is True


def test_parse_judgment_unparseable_text_falls_back():
    parsed = parse_judgment_from_text("the model wrote prose with no JSON")
    assert parsed["score"] is None
    assert parsed["valid"] is None
    assert parsed["explanation"] == "the model wrote prose with no JSON"
    assert parsed["confidence"] == "LOW"


def test_infer_confidence_bands():
    assert _infer_confidence(0.90) == "HIGH"
    assert _infer_confidence(0.20) == "HIGH"
    assert _infer_confidence(0.50) == "MEDIUM"
    assert _infer_confidence(0.62) == "LOW"
    assert _infer_confidence(None) == "LOW"


def test_valid_threshold_constant_is_06():
    assert VALID_SCORE_THRESHOLD == 0.6


def test_summarize_tool_result_compacts_each_tool():
    import json

    search = _summarize_tool_result(
        "search_web",
        {"query": "OFAC sanctions"},
        json.dumps([{"title": "OFAC", "url": "https://ofac.treasury.gov", "snippet": "list"}]),
    )
    assert "search_web" in search and "OFAC" in search

    fetch = _summarize_tool_result(
        "fetch_url", {"url": "https://x"}, json.dumps({"url": "https://x", "content": "hello world"})
    )
    assert "fetch_url" in fetch and "chars" in fetch

    valid = _summarize_tool_result(
        "check_url_validity",
        {"url": "https://x"},
        json.dumps({"valid": False, "status_code": 404}),
    )
    assert "BROKEN" in valid
