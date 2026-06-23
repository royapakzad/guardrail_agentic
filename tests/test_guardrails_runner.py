"""Tests for guardrails_runner pure-logic helpers (network-free)."""

import pytest
from guardrails_runner import _extract_first_json_object, _rederive_score_from_explanation

# ── _extract_first_json_object ────────────────────────────────────────────────


def test_extract_json_fenced_block():
    text = '```json\n{"score": 0.80, "valid": true}\n```'
    result = _extract_first_json_object(text)
    assert result is not None
    assert result["score"] == 0.80


def test_extract_json_bare_object():
    text = 'some prefix {"score": 0.70} suffix'
    result = _extract_first_json_object(text)
    assert result is not None
    assert result.get("score") == 0.70


def test_extract_json_prefers_fenced_block_over_bare():
    # Fenced block has different score than bare — should pick fenced
    text = '```json\n{"score": 0.90}\n```\n{"score": 0.10}'
    result = _extract_first_json_object(text)
    assert result is not None
    assert result["score"] == 0.90


def test_extract_json_returns_none_on_no_json():
    assert _extract_first_json_object("no json here") is None


def test_extract_json_returns_none_on_empty():
    assert _extract_first_json_object("") is None


def test_extract_json_invalid_json_in_braces():
    result = _extract_first_json_object("{not valid json}")
    assert result is None


def test_extract_json_nested_object():
    text = '{"outer": {"inner": 42}}'
    result = _extract_first_json_object(text)
    assert result is not None
    assert result["outer"]["inner"] == 42


# ── _rederive_score_from_explanation ─────────────────────────────────────────


def test_rederive_score_standard():
    exp = "Final score: max(0.05, 1.0 − 0.30) = 0.70"
    assert _rederive_score_from_explanation(exp) == pytest.approx(0.70)


def test_rederive_score_hyphen():
    exp = "Final score: max(0.05, 1.0 - 0.40) = 0.60"
    assert _rederive_score_from_explanation(exp) == pytest.approx(0.60)


def test_rederive_score_minimum_clamp():
    exp = "Final score: max(0.05, 1.0 − 1.20) = 0.05"
    assert _rederive_score_from_explanation(exp) == pytest.approx(0.05)


def test_rederive_score_no_match():
    assert _rederive_score_from_explanation("No arithmetic found.") is None


def test_rederive_score_empty():
    assert _rederive_score_from_explanation("") is None
