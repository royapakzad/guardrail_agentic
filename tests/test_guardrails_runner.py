"""Tests for the non-agentic guardrail runner's pure helpers."""

from guardrails_runner import (
    NONAGENTIC_VALID_THRESHOLD,
    _extract_first_json_object,
    _rederive_score_from_explanation,
)


def test_extract_first_json_object_prefers_fenced_block():
    text = 'prose ```json\n{"score": 0.9, "valid": true}\n``` more prose'
    obj = _extract_first_json_object(text)
    assert obj == {"score": 0.9, "valid": True}


def test_extract_first_json_object_walks_unfenced_braces():
    text = 'no fence here, just {"score": 0.5} inline'
    assert _extract_first_json_object(text) == {"score": 0.5}


def test_extract_first_json_object_skips_invalid_then_finds_valid():
    text = 'broken {not json} then good {"score": 0.7}'
    assert _extract_first_json_object(text) == {"score": 0.7}


def test_extract_first_json_object_returns_none_when_absent():
    assert _extract_first_json_object("no json at all") is None


def test_rederive_score_from_explanation_recomputes():
    expl = "Final score: max(0.05, 1.0 − 0.25) = 0.80"
    assert _rederive_score_from_explanation(expl) == 0.75


def test_threshold_constant_is_06():
    assert NONAGENTIC_VALID_THRESHOLD == 0.6
