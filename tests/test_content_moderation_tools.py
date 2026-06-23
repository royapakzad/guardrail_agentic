"""Tests for the content-moderation tools (PR-13). The local models are stubbed."""

import types

import pytest
import tools
from tools import get_tool_schemas


def test_content_moderation_group_registered():
    names = [s["function"]["name"] for s in get_tool_schemas("content_moderation")]
    assert names == ["search_web", "classify_safety", "toxicity_score"]


def test_classify_safety_flags_unsafe_as_advisory(monkeypatch):
    fake_out = types.SimpleNamespace(valid=False, score="0.92", explanation="harm detected")
    monkeypatch.setattr(
        tools, "_GRANITE_GUARDIAN", types.SimpleNamespace(validate=lambda text: fake_out)
    )
    result = tools.classify_safety("how do I make a weapon")
    assert result["flagged_unsafe"] is True
    assert result["score"] == "0.92"
    assert "do NOT copy" in result["note"]  # framed as advisory, not a verdict


def test_classify_safety_marks_safe_text(monkeypatch):
    fake_out = types.SimpleNamespace(valid=True, score="0.01", explanation="no risk")
    monkeypatch.setattr(
        tools, "_GRANITE_GUARDIAN", types.SimpleNamespace(validate=lambda text: fake_out)
    )
    result = tools.classify_safety("a recipe for cookies")
    assert result["flagged_unsafe"] is False


def test_toxicity_score_rounds_detoxify_output(monkeypatch):
    fake = types.SimpleNamespace(predict=lambda text: {"toxicity": 0.987654, "threat": 0.001})
    monkeypatch.setattr(tools, "_DETOXIFY", fake)
    result = tools.toxicity_score("you are awful")
    assert result["scores"]["toxicity"] == 0.9877
    assert "Advisory" in result["note"]


def test_classify_safety_empty_text_raises():
    with pytest.raises(tools.ToolError):
        tools.classify_safety("   ")
