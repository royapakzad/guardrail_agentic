"""Tests for the guardrail backend adapters (PR-4).

No network or model loading: a fake guardrail stands in for the any-guardrail
library object, and the Anthropic/Gemini fallback is monkeypatched.
"""

import pytest

import guardrails_runner as gr
from guardrails_runner import AnyLlmAdapter, FlowJudgeAdapter, GliderAdapter, create_guardrail


class _FakeRaw:
    def __init__(self, score, explanation="why"):
        self.score = score
        self.explanation = explanation


class _FakeGuardrail:
    """Stand-in for an any-guardrail backend object with a .validate()."""

    def __init__(self, score):
        self._score = score
        self.calls = []

    def validate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _FakeRaw(self._score)


def test_create_guardrail_returns_anyllm_adapter():
    g = create_guardrail("anyllm")
    assert isinstance(g, AnyLlmAdapter)
    assert g.backend_name == "anyllm"


def test_create_guardrail_rejects_unknown_name():
    with pytest.raises(ValueError, match="must be one of"):
        create_guardrail("bogus")


def test_create_guardrail_glider_requires_criteria():
    with pytest.raises(ValueError, match="pass_criteria"):
        create_guardrail("glider")


def test_anyllm_adapter_openai_path_derives_valid_from_score():
    fake = _FakeGuardrail(score=0.82)
    judgment = AnyLlmAdapter(fake).evaluate(
        eval_text="t", policy_text="p", assistant_response="r", model_id="openai:gpt-5-nano"
    )
    assert judgment.score == 0.82
    assert judgment.valid is True
    assert judgment.overall_verdict == "PASS"
    assert fake.calls  # the library path was used (not the fallback)


def test_anyllm_adapter_routes_anthropic_to_fallback(monkeypatch):
    sentinel = gr.NonAgenticJudgment(valid=False, score=0.3, explanation="fallback")
    captured = {}

    def fake_fallback(*, model_id, policy_text, eval_text):
        captured["model_id"] = model_id
        return sentinel

    monkeypatch.setattr(gr, "_run_nonagentic_fallback", fake_fallback)
    fake = _FakeGuardrail(score=0.9)  # must NOT be consulted
    judgment = AnyLlmAdapter(fake).evaluate(
        eval_text="t", policy_text="p", assistant_response="r", model_id="anthropic:claude-sonnet-4-6"
    )
    assert judgment is sentinel
    assert captured["model_id"] == "anthropic:claude-sonnet-4-6"
    assert not fake.calls  # fallback used; library never called


def test_glider_and_flowjudge_adapters_wrap_score():
    glider = GliderAdapter(_FakeGuardrail(score=0.4)).evaluate(eval_text="t", policy_text="p", assistant_response="r")
    assert glider.score == 0.4
    assert glider.valid is False

    flow = FlowJudgeAdapter(_FakeGuardrail(score=0.7)).evaluate(eval_text="t", policy_text="p", assistant_response="r")
    assert flow.score == 0.7
    assert flow.valid is True
