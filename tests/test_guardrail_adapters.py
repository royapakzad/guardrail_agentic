"""Tests for the guardrail backend adapters (PR-4).

No network or model loading: a fake guardrail stands in for the any-guardrail
library object, and the Anthropic/Gemini fallback is monkeypatched.
"""

import guardrails_runner as gr
import pytest
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


@pytest.mark.parametrize(
    "model_id",
    ["openai:gpt-5-nano", "anthropic:claude-sonnet-4-6", "gemini:gemini-2.5-flash"],
)
def test_anyllm_adapter_routes_every_provider_to_generative_judge(monkeypatch, model_id):
    # The library's AnyLlm.validate() coerces score to int, so all providers go
    # through the repo's prompt-based generative judge (float scores).
    sentinel = gr.NonAgenticJudgment(valid=True, score=0.72, explanation="judged")
    captured = {}

    def fake_judge(*, model_id, policy_text, eval_text):
        captured["model_id"] = model_id
        return sentinel

    monkeypatch.setattr(gr, "_run_generative_judge", fake_judge)
    judgment = AnyLlmAdapter().evaluate(eval_text="t", policy_text="p", assistant_response="r", model_id=model_id)
    assert judgment is sentinel
    assert captured["model_id"] == model_id


def test_anyllm_adapter_defaults_model_when_unspecified(monkeypatch):
    captured = {}

    def fake_judge(*, model_id, policy_text, eval_text):
        captured["model_id"] = model_id
        return gr.NonAgenticJudgment(valid=None, score=None, explanation="")

    monkeypatch.setattr(gr, "_run_generative_judge", fake_judge)
    AnyLlmAdapter().evaluate(eval_text="t", policy_text="p", assistant_response="r")
    assert captured["model_id"] == gr._DEFAULT_GENERATIVE_JUDGE_MODEL


def test_glider_and_flowjudge_adapters_wrap_score():
    glider = GliderAdapter(_FakeGuardrail(score=0.4)).evaluate(eval_text="t", policy_text="p", assistant_response="r")
    assert glider.score == 0.4
    assert glider.valid is False

    flow = FlowJudgeAdapter(_FakeGuardrail(score=0.7)).evaluate(eval_text="t", policy_text="p", assistant_response="r")
    assert flow.score == 0.7
    assert flow.valid is True
