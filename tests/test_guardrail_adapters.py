"""Tests for GuardrailAdapter protocol and concrete adapters (PR #11)."""

from unittest.mock import MagicMock, patch

import pytest
from guardrails_runner import (
    AnyLlmAdapter,
    FlowJudgeAdapter,
    GliderAdapter,
    GuardrailAdapter,
    NonAgenticJudgment,
)


def test_anyllm_adapter_backend_name():
    assert AnyLlmAdapter().backend_name == "anyllm"


def test_flowjudge_adapter_backend_name():
    adapter = FlowJudgeAdapter(MagicMock())
    assert adapter.backend_name == "flowjudge"


def test_glider_adapter_backend_name():
    adapter = GliderAdapter(MagicMock())
    assert adapter.backend_name == "glider"


def test_anyllm_adapter_is_protocol_compliant():
    adapter = AnyLlmAdapter()
    assert isinstance(adapter, GuardrailAdapter)


def test_flowjudge_adapter_is_protocol_compliant():
    adapter = FlowJudgeAdapter(MagicMock())
    assert isinstance(adapter, GuardrailAdapter)


def test_glider_adapter_is_protocol_compliant():
    adapter = GliderAdapter(MagicMock())
    assert isinstance(adapter, GuardrailAdapter)


def test_flowjudge_adapter_evaluate_returns_nonagentic_judgment():
    mock_guardrail = MagicMock()
    mock_guardrail.validate.return_value = MagicMock(score=0.8, explanation="Looks good.")
    adapter = FlowJudgeAdapter(mock_guardrail)
    result = adapter.evaluate("eval text", "policy", assistant_response="response")
    assert isinstance(result, NonAgenticJudgment)
    assert result.score == pytest.approx(0.8)
    assert result.valid is True


def test_glider_adapter_evaluate_returns_nonagentic_judgment():
    mock_guardrail = MagicMock()
    mock_guardrail.validate.return_value = MagicMock(score=0.5, explanation="Issues found.")
    adapter = GliderAdapter(mock_guardrail)
    result = adapter.evaluate("eval text", "policy")
    assert isinstance(result, NonAgenticJudgment)
    assert result.score == pytest.approx(0.5)
    assert result.valid is False


def test_anyllm_adapter_evaluate_calls_generative_judge():
    fake_judgment = NonAgenticJudgment(valid=True, score=0.85, explanation="ok")
    with patch("guardrails_runner._run_generative_judge", return_value=fake_judgment) as mock_judge:
        adapter = AnyLlmAdapter()
        result = adapter.evaluate("eval text", "policy", model_id="openai:gpt-5-mini")
    mock_judge.assert_called_once()
    assert result.score == pytest.approx(0.85)


def test_anyllm_adapter_evaluate_raises_without_model_id():
    adapter = AnyLlmAdapter()
    with pytest.raises(ValueError, match="model_id"):
        adapter.evaluate("eval text", "policy")
