"""
scenario_logger.py
------------------
Per-scenario observability logger for the agentic guardrail pipeline.

For each scenario processed by run_agentic_comparison.py this logger
produces two files in the log directory:

  <log_dir>/<scenario_id>_<policy_label>.txt   — human-readable step log
  <log_dir>/<scenario_id>_<policy_label>.json  — machine-readable full log

The .txt file records every step in sequence:
  1. Response generation  (assistant system prompt → scenario → response)
  2. Non-agentic guardrail  (input text + output verdict)
  3. Agentic guardrail      (system prompt, user message, every tool call
                             and its full result, final raw reasoning, output)
  4. Comparison summary

Usage (internal):
    logger = ScenarioLogger(log_dir="logs/", scenario_id="1", language="en")
    logger.log_response_generation(...)
    logger.log_nonagentic_eval(...)
    logger.begin_agentic_eval(policy_label, ...)
    logger.log_tool_call(...)          # called once per tool invocation
    logger.log_agentic_final(...)
    logger.log_comparison(...)
    logger.finalize()                  # flushes JSON file; txt is written live
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Optional

# Width for separator lines in the text log
_W = 100


def _slug(text: str) -> str:
    """Convert arbitrary text to a filesystem-safe slug (max 40 chars)."""
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[\s_-]+", "_", text).strip("_")
    return text[:40]


def _box(heading: str, body: str, indent: int = 2) -> str:
    """Return a formatted box block as a string."""
    pad = " " * indent
    top = f"{'─'*4} {heading} {'─'*(_W - 6 - len(heading))}"
    lines = [top]
    for raw_line in body.splitlines():
        lines.append(pad + raw_line if raw_line.strip() else "")
    lines.append("")
    return "\n".join(lines)


def _section(title: str) -> str:
    eq = "═" * _W
    side = (_W - len(title) - 2) // 2
    header = "═" * side + " " + title + " " + "═" * (_W - side - len(title) - 2)
    return f"\n{header}\n"


def _divider(label: str = "") -> str:
    if label:
        side = (_W - len(label) - 2) // 2
        return "─" * side + " " + label + " " + "─" * (_W - side - len(label) - 2)
    return "─" * _W


class ScenarioLogger:
    """
    Writes a full observability log for one scenario.

    The text file is written incrementally (each log_* call appends immediately)
    so a partial log is produced even if the run crashes partway through.
    The JSON file is written in one shot by finalize().
    """

    def __init__(
        self,
        log_dir: str,
        scenario_id: Any,
        scenario_text: str,
        language: str = "",
    ) -> None:
        os.makedirs(log_dir, exist_ok=True)
        safe_id = _slug(str(scenario_id)) or str(scenario_id)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        base_name = f"scenario_{safe_id}_{timestamp}"
        self._txt_path = os.path.join(log_dir, base_name + ".txt")
        self._json_path = os.path.join(log_dir, base_name + ".json")

        self._data: dict[str, Any] = {
            "scenario_id": scenario_id,
            "language": language,
            "timestamp": datetime.now().isoformat(),
            "scenario_text": scenario_text,
        }
        self._current_policy_key: str = ""

        # Write the file header once
        self._write(
            f"{'╔' + '═'*(_W-2) + '╗'}\n"
            f"{'║'} {'AGENTIC GUARDRAIL SCENARIO LOG':^{_W-4}} {'║'}\n"
            f"{'║'} {f'Scenario ID: {scenario_id}  |  Language: {language}  |  {timestamp}':^{_W-4}} {'║'}\n"
            f"{'╚' + '═'*(_W-2) + '╝'}\n"
        )
        self._write(_box("SCENARIO INPUT", scenario_text))

    # ── internal helpers ──────────────────────────────────────────────────────

    def _write(self, text: str) -> None:
        """Append text to the .txt log file."""
        with open(self._txt_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    # ── step 1: response generation ───────────────────────────────────────────

    def log_response_generation(
        self,
        *,
        provider: str,
        model: str,
        system_prompt: str,
        user_message: str,
        assistant_response: str,
    ) -> None:
        """Log the assistant LLM call (input + output)."""
        self._data["response_generation"] = {
            "provider": provider,
            "model": model,
            "system_prompt": system_prompt,
            "user_message": user_message,
            "assistant_response": assistant_response,
        }
        self._write(_section("STEP 1 — RESPONSE GENERATION"))
        self._write(f"  Provider : {provider}  |  Model : {model}\n")
        self._write(_box("ASSISTANT SYSTEM PROMPT", system_prompt or "<empty>"))
        self._write(_box("USER INPUT (SCENARIO)", user_message))
        self._write(_box("ASSISTANT RESPONSE", assistant_response))

    # ── step 2a: non-agentic guardrail ────────────────────────────────────────

    def log_nonagentic_eval(
        self,
        *,
        policy_label: str,
        policy_text: str,
        rubric: str,
        eval_input_text: str,
        valid: Optional[bool],
        score: Optional[float],
        explanation: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> None:
        """Log one non-agentic guardrail evaluation pass."""
        key = f"policy_{policy_label}"
        if key not in self._data:
            self._data[key] = {}
        self._data[key]["nonagentic"] = {
            "policy_text": policy_text,
            "rubric": rubric,
            "eval_input_text": eval_input_text,
            "valid": valid,
            "score": score,
            "explanation": explanation,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        self._write(_section(f"STEP 2A — NON-AGENTIC GUARDRAIL  [policy: {policy_label}]"))
        self._write(_box("FULL TEXT SENT TO NON-AGENTIC GUARDRAIL", eval_input_text))
        self._write(
            f"  Output:\n"
            f"    valid       : {valid}\n"
            f"    score       : {score}\n"
        )
        if prompt_tokens is not None:
            self._write(
                f"  Token usage (tiktoken — same tokenizer the model uses):\n"
                f"    prompt tokens (eval_input_text)  : {prompt_tokens:,}\n"
                f"    completion tokens (explanation)  : {(completion_tokens or 0):,}\n"
                f"    total tokens                     : {(total_tokens or 0):,}\n"
            )
        self._write(_box("explanation", explanation, indent=4))

    # ── step 2b: agentic guardrail ────────────────────────────────────────────

    def begin_agentic_eval(
        self,
        *,
        policy_label: str,
        policy_text: str,
        rubric: str,
        guardrail_system_prompt: str,
        guardrail_user_message: str,
        provider: str,
        model: str,
        max_tool_calls: int,
    ) -> None:
        """
        Log the beginning of an agentic guardrail pass.
        Must be called before log_tool_call() and log_agentic_final().
        """
        self._current_policy_key = f"policy_{policy_label}"
        if self._current_policy_key not in self._data:
            self._data[self._current_policy_key] = {}
        self._data[self._current_policy_key]["agentic"] = {
            "provider": provider,
            "model": model,
            "max_tool_calls": max_tool_calls,
            "policy_text": policy_text,
            "rubric": rubric,
            "guardrail_system_prompt": guardrail_system_prompt,
            "guardrail_user_message": guardrail_user_message,
            "tool_calls": [],
        }

        self._write(_section(f"STEP 2B — AGENTIC GUARDRAIL  [policy: {policy_label}]"))
        self._write(f"  Provider : {provider}  |  Model : {model}  |  Max tool calls : {max_tool_calls}\n")
        self._write(
            _box(
                "GUARDRAIL SYSTEM PROMPT  (contains POLICY + RUBRIC + phase instructions)",
                guardrail_system_prompt,
            )
        )
        self._write(_box("GUARDRAIL USER MESSAGE  (conversation to evaluate)", guardrail_user_message))

    def log_tool_call(
        self,
        *,
        call_number: int,
        tool_name: str,
        input_args: dict,
        result_raw: str,
    ) -> None:
        """Log one tool call and its full result."""
        try:
            result_parsed = json.loads(result_raw)
        except (json.JSONDecodeError, ValueError):
            result_parsed = result_raw

        entry = {
            "call_number": call_number,
            "tool": tool_name,
            "input": input_args,
            "result": result_parsed,
        }
        agentic = self._data.get(self._current_policy_key, {}).get("agentic")
        if agentic is not None:
            agentic["tool_calls"].append(entry)

        # Human-readable rendering of the result
        result_display = json.dumps(result_parsed, indent=2, ensure_ascii=False, default=str)

        self._write(_divider(f"Tool Call #{call_number} — {tool_name}"))
        self._write(f"  INPUT  : {json.dumps(input_args, ensure_ascii=False)}")
        self._write(_box("FULL RESULT", result_display, indent=4))

    def log_agentic_final(
        self,
        *,
        raw_final_message: str,
        valid: Optional[bool],
        score: Optional[float],
        explanation: str,
        tool_calls_made: int,
        sources_used: list,
        url_checks: list,
        prompt_tokens_total: Optional[int] = None,
        completion_tokens_total: Optional[int] = None,
        total_tokens_used: Optional[int] = None,
        peak_prompt_tokens: Optional[int] = None,
        token_usage_per_turn: Optional[list] = None,
    ) -> None:
        """Log the agentic guardrail's final reasoning and parsed verdict."""
        agentic = self._data.get(self._current_policy_key, {}).get("agentic")
        if agentic is not None:
            agentic.update(
                {
                    "raw_final_message": raw_final_message,
                    "valid": valid,
                    "score": score,
                    "explanation": explanation,
                    "tool_calls_made": tool_calls_made,
                    "sources_used": sources_used,
                    "url_checks": url_checks,
                    "prompt_tokens_total": prompt_tokens_total,
                    "completion_tokens_total": completion_tokens_total,
                    "total_tokens_used": total_tokens_used,
                    "peak_prompt_tokens": peak_prompt_tokens,
                    "token_usage_per_turn": token_usage_per_turn or [],
                }
            )

        self._write(_box("GUARDRAIL FINAL REASONING  (raw LLM output)", raw_final_message))
        self._write(
            f"  Parsed output:\n"
            f"    valid           : {valid}\n"
            f"    score           : {score}\n"
            f"    tool_calls_made : {tool_calls_made}\n"
        )
        self._write(_box("explanation", explanation, indent=4))

        # Token usage summary
        if prompt_tokens_total is not None or completion_tokens_total is not None:
            self._write(f"  Token usage (exact — from provider API, summed across all {len(token_usage_per_turn or [])} turns):\n")
            self._write(f"    prompt tokens (input read)   : {(prompt_tokens_total or 0):,}")
            self._write(f"    completion tokens (generated): {(completion_tokens_total or 0):,}")
            self._write(f"    total tokens                 : {(total_tokens_used or 0):,}")
            self._write(f"    peak prompt tokens (any turn): {(peak_prompt_tokens or 0):,}  ← context window high-water mark")
            if token_usage_per_turn:
                self._write(f"\n  Per-turn token breakdown:")
                self._write(f"    {'Turn':>4}  {'Prompt':>8}  {'Completion':>10}  {'Total':>8}  Tool calls?")
                self._write(f"    {'─'*4}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*10}")
                for t in token_usage_per_turn:
                    has_tc = "yes" if t.get("has_tool_calls") else "no"
                    self._write(
                        f"    {t['turn']:>4}  {t['prompt_tokens']:>8,}  "
                        f"{t['completion_tokens']:>10,}  {t['total_tokens']:>8,}  {has_tc}"
                    )
            self._write("")
        else:
            self._write("  Token usage: not available (provider did not return usage metadata)\n")

        if sources_used:
            self._write(f"  Sources used ({len(sources_used)}):\n")
            for s in sources_used:
                self._write(f"    • {s}")
            self._write("")

        if url_checks:
            self._write(f"  URL checks ({len(url_checks)}):\n")
            for uc in url_checks:
                icon = "✓" if uc.get("valid") else "✗"
                self._write(
                    f"    {icon} {uc.get('url','')}  →  "
                    f"HTTP {uc.get('status_code','?')}  "
                    f"valid={uc.get('valid')}  "
                    f"redirects={uc.get('redirect_count', 0)}"
                )
            self._write("")
        else:
            self._write("  URL checks: none\n")

    # ── step 3: comparison summary ────────────────────────────────────────────

    def log_comparison(
        self,
        *,
        policy_label: str,
        nonagentic_valid: Optional[bool],
        nonagentic_score: Optional[float],
        agentic_valid: Optional[bool],
        agentic_score: Optional[float],
        score_delta: Optional[float],
        judgment_changed: Optional[bool],
        agentic_used_tools: bool,
    ) -> None:
        """Log the side-by-side comparison between the two paths."""
        key = f"policy_{policy_label}"
        if key not in self._data:
            self._data[key] = {}
        self._data[key]["comparison"] = {
            "nonagentic_valid": nonagentic_valid,
            "nonagentic_score": nonagentic_score,
            "agentic_valid": agentic_valid,
            "agentic_score": agentic_score,
            "score_delta": score_delta,
            "judgment_changed": judgment_changed,
            "agentic_used_tools": agentic_used_tools,
        }

        delta_str = f"{score_delta:+.4f}" if score_delta is not None else "n/a"
        self._write(_section(f"COMPARISON SUMMARY  [policy: {policy_label}]"))
        col_w = 30
        sep = "─" * col_w
        sep2 = "─" * 15
        self._write(
            f"\n"
            f"  {'':30s}  {'NON-AGENTIC':>15s}    {'AGENTIC':>15s}\n"
            f"  {sep}  {sep2}    {sep2}\n"
            f"  {'valid':30s}  {str(nonagentic_valid):>15s}    {str(agentic_valid):>15s}\n"
            f"  {'score':30s}  {str(nonagentic_score):>15s}    {str(agentic_score):>15s}\n"
            f"  {'score delta (agentic − non-agentic)':30s}  {'':>15s}    {delta_str:>15s}\n"
            f"  {'judgment changed?':30s}  {'':>15s}    {str(judgment_changed):>15s}\n"
            f"  {'agentic used tools?':30s}  {'':>15s}    {str(agentic_used_tools):>15s}\n"
        )

    # ── finalize ──────────────────────────────────────────────────────────────

    def finalize(self) -> tuple[str, str]:
        """
        Write the JSON log file and close the text log with a footer.
        Returns (txt_path, json_path).
        """
        self._write("═" * _W)
        self._write(f"  Log written : {self._txt_path}")
        self._write(f"  JSON written: {self._json_path}")
        self._write("═" * _W + "\n")

        with open(self._json_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2, default=str)

        return self._txt_path, self._json_path

    @property
    def txt_path(self) -> str:
        return self._txt_path

    @property
    def json_path(self) -> str:
        return self._json_path
