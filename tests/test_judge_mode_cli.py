"""Tests for the --judge-granularity / --judge-mode CLI surface and the
frozen-non-agentic reuse path (Issue #91) in run_agentic_comparison.py.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agentic_guardrails"))

import pytest

import run_agentic_comparison as rac


# ── argparse surface ────────────────────────────────────────────────────────

def _parse(argv):
    parser = rac.build_arg_parser()
    return parser.parse_args(argv)


BASE_ARGS = [
    "--input", "in.csv",
    "--output-prefix", "out/run1",
    "--policy-files", "policy.txt",
]


def test_judge_granularity_defaults_to_split():
    args = _parse(BASE_ARGS)
    assert args.judge_granularity == "split"


def test_judge_mode_defaults_to_both():
    args = _parse(BASE_ARGS)
    assert args.judge_mode == "both"


def test_judge_granularity_accepts_full_policy():
    args = _parse(BASE_ARGS + ["--judge-granularity", "full-policy"])
    assert args.judge_granularity == "full-policy"


def test_judge_mode_accepts_agentic_only_and_non_agentic_only():
    args = _parse(BASE_ARGS + ["--judge-mode", "agentic-only"])
    assert args.judge_mode == "agentic-only"
    args = _parse(BASE_ARGS + ["--judge-mode", "non-agentic-only"])
    assert args.judge_mode == "non-agentic-only"


def test_judge_granularity_rejects_unknown_value():
    with pytest.raises(SystemExit):
        _parse(BASE_ARGS + ["--judge-granularity", "per-criterion"])


def test_judge_mode_rejects_unknown_value():
    with pytest.raises(SystemExit):
        _parse(BASE_ARGS + ["--judge-mode", "agentic-and-a-bit"])


# ── _parse_frozen_nonagentic ─────────────────────────────────────────────────
# row values here are plain strings, matching what csv.DictReader actually
# hands back for every column, including a prior run's output file.

FROZEN_ROW = {
    "id": "IR01",
    "policy_claude_opus_4_6_nonagentic_valid": "True",
    "policy_claude_opus_4_6_nonagentic_score": "0.8",
    "policy_claude_opus_4_6_nonagentic_explanation": "1. A: fine\n→ Verdict: compliant",
    "policy_claude_opus_4_6_nonagentic_criteria_verdicts": (
        '[{"criterion": "A", "verdict": "COMPLIANT"}]'
    ),
    "policy_claude_opus_4_6_nonagentic_improvements": "[]",
}


def test_parse_frozen_nonagentic_reconstructs_judgment():
    na = rac._parse_frozen_nonagentic(FROZEN_ROW, "policy_claude_opus_4_6")
    assert na.valid is True
    assert na.score == 0.8
    assert na.criteria_verdicts == [{"criterion": "A", "verdict": "COMPLIANT"}]
    assert na.improvements == []


def test_parse_frozen_nonagentic_fails_fast_when_score_missing():
    row = {"id": "IR02"}  # no frozen columns at all for this base
    with pytest.raises(ValueError, match="agentic-only"):
        rac._parse_frozen_nonagentic(row, "policy_claude_opus_4_6")


def test_parse_frozen_nonagentic_handles_false_valid_and_empty_lists():
    row = {
        "id": "IR03",
        "policy_x_nonagentic_valid": "False",
        "policy_x_nonagentic_score": "0.2",
        "policy_x_nonagentic_explanation": "",
        "policy_x_nonagentic_criteria_verdicts": "",
        "policy_x_nonagentic_improvements": "",
    }
    na = rac._parse_frozen_nonagentic(row, "policy_x")
    assert na.valid is False
    assert na.score == 0.2
    assert na.criteria_verdicts == []
    assert na.improvements == []
