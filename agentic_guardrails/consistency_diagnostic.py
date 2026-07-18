"""
consistency_diagnostic.py
--------------------------
Run the SAME frozen scenario + assistant response through the SAME
guardrail judge N times, to measure how much scoring varies purely from
repeated judge-side sampling and tool-evidence gathering -- independent of
any variance from assistant-response generation (Issue #50).

Domain-agnostic by construction: it just calls run_split_criteria_guardrail(),
the same function the main pipeline (run_agentic_comparison.py) uses, so it
works unmodified for humanitarian, financial, cybersecurity, or any future
policy/tool-group -- whatever is passed via --policy-file/--tool-group.

Usage:
  python consistency_diagnostic.py \\
    --frozen-input ../data/consistency_fixtures/cybersecurity_frozen.json \\
    --policy-file ../config/policy_cybersecurity.txt \\
    --rubric-file ../config/rubric.txt \\
    --tool-group cybersecurity \\
    --provider anthropic --guardrail-model claude-sonnet-4-6 \\
    --n-runs 6 \\
    --output-prefix ../outputs/consistency_cybersecurity

The --frozen-input file must contain {"scenario": "...", "assistant_response":
"..."} -- generate it once (e.g. from a real pipeline run's output JSON) and
reuse it across every diagnostic run, so only the JUDGE varies between runs,
not the assistant response being judged.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from dotenv import load_dotenv

# Load .env from the repo root regardless of the caller's cwd or where this
# script happens to live relative to it.
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import tools as _tools_mod
from guardrails_runner import create_guardrail
from agentic_runner import run_split_criteria_guardrail
from reliability_metrics import (
    build_reliability_report,
    compare_collapse_schemes,
    compare_tool_call_outputs,
    format_evidence_reproducibility_report,
)


def _load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated identical judge calls on a frozen scenario/response "
            "to measure scoring reliability (Issue #50). Works for any domain "
            "-- pass the policy/tool-group for humanitarian, financial, "
            "cybersecurity, or a future use case."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--frozen-input",
        required=True,
        help="JSON file with {'scenario': ..., 'assistant_response': ...}. "
        "Held constant across all N runs so only judge-side variance is measured.",
    )
    parser.add_argument("--policy-file", required=True)
    parser.add_argument("--rubric-file", required=True)
    parser.add_argument(
        "--tool-group",
        default="default",
        help="Must match a key in tools.TOOL_GROUPS, e.g. humanitarian/financial/cybersecurity.",
    )
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--guardrail-model", default="claude-sonnet-4-6")
    parser.add_argument("--n-runs", type=int, default=6)
    parser.add_argument("--max-tool-calls", type=int, default=12)
    parser.add_argument(
        "--web-search-tool",
        default="tavily",
        choices=["duckduckgo", "searxng", "tavily"],
        help=(
            "Web search backend for search_web (default: tavily, unlike the main "
            "pipeline's duckduckgo default -- duckduckgo has been observed to stall "
            "for over an hour under rate-limiting in this environment, which is "
            "fatal for a diagnostic that needs N repeated runs to complete cleanly)."
        ),
    )
    parser.add_argument("--output-prefix", required=True)
    return parser


def run_diagnostic(
    *,
    scenario: str,
    assistant_response: str,
    policy_text: str,
    rubric: str,
    provider: str,
    guardrail_model: str,
    tool_group: str,
    n_runs: int,
    max_tool_calls: int,
    web_search_tool: str = "tavily",
) -> list[dict]:
    """Run N identical judge calls; return one result dict per run."""
    _tools_mod.set_search_backend(web_search_tool)
    print(f"Web search backend: {web_search_tool.upper()}")
    model_id = f"{provider}:{guardrail_model}"
    guardrail = create_guardrail("anyllm")

    results = []
    for i in range(1, n_runs + 1):
        print(f"=== Run {i}/{n_runs} ===", flush=True)
        t0 = time.perf_counter()
        na, ag = run_split_criteria_guardrail(
            guardrail=guardrail,
            provider=provider,
            guardrail_model=guardrail_model,
            model_id=model_id,
            policy_text=policy_text,
            rubric=rubric,
            system_prompt="",
            user_message=scenario,
            assistant_response=assistant_response,
            max_tool_calls=max_tool_calls,
            tool_group=tool_group,
            verbose=False,
            logger=None,
            policy_label=f"consistency_run_{i}",
            scenario_language="en",
        )
        elapsed = round(time.perf_counter() - t0, 1)
        tools_called = [c["tool"] for c in ag.tool_call_log]
        print(
            f"  non-agentic: score={na.score} valid={na.valid} | "
            f"agentic: score={ag.score} valid={ag.valid} tool_calls={ag.tool_calls_made} "
            f"tools={tools_called} ({elapsed}s)",
            flush=True,
        )
        results.append(
            {
                "run": i,
                "na_score": na.score,
                "na_valid": na.valid,
                "na_criteria": {cv["criterion"]: cv["verdict"] for cv in na.criteria_verdicts},
                "ag_score": ag.score,
                "ag_valid": ag.valid,
                "ag_criteria": {cv["criterion"]: cv["verdict"] for cv in ag.criteria_verdicts},
                "ag_tools_called": tools_called,
                "ag_tool_calls_made": ag.tool_calls_made,
                # Full tool call log (not just names) -- Issue #54: lets us
                # check whether the EVIDENCE itself (tool inputs/outputs) is
                # reproducible across runs, separately from whether the
                # verdict built on top of it is.
                "ag_tool_call_log": ag.tool_call_log,
            }
        )
    return results


def main() -> None:
    args = build_arg_parser().parse_args()

    with open(args.frozen_input, "r", encoding="utf-8") as f:
        frozen = json.load(f)

    results = run_diagnostic(
        scenario=frozen["scenario"],
        assistant_response=frozen["assistant_response"],
        policy_text=_load_text_file(args.policy_file),
        rubric=_load_text_file(args.rubric_file),
        provider=args.provider,
        guardrail_model=args.guardrail_model,
        tool_group=args.tool_group,
        n_runs=args.n_runs,
        max_tool_calls=args.max_tool_calls,
        web_search_tool=args.web_search_tool,
    )

    out_path = f"{args.output_prefix}_runs.json"
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    na_report = build_reliability_report(
        [r["na_criteria"] for r in results], label="NON-AGENTIC (text-only, no tools)"
    )
    ag_report = build_reliability_report(
        [r["ag_criteria"] for r in results], label="AGENTIC (tool-verified)"
    )

    # Issue #54: does coarsening the verdict scale actually reduce measured
    # instability, and is the underlying tool evidence itself reproducible?
    na_collapse = compare_collapse_schemes([r["na_criteria"] for r in results], label="non-agentic")
    ag_collapse = compare_collapse_schemes([r["ag_criteria"] for r in results], label="agentic")
    collapse_lines = ["=== Verdict scale collapse comparison (Issue #54) ==="]
    for path_label, comparison in [("non-agentic", na_collapse), ("agentic", ag_collapse)]:
        collapse_lines.append(f"  {path_label}:")
        for scheme, stats in comparison.items():
            kappa_str = f"{stats['kappa']:.4f}" if stats["kappa"] is not None else "undefined"
            collapse_lines.append(
                f"    {scheme:10s} kappa={kappa_str}  "
                f"unstable={stats['n_unstable_criteria']}/{stats['n_criteria']}"
            )
    collapse_text = "\n".join(collapse_lines)

    evidence_entries = compare_tool_call_outputs([r["ag_tool_call_log"] for r in results])
    evidence_text = format_evidence_reproducibility_report(evidence_entries)

    report_text = "\n\n".join(
        [na_report.format_report(), ag_report.format_report(), collapse_text, evidence_text]
    )
    print("\n" + report_text)

    report_path = f"{args.output_prefix}_reliability_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    print(f"\nRaw runs written to {out_path}")
    print(f"Reliability report written to {report_path}")


if __name__ == "__main__":
    main()
