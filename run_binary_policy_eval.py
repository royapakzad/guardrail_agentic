"""
run_binary_policy_eval.py
--------------------------
CLI wrapper that runs the financial and cybersecurity "short" scenario sets
against their binary (COMPLIANT / NOT_FULLY_COMPLIANT) policy files, using
the existing agentic comparison pipeline
(agentic_guardrails/run_agentic_comparison.py).

Each domain is preconfigured with its matching scenario CSV, binary policy
file, and tool group:

    financial      -> data/financial_scenarios_short.csv
                       config/financial_policy_explicit_en_binary.txt
                       --tool-group financial

    cybersecurity  -> data/cybersecurity_phishing_scenarios_short.csv
                       config/policy_cybersecurity_explicit_binary.txt
                       --tool-group cybersecurity

Usage
~~~~~
  # Run one domain with sensible defaults:
  python run_binary_policy_eval.py --domain financial

  # Run both domains back to back:
  python run_binary_policy_eval.py --domain both

  # Override the assistant/judge models:
  python run_binary_policy_eval.py --domain cybersecurity \\
    --provider openai --model gpt-5-mini \\
    --guardrail-judges openai:gpt-5-nano anthropic:claude-sonnet-4-6

  # Preview the underlying command without running it:
  python run_binary_policy_eval.py --domain financial --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, List

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(REPO_ROOT, "agentic_guardrails", "run_agentic_comparison.py")

DOMAIN_PRESETS: Dict[str, Dict[str, str]] = {
    "financial": {
        "input": os.path.join("data", "financial_scenarios_short.csv"),
        "policy_file": os.path.join("config", "financial_policy_explicit_en_binary.txt"),
        "tool_group": "financial",
        "output_prefix": os.path.join("outputs", "financial_binary_run"),
    },
    "cybersecurity": {
        "input": os.path.join("data", "cybersecurity_phishing_scenarios_short.csv"),
        "policy_file": os.path.join("config", "policy_cybersecurity_explicit_binary.txt"),
        "tool_group": "cybersecurity",
        "output_prefix": os.path.join("outputs", "cybersecurity_binary_run"),
    },
}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the financial and/or cybersecurity short scenario sets "
            "through the binary (COMPLIANT / NOT_FULLY_COMPLIANT) policy "
            "files via run_agentic_comparison.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--domain",
        required=True,
        choices=["financial", "cybersecurity", "both"],
        help="Which preconfigured scenario set + binary policy file to run.",
    )

    # ---- Assistant LLM -------------------------------------------------
    p.add_argument("--provider", default="openai", help="Assistant LLM provider (default: openai).")
    p.add_argument("--model", default="gpt-5-mini", help="Assistant model (default: gpt-5-mini).")
    p.add_argument(
        "--assistant-system-prompt-file",
        default=os.path.join("config", "assistant_system_prompt.txt"),
        help="Assistant system prompt file (default: config/assistant_system_prompt.txt).",
    )

    # ---- Guardrail backend + judges ------------------------------------
    p.add_argument(
        "--guardrail",
        default="anyllm",
        choices=["flowjudge", "glider", "anyllm"],
        help="Guardrail backend for the non-agentic path (default: anyllm).",
    )
    p.add_argument(
        "--guardrail-judges",
        nargs="+",
        metavar="PROVIDER:MODEL",
        default=None,
        help="One or more 'provider:model' judges (anyllm only). "
        "Example: openai:gpt-5-nano anthropic:claude-sonnet-4-6",
    )
    p.add_argument("--guardrail-provider", default=None, help="Single-judge mode provider override.")
    p.add_argument("--guardrail-model", default="gpt-5-mini", help="Single-judge mode model override.")

    # ---- Policy / rubric -------------------------------------------------
    p.add_argument(
        "--rubric-file",
        default=os.path.join("config", "rubric.txt"),
        help="Scoring rubric file (default: config/rubric.txt).",
    )

    # ---- Agentic settings ------------------------------------------------
    p.add_argument(
        "--web-search-tool",
        default="duckduckgo",
        choices=["duckduckgo", "searxng", "tavily"],
        help="Web search backend for the agentic guardrail (default: duckduckgo).",
    )
    p.add_argument("--max-tool-calls", type=int, default=8, help="Max tool calls per agentic evaluation (default: 8).")
    p.add_argument(
        "--specialized-tool-reserve",
        type=int,
        default=2,
        help=(
            "Reserve the last N tool calls of the budget for this domain's specialized "
            "tools only, once generic tools would otherwise consume the whole budget "
            "(default: 2). Set to 0 to disable."
        ),
    )
    p.add_argument("--verbose", action="store_true", help="Print tool calls and search results in real time.")

    # ---- Output ------------------------------------------------------------
    p.add_argument(
        "--output-prefix",
        default=None,
        help="Override the output prefix. Ignored when --domain both is used "
        "(each domain keeps its own preset prefix).",
    )
    p.add_argument("--log-dir", default=None, help="Per-scenario log directory override.")

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the underlying run_agentic_comparison.py command instead of running it.",
    )
    return p


def build_command(domain: str, args: argparse.Namespace) -> List[str]:
    preset = DOMAIN_PRESETS[domain]
    output_prefix = args.output_prefix if (args.output_prefix and args.domain != "both") else preset["output_prefix"]

    cmd = [
        sys.executable,
        RUNNER,
        "--input", preset["input"],
        "--output-prefix", output_prefix,
        "--guardrail", args.guardrail,
        "--provider", args.provider,
        "--model", args.model,
        "--assistant-system-prompt-file", args.assistant_system_prompt_file,
        "--policy-files", preset["policy_file"],
        "--rubric-file", args.rubric_file,
        "--tool-group", preset["tool_group"],
        "--web-search-tool", args.web_search_tool,
        "--max-tool-calls", str(args.max_tool_calls),
        "--specialized-tool-reserve", str(args.specialized_tool_reserve),
    ]

    if args.guardrail_judges:
        cmd += ["--guardrail-judges", *args.guardrail_judges]
    else:
        if args.guardrail_provider:
            cmd += ["--guardrail-provider", args.guardrail_provider]
        cmd += ["--guardrail-model", args.guardrail_model]

    if args.log_dir:
        cmd += ["--log-dir", args.log_dir]
    if args.verbose:
        cmd += ["--verbose"]

    return cmd


def main() -> None:
    args = build_arg_parser().parse_args()
    domains = ["financial", "cybersecurity"] if args.domain == "both" else [args.domain]

    for domain in domains:
        cmd = build_command(domain, args)
        print(f"\n=== {domain} ===")
        print(" ".join(cmd))

        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
