"""
run_agentic_comparison.py
-------------------------
Agentic vs Non-Agentic Guardrail Comparison Pipeline

Research question
~~~~~~~~~~~~~~~~~
Do guardrails make better safety judgments when they can verify factual claims
through retrieval (agentic), compared to guardrails that rely solely on their
built-in knowledge (non-agentic)?

Tested across English (EN), Farsi (FA), and optionally Spanish (ES) using
text-based scenarios in the asylum / humanitarian domain.

Pipeline per scenario row
~~~~~~~~~~~~~~~~~~~~~~~~~~
  1. Call the assistant LLM → initial response
  2a. Non-agentic guardrail  → score + flag (single LLM call, no tools)
  2b. Agentic guardrail      → retrieval + verification → adjusted judgment
                               (LLM with search_web / fetch_url tool calls)
  3. Compare both judgments  → score delta, judgment_changed, sources used

Usage example
~~~~~~~~~~~~~
  python run_agentic_comparison.py \\
    --input ../data/scenarios_sample_short.csv \\
    --output-prefix ../outputs/agentic_run1 \\
    --guardrail flowjudge \\
    --provider openai --model gpt-4o-mini \\
    --guardrail-provider openai --guardrail-model gpt-4o \\
    --assistant-system-prompt-file ../config/assistant_system_prompt.txt \\
    --policy-files ../config/policy.txt ../config/policy_fa.txt \\
    --rubric-file ../config/rubric.txt \\
    --max-tool-calls 5
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Make sure imports resolve correctly whether run as a script or as a module.
sys.path.insert(0, os.path.dirname(__file__))

from providers import build_client, call_llm
from guardrails_runner import (
    create_guardrail,
    load_text_file,
    run_guardrail_for_policy,
)
from agentic_runner import run_agentic_guardrail, AgenticJudgment
from comparison import compare_judgments
from output_writer import write_outputs


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run scenarios through an assistant LLM and compare "
            "non-agentic (score-only) vs agentic (retrieval-enabled) guardrail judgments."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- I/O ----------------------------------------------------------------
    p.add_argument(
        "--input",
        required=True,
        help="Path to input CSV (must contain a 'scenario' column).",
    )
    p.add_argument(
        "--output-prefix",
        required=True,
        help="Prefix for output files, e.g. outputs/run1 (creates .csv and .json).",
    )

    # ---- Assistant LLM ------------------------------------------------------
    p.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "gemini", "mistral"],
        help="LLM provider for the assistant model (default: openai).",
    )
    p.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Assistant model name (default: gpt-4o-mini).",
    )
    p.add_argument(
        "--assistant-system-prompt-file",
        help="Path to a text file with the assistant system prompt.",
    )

    # ---- Guardrail (non-agentic backend) ------------------------------------
    p.add_argument(
        "--guardrail",
        default="flowjudge",
        choices=["flowjudge", "glider", "anyllm"],
        help="Guardrail backend for the non-agentic path (default: flowjudge).",
    )
    p.add_argument(
        "--glider-pass-criteria-file",
        help="Only used with --guardrail glider. Pass criteria file.",
    )
    p.add_argument(
        "--glider-rubric-file",
        help="Only used with --guardrail glider. Scoring rubric file.",
    )
    p.add_argument(
        "--flowjudge-metric-name",
        default="policy_compliance",
        help="Only used with --guardrail flowjudge. Metric name.",
    )
    p.add_argument(
        "--flowjudge-criteria-file",
        help="Only used with --guardrail flowjudge. Criteria description file.",
    )

    # ---- Guardrail judge LLM (used for BOTH paths) -------------------------
    p.add_argument(
        "--guardrail-provider",
        default=None,
        choices=["openai", "gemini", "mistral"],
        help=(
            "Provider for the guardrail/judge model. "
            "Defaults to --provider if not set. "
            "Must support OpenAI-compatible function calling."
        ),
    )
    p.add_argument(
        "--guardrail-model",
        default="gpt-4o-mini",
        help=(
            "Model used for both non-agentic (via any-guardrail) and agentic "
            "guardrail calls. For the agentic path this model must support "
            "function/tool calling. (default: gpt-4o-mini)"
        ),
    )

    # ---- Policies and rubric ------------------------------------------------
    p.add_argument(
        "--policy-files",
        nargs="+",
        required=True,
        help="One or more policy text files, e.g. config/policy.txt config/policy_fa.txt",
    )
    p.add_argument(
        "--rubric-file",
        help="Path to rubric text file.",
    )

    # ---- Agentic settings ---------------------------------------------------
    p.add_argument(
        "--max-tool-calls",
        type=int,
        default=5,
        help="Maximum tool invocations per agentic evaluation (default: 5).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print tool calls and search results in real time.",
    )

    return p


def process_row(
    row: Dict[str, Any],
    *,
    assistant_client,
    assistant_model: str,
    assistant_system_prompt: str,
    guardrail,
    guardrail_client,
    guardrail_model: str,
    policies: List[Tuple[str, str]],
    rubric: str,
    max_tool_calls: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Full pipeline for one CSV row.

    Steps:
      1. Call assistant LLM → assistant_response
      2. For each policy:
         a. Non-agentic guardrail evaluation
         b. Agentic guardrail evaluation
         c. Comparison
      3. Return flat dict with all output columns.
    """
    if "scenario" not in row:
        raise ValueError("Input CSV row is missing the required 'scenario' column.")

    scenario = row["scenario"]

    # 1. Assistant response
    assistant_response = call_llm(
        client=assistant_client,
        model=assistant_model,
        system_prompt=assistant_system_prompt,
        user_message=scenario,
    )

    out: Dict[str, Any] = dict(row)
    out.update(
        {
            "provider": assistant_client.__class__.__name__,  # informational
            "model": assistant_model,
            "assistant_system_prompt": assistant_system_prompt,
            "assistant_response": assistant_response,
            "guardrail_backend": type(guardrail).__name__,
            "guardrail_model": guardrail_model,
            "max_tool_calls_allowed": max_tool_calls,
        }
    )

    # 2. Evaluate with each policy
    for policy_label, policy_text in policies:
        base = policy_label  # e.g. "policy", "policy_fa"

        if verbose:
            print(f"    [policy: {policy_label}]")

        # 2a. Non-agentic path (uses any-guardrail library)
        if verbose:
            print(f"      non-agentic eval ...", end=" ", flush=True)
        try:
            gr = run_guardrail_for_policy(
                guardrail=guardrail,
                policy_text=policy_text,
                rubric=rubric,
                system_prompt=assistant_system_prompt,
                user_message=scenario,
                assistant_response=assistant_response,
            )
            out[f"{base}_nonagentic_valid"] = gr.valid
            out[f"{base}_nonagentic_score"] = gr.score
            out[f"{base}_nonagentic_explanation"] = gr.explanation
            if verbose:
                print(f"score={gr.score}  valid={gr.valid}")
        except Exception as e:
            out[f"{base}_nonagentic_valid"] = None
            out[f"{base}_nonagentic_score"] = None
            out[f"{base}_nonagentic_explanation"] = f"ERROR: {e}"
            if verbose:
                print(f"ERROR: {e}")
            gr = None  # type: ignore[assignment]

        nonagentic_valid = out.get(f"{base}_nonagentic_valid")
        nonagentic_score = out.get(f"{base}_nonagentic_score")

        # 2b. Agentic path (direct OpenAI call with tool calling)
        if verbose:
            print(f"      agentic eval (max {max_tool_calls} tool calls) ...")
        try:
            aj: AgenticJudgment = run_agentic_guardrail(
                client=guardrail_client,
                guardrail_model=guardrail_model,
                policy_text=policy_text,
                rubric=rubric,
                system_prompt=assistant_system_prompt,
                user_message=scenario,
                assistant_response=assistant_response,
                max_tool_calls=max_tool_calls,
                verbose=verbose,
            )
        except Exception as e:
            from agentic_runner import AgenticJudgment
            aj = AgenticJudgment(
                valid=None,
                score=None,
                explanation=f"ERROR: {e}",
                tool_calls_made=0,
            )
            if verbose:
                print(f"      ERROR in agentic eval: {e}")

        out[f"{base}_agentic_valid"] = aj.valid
        out[f"{base}_agentic_score"] = aj.score
        out[f"{base}_agentic_explanation"] = aj.explanation
        out[f"{base}_agentic_tool_calls_made"] = aj.tool_calls_made
        out[f"{base}_agentic_sources_used"] = aj.sources_used
        out[f"{base}_agentic_tool_call_log"] = aj.tool_call_log
        out[f"{base}_agentic_url_checks"] = aj.url_checks

        # 2c. Comparison
        cmp = compare_judgments(
            nonagentic_valid=nonagentic_valid,
            nonagentic_score=nonagentic_score,
            agentic_judgment=aj,
        )
        out[f"{base}_score_delta"] = cmp.score_delta
        out[f"{base}_judgment_changed"] = cmp.judgment_changed
        out[f"{base}_agentic_used_tools"] = cmp.agentic_used_tools

    return out


def main() -> None:
    load_dotenv()

    parser = build_arg_parser()
    args = parser.parse_args()

    # ---- Load text configs --------------------------------------------------
    assistant_system_prompt = load_text_file(
        args.assistant_system_prompt_file, default=""
    )
    rubric = load_text_file(args.rubric_file, default="")

    # ---- Glider setup -------------------------------------------------------
    glider_pass_criteria = ""
    glider_rubric = ""
    if args.guardrail == "glider":
        if not args.glider_pass_criteria_file:
            parser.error(
                "When using --guardrail glider, you must provide --glider-pass-criteria-file."
            )
        glider_pass_criteria = load_text_file(args.glider_pass_criteria_file, default="")
        if not glider_pass_criteria:
            parser.error(
                f"Glider pass criteria file is empty or missing: {args.glider_pass_criteria_file}"
            )
        if args.glider_rubric_file:
            glider_rubric = load_text_file(args.glider_rubric_file, default="")
        else:
            glider_rubric = rubric
        if not glider_rubric:
            parser.error(
                "When using --guardrail glider, either --glider-rubric-file or "
                "--rubric-file must point to a non-empty file."
            )

    # ---- FlowJudge setup ----------------------------------------------------
    flowjudge_criteria: Optional[str] = None
    if args.guardrail == "flowjudge" and args.flowjudge_criteria_file:
        flowjudge_criteria = load_text_file(args.flowjudge_criteria_file, default="")
        if not flowjudge_criteria:
            parser.error(
                f"FlowJudge criteria file is empty or missing: {args.flowjudge_criteria_file}"
            )

    # ---- Load policies ------------------------------------------------------
    policies: List[Tuple[str, str]] = []
    for policy_path in args.policy_files:
        text = load_text_file(policy_path, default="")
        if not text:
            parser.error(f"Policy file is empty or missing: {policy_path}")
        label = os.path.splitext(os.path.basename(policy_path))[0]
        policies.append((label, text))

    # ---- Build clients ------------------------------------------------------
    guardrail_provider = args.guardrail_provider or args.provider

    try:
        assistant_client = build_client(args.provider)
    except RuntimeError as e:
        sys.exit(f"Assistant client error: {e}")

    try:
        guardrail_client = build_client(guardrail_provider)
    except RuntimeError as e:
        sys.exit(f"Guardrail client error: {e}")

    # ---- Create non-agentic guardrail backend -------------------------------
    guardrail = create_guardrail(
        args.guardrail,
        glider_pass_criteria=glider_pass_criteria if args.guardrail == "glider" else None,
        glider_rubric=glider_rubric if args.guardrail == "glider" else None,
        flowjudge_metric_name=args.flowjudge_metric_name,
        flowjudge_criteria=flowjudge_criteria,
    )

    # ---- Read input CSV -----------------------------------------------------
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows_in = list(reader)

    if not rows_in:
        sys.exit(f"No rows found in input CSV: {args.input}")

    # ---- Run pipeline -------------------------------------------------------
    rows_out = []
    total = len(rows_in)

    for idx, row in enumerate(rows_in, start=1):
        scenario_id = row.get("id", idx)
        lang = row.get("language", "?")
        print(f"[{idx}/{total}] scenario id={scenario_id} lang={lang} ...")
        try:
            out_row = process_row(
                row,
                assistant_client=assistant_client,
                assistant_model=args.model,
                assistant_system_prompt=assistant_system_prompt,
                guardrail=guardrail,
                guardrail_client=guardrail_client,
                guardrail_model=args.guardrail_model,
                policies=policies,
                rubric=rubric,
                max_tool_calls=args.max_tool_calls,
                verbose=args.verbose,
            )
        except Exception as e:
            out_row = dict(row)
            out_row["error"] = str(e)
            print(f"  ERROR: {e}")
        rows_out.append(out_row)

    # ---- Write outputs ------------------------------------------------------
    csv_path, json_path = write_outputs(rows_out, args.output_prefix)

    print("\nDone.")
    print(f"CSV  → {csv_path}")
    print(f"JSON → {json_path}")
    print(f"Rows processed: {len(rows_out)}")


if __name__ == "__main__":
    main()
