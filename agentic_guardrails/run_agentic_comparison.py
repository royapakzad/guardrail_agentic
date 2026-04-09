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
    --provider openai --model gpt-5-mini \\
    --guardrail-provider openai --guardrail-model gpt-5 \\
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
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Suppress a harmless Pydantic serialization warning produced by any-guardrail 0.2.2
# when used with openai>=2.x. The warning fires because the newer OpenAI SDK populates
# a `parsed` field that any-guardrail's internal model declares as None. Results are
# unaffected — the GuardrailOutput is captured correctly before serialization runs.
warnings.filterwarnings(
    "ignore",
    message=".*PydanticSerializationUnexpectedValue.*parsed.*",
    category=UserWarning,
)

from dotenv import load_dotenv

# Make sure imports resolve correctly whether run as a script or as a module.
sys.path.insert(0, os.path.dirname(__file__))

from providers import call_llm
from guardrails_runner import (
    create_guardrail,
    load_text_file,
    build_guardrail_input_text,
    run_guardrail_for_policy,
)
from agentic_runner import run_agentic_guardrail, AgenticJudgment
from comparison import compare_judgments
from output_writer import write_outputs
from scenario_logger import ScenarioLogger


def _count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count tokens in text using tiktoken.

    For OpenAI GPT-4 / GPT-4o family models this is exact — tiktoken is the
    same tokenizer OpenAI uses server-side. For other providers (Gemini,
    Mistral, etc.) cl100k_base is used as a close approximation; the count
    will be accurate enough for context-window comparisons.

    Falls back to len(text) // 4 only if tiktoken is not installed (run
    `pip install tiktoken` to get exact counts).

    Why this works for the non-agentic path:
        The any-guardrail library does not expose resp.usage, but we built
        eval_input_text ourselves, so we know exactly what text was sent.
        Counting its tokens with the same tokenizer the model uses gives an
        exact prompt-token count. The completion text (gr.explanation) is
        likewise exactly what the model returned, so tiktoken on that text
        gives the exact completion-token count.
    """
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            # Model not in tiktoken's registry (non-OpenAI provider) —
            # cl100k_base (GPT-4 family) is a good approximation.
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # tiktoken not installed — fall back to character heuristic.
        return max(1, len(text) // 4)


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
    p.add_argument(
        "--log-dir",
        default=None,
        help=(
            "Directory for per-scenario observability logs (.txt and .json). "
            "Defaults to <output-prefix>_logs/. Pass 'none' to disable logging."
        ),
    )

    # ---- Assistant LLM ------------------------------------------------------
    p.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "anthropic", "gemini", "mistral", "cohere", "deepseek", "cerebras", "ollama"],
        help="LLM provider for the assistant model (default: openai).",
    )
    p.add_argument(
        "--model",
        default="gpt-5-mini",
        help="Assistant model name (default: gpt-5-mini — fast, free-tier).",
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
        choices=["openai", "anthropic", "gemini", "mistral", "cohere", "deepseek", "cerebras", "ollama"],
        help=(
            "Provider for the guardrail/judge model (defaults to --provider). "
            "Must support function/tool calling for the agentic path."
        ),
    )
    p.add_argument(
        "--guardrail-model",
        default="gpt-5-mini",
        help=(
            "Model used for both non-agentic (via any-guardrail) and agentic "
            "guardrail calls. For the agentic path this model must support "
            "function/tool calling. Use gpt-5 for stronger judgments, "
            "gpt-5-mini for faster/cheaper runs. (default: gpt-5-mini)"
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
    assistant_provider: str,
    assistant_model: str,
    assistant_system_prompt: str,
    guardrail,
    guardrail_provider: str,
    guardrail_model: str,
    policies: List[Tuple[str, str]],
    rubric: str,
    max_tool_calls: int,
    verbose: bool = False,
    log_dir: Optional[str] = None,
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
    scenario_id = row.get("id", "unknown")
    language = row.get("language", "")

    # Create a per-scenario logger if a log directory was specified.
    logger: Optional[ScenarioLogger] = None
    if log_dir and log_dir.lower() != "none":
        logger = ScenarioLogger(
            log_dir=log_dir,
            scenario_id=scenario_id,
            scenario_text=scenario,
            language=language,
        )

    # 1. Assistant response
    assistant_response = call_llm(
        provider=assistant_provider,
        model=assistant_model,
        system_prompt=assistant_system_prompt,
        user_message=scenario,
    )

    if logger is not None:
        logger.log_response_generation(
            provider=assistant_provider,
            model=assistant_model,
            system_prompt=assistant_system_prompt,
            user_message=scenario,
            assistant_response=assistant_response,
        )

    out: Dict[str, Any] = dict(row)
    out.update(
        {
            "provider": assistant_provider,
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
        # Build the input text here so we can log it; run_guardrail_for_policy
        # builds it internally too, but we want to capture what was sent.
        nonagentic_eval_text = build_guardrail_input_text(
            policy=policy_text,
            rubric=rubric,
            system_prompt=assistant_system_prompt,
            user_message=scenario,
            assistant_response=assistant_response,
        )
        # Count tokens for the non-agentic path using tiktoken on the text we have.
        # Prompt tokens: eval_input_text is exactly what gets sent to the guardrail —
        #   we built it, so we know its contents precisely.
        # Completion tokens: gr.explanation is exactly what the model returned.
        # This gives the same counts the provider would report in resp.usage.
        na_prompt_tokens = _count_tokens(nonagentic_eval_text, model=guardrail_model)
        try:
            gr = run_guardrail_for_policy(
                guardrail=guardrail,
                policy_text=policy_text,
                rubric=rubric,
                system_prompt=assistant_system_prompt,
                user_message=scenario,
                assistant_response=assistant_response,
            )
            na_completion_tokens = _count_tokens(str(gr.explanation or ""), model=guardrail_model)
            na_total_tokens = na_prompt_tokens + na_completion_tokens
            out[f"{base}_nonagentic_valid"] = gr.valid
            out[f"{base}_nonagentic_score"] = gr.score
            out[f"{base}_nonagentic_explanation"] = gr.explanation
            out[f"{base}_nonagentic_prompt_tokens"] = na_prompt_tokens
            out[f"{base}_nonagentic_completion_tokens"] = na_completion_tokens
            out[f"{base}_nonagentic_total_tokens"] = na_total_tokens
            if logger is not None:
                logger.log_nonagentic_eval(
                    policy_label=policy_label,
                    policy_text=policy_text,
                    rubric=rubric,
                    eval_input_text=nonagentic_eval_text,
                    valid=gr.valid,
                    score=gr.score,
                    explanation=str(gr.explanation),
                    prompt_tokens=na_prompt_tokens,
                    completion_tokens=na_completion_tokens,
                    total_tokens=na_total_tokens,
                )
            if verbose:
                print(f"score={gr.score}  valid={gr.valid}  tokens={na_total_tokens:,}")
        except Exception as e:
            out[f"{base}_nonagentic_valid"] = None
            out[f"{base}_nonagentic_score"] = None
            out[f"{base}_nonagentic_explanation"] = f"ERROR: {e}"
            out[f"{base}_nonagentic_prompt_tokens"] = na_prompt_tokens
            out[f"{base}_nonagentic_completion_tokens"] = None
            out[f"{base}_nonagentic_total_tokens"] = na_prompt_tokens
            if logger is not None:
                logger.log_nonagentic_eval(
                    policy_label=policy_label,
                    policy_text=policy_text,
                    rubric=rubric,
                    eval_input_text=nonagentic_eval_text,
                    valid=None,
                    score=None,
                    explanation=f"ERROR: {e}",
                    prompt_tokens=na_prompt_tokens,
                    completion_tokens=None,
                    total_tokens=na_prompt_tokens,
                )
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
                provider=guardrail_provider,
                guardrail_model=guardrail_model,
                policy_text=policy_text,
                rubric=rubric,
                system_prompt=assistant_system_prompt,
                user_message=scenario,
                assistant_response=assistant_response,
                max_tool_calls=max_tool_calls,
                verbose=verbose,
                logger=logger,
                policy_label=policy_label,
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
        # Token usage — exact from provider API (None if provider did not return usage)
        out[f"{base}_agentic_prompt_tokens_total"] = aj.prompt_tokens_total
        out[f"{base}_agentic_completion_tokens_total"] = aj.completion_tokens_total
        out[f"{base}_agentic_total_tokens"] = aj.total_tokens_used
        out[f"{base}_agentic_peak_prompt_tokens"] = aj.peak_prompt_tokens
        out[f"{base}_agentic_token_usage_per_turn"] = aj.token_usage_per_turn

        # 2c. Comparison
        cmp = compare_judgments(
            nonagentic_valid=nonagentic_valid,
            nonagentic_score=nonagentic_score,
            agentic_judgment=aj,
        )
        out[f"{base}_score_delta"] = cmp.score_delta
        out[f"{base}_judgment_changed"] = cmp.judgment_changed
        out[f"{base}_agentic_used_tools"] = cmp.agentic_used_tools

        if logger is not None:
            logger.log_comparison(
                policy_label=policy_label,
                nonagentic_valid=nonagentic_valid,
                nonagentic_score=nonagentic_score,
                agentic_valid=aj.valid,
                agentic_score=aj.score,
                score_delta=cmp.score_delta,
                judgment_changed=cmp.judgment_changed,
                agentic_used_tools=cmp.agentic_used_tools,
            )

    if logger is not None:
        txt_path, json_path = logger.finalize()
        if verbose:
            print(f"    [log] {txt_path}")

    return out


def main() -> None:
    # Load .env so API keys are available without shell-level exports.
    load_dotenv()

    parser = build_arg_parser()
    args = parser.parse_args()

    # ---- Load shared text configs -------------------------------------------
    # These files are read once and passed to every row's evaluation.
    assistant_system_prompt = load_text_file(
        args.assistant_system_prompt_file, default=""
    )
    rubric = load_text_file(args.rubric_file, default="")

    # ---- Backend-specific setup --------------------------------------------
    # Only the selected guardrail backend's config files are loaded.

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
        # Fall back to the shared rubric if no Glider-specific rubric is given.
        if args.glider_rubric_file:
            glider_rubric = load_text_file(args.glider_rubric_file, default="")
        else:
            glider_rubric = rubric
        if not glider_rubric:
            parser.error(
                "When using --guardrail glider, either --glider-rubric-file or "
                "--rubric-file must point to a non-empty file."
            )

    flowjudge_criteria: Optional[str] = None
    if args.guardrail == "flowjudge" and args.flowjudge_criteria_file:
        flowjudge_criteria = load_text_file(args.flowjudge_criteria_file, default="")
        if not flowjudge_criteria:
            parser.error(
                f"FlowJudge criteria file is empty or missing: {args.flowjudge_criteria_file}"
            )

    # ---- Load policy files -------------------------------------------------
    # Each policy file becomes its own set of output columns (non-agentic +
    # agentic + comparison), labelled by the filename without extension.
    policies: List[Tuple[str, str]] = []
    for policy_path in args.policy_files:
        text = load_text_file(policy_path, default="")
        if not text:
            parser.error(f"Policy file is empty or missing: {policy_path}")
        label = os.path.splitext(os.path.basename(policy_path))[0]  # e.g. "policy_fa"
        policies.append((label, text))

    # ---- Resolve providers -------------------------------------------------
    # The assistant and guardrail judge can use different providers/models.
    # If --guardrail-provider is not set, it defaults to the assistant provider.
    # API keys are picked up automatically from environment variables by any-llm-sdk.
    assistant_provider = args.provider
    guardrail_provider = args.guardrail_provider or args.provider

    # ---- Instantiate the non-agentic guardrail backend ---------------------
    # This object is reused across all rows. The agentic path does not use it —
    # it calls run_agentic_guardrail() directly via the tool-calling loop.
    guardrail = create_guardrail(
        args.guardrail,
        glider_pass_criteria=glider_pass_criteria if args.guardrail == "glider" else None,
        glider_rubric=glider_rubric if args.guardrail == "glider" else None,
        flowjudge_metric_name=args.flowjudge_metric_name,
        flowjudge_criteria=flowjudge_criteria,
    )

    # ---- Read input CSV ----------------------------------------------------
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows_in = list(reader)

    if not rows_in:
        sys.exit(f"No rows found in input CSV: {args.input}")

    # ---- Resolve log directory ---------------------------------------------
    # Default to <output_prefix>_logs/ if not explicitly set.
    log_dir: Optional[str] = args.log_dir
    if log_dir is None:
        log_dir = args.output_prefix + "_logs"
    elif log_dir.lower() == "none":
        log_dir = None

    if log_dir:
        print(f"Scenario logs → {log_dir}/")

    # ---- Main pipeline loop ------------------------------------------------
    # Each row is independent. Errors are caught per-row so a single API
    # failure does not abort the entire run.
    rows_out = []
    total = len(rows_in)

    for idx, row in enumerate(rows_in, start=1):
        scenario_id = row.get("id", idx)
        lang = row.get("language", "?")
        print(f"[{idx}/{total}] scenario id={scenario_id} lang={lang} ...")
        try:
            out_row = process_row(
                row,
                assistant_provider=assistant_provider,
                assistant_model=args.model,
                assistant_system_prompt=assistant_system_prompt,
                guardrail=guardrail,
                guardrail_provider=guardrail_provider,
                guardrail_model=args.guardrail_model,
                policies=policies,
                rubric=rubric,
                max_tool_calls=args.max_tool_calls,
                verbose=args.verbose,
                log_dir=log_dir,
            )
        except Exception as e:
            # Preserve original columns and append the error so the row is
            # still present in the output for debugging.
            out_row = dict(row)
            out_row["error"] = str(e)
            print(f"  ERROR: {e}")
        rows_out.append(out_row)

    # ---- Write outputs -----------------------------------------------------
    # write_outputs() handles CSV (with JSON-encoded complex values) and JSON
    # (with native Python types). Both files share the same output_prefix.
    csv_path, json_path = write_outputs(rows_out, args.output_prefix)

    print("\nDone.")
    print(f"CSV  → {csv_path}")
    print(f"JSON → {json_path}")
    if log_dir:
        print(f"Logs → {log_dir}/")
    print(f"Rows processed: {len(rows_out)}")


if __name__ == "__main__":
    main()
