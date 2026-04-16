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
import re
import sys
import warnings
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

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


class JudgeSpec(NamedTuple):
    """A single guardrail judge: provider + model + a safe label for column names."""
    provider: str   # e.g. "openai", "anthropic", "gemini"
    model: str      # e.g. "gpt-5-nano", "claude-sonnet-4-6", "gemini-2.5-flash"
    label: str      # sanitized for CSV column names, e.g. "gpt_5_nano"

    @property
    def model_id(self) -> str:
        """any-llm / any-guardrail model_id format: 'provider:model'."""
        return f"{self.provider}:{self.model}"


def _parse_judge(spec: str) -> JudgeSpec:
    """
    Parse a 'provider:model' string into a JudgeSpec.
    The label is the model name with non-alphanumeric characters replaced by '_'.
    Example: 'anthropic:claude-sonnet-4-6' → JudgeSpec('anthropic', 'claude-sonnet-4-6', 'claude_sonnet_4_6')
    """
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"Invalid judge format {spec!r}. Expected 'provider:model', "
            "e.g. 'openai:gpt-5-nano' or 'anthropic:claude-sonnet-4-6'."
        )
    provider, model = spec.split(":", 1)
    label = re.sub(r"[^a-zA-Z0-9]", "_", model)
    return JudgeSpec(provider=provider.strip(), model=model.strip(), label=label)


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

    # ---- Guardrail judge LLM(s) — used for BOTH non-agentic and agentic ----
    #
    # --guardrail-judges  (preferred, multi-judge)
    #   Pass one or more 'provider:model' strings. Each judge runs independently
    #   on the same assistant response. Output columns are namespaced by model,
    #   e.g. policy_gpt_5_nano_nonagentic_score, policy_claude_sonnet_4_6_agentic_score.
    #   Only works with --guardrail anyllm (flowjudge/glider ignore this flag).
    #   Example:
    #     --guardrail-judges openai:gpt-5-nano anthropic:claude-sonnet-4-6 gemini:gemini-2.5-flash
    #
    # --guardrail-provider / --guardrail-model  (legacy, single-judge)
    #   Kept for backward compatibility. Ignored when --guardrail-judges is set.
    p.add_argument(
        "--guardrail-judges",
        nargs="+",
        metavar="PROVIDER:MODEL",
        default=None,
        help=(
            "One or more guardrail judges as 'provider:model' pairs. "
            "Each judge evaluates every scenario independently (non-agentic + agentic). "
            "Only applies when --guardrail anyllm is used. "
            "Example: --guardrail-judges openai:gpt-5-nano anthropic:claude-sonnet-4-6"
        ),
    )
    p.add_argument(
        "--guardrail-provider",
        default=None,
        choices=["openai", "anthropic", "gemini", "mistral", "cohere", "deepseek", "cerebras", "ollama"],
        help=(
            "Single-judge mode: provider for the guardrail/judge model "
            "(defaults to --provider). Ignored when --guardrail-judges is set."
        ),
    )
    p.add_argument(
        "--guardrail-model",
        default="gpt-5-mini",
        help=(
            "Single-judge mode: model for both non-agentic and agentic guardrail calls. "
            "Ignored when --guardrail-judges is set. (default: gpt-5-mini)"
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
    judges: List[JudgeSpec],
    policies: List[Tuple[str, str]],
    rubric: str,
    max_tool_calls: int,
    verbose: bool = False,
    log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full pipeline for one CSV row.

    Steps:
      1. Call assistant LLM once → assistant_response
      2. For each policy × each judge:
         a. Non-agentic guardrail evaluation (uses judge.model_id)
         b. Agentic guardrail evaluation    (uses judge.provider + judge.model)
         c. Comparison
      3. Return flat dict with all output columns.

    Output columns are namespaced by judge label so multiple judges can be
    compared side by side:
      {policy}_{judge_label}_nonagentic_score
      {policy}_{judge_label}_agentic_score
      {policy}_{judge_label}_score_delta
      ... etc.
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

    # 1. Call the assistant LLM once — all judges evaluate this same response.
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
            "guardrail_judges": [j.model_id for j in judges],
            "max_tool_calls_allowed": max_tool_calls,
        }
    )

    # 2. Evaluate with each policy × each judge independently.
    for policy_label, policy_text in policies:
        if verbose:
            print(f"    [policy: {policy_label}]")

        # Build eval text once per policy — same text goes to every judge.
        nonagentic_eval_text = build_guardrail_input_text(
            policy=policy_text,
            rubric=rubric,
            system_prompt=assistant_system_prompt,
            user_message=scenario,
            assistant_response=assistant_response,
        )

        for judge in judges:
            # Column prefix: policy_judgerlabel_ e.g. "policy_claude_sonnet_4_6_"
            base = f"{policy_label}_{judge.label}"

            if verbose:
                print(f"      [judge: {judge.model_id}]")

            # 2a. Non-agentic path
            if verbose:
                print(f"        non-agentic eval ...", end=" ", flush=True)
            na_prompt_tokens = _count_tokens(nonagentic_eval_text, model=judge.model)
            try:
                gr = run_guardrail_for_policy(
                    guardrail=guardrail,
                    policy_text=policy_text,
                    rubric=rubric,
                    system_prompt=assistant_system_prompt,
                    user_message=scenario,
                    assistant_response=assistant_response,
                    model_id=judge.model_id,
                )
                na_completion_tokens = _count_tokens(str(gr.explanation or ""), model=judge.model)
                na_total_tokens = na_prompt_tokens + na_completion_tokens
                out[f"{base}_nonagentic_valid"] = gr.valid
                out[f"{base}_nonagentic_score"] = gr.score
                out[f"{base}_nonagentic_explanation"] = gr.explanation
                out[f"{base}_nonagentic_prompt_tokens"] = na_prompt_tokens
                out[f"{base}_nonagentic_completion_tokens"] = na_completion_tokens
                out[f"{base}_nonagentic_total_tokens"] = na_total_tokens
                if logger is not None:
                    logger.log_nonagentic_eval(
                        policy_label=f"{policy_label}[{judge.model_id}]",
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
                        policy_label=f"{policy_label}[{judge.model_id}]",
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

            nonagentic_valid = out.get(f"{base}_nonagentic_valid")
            nonagentic_score = out.get(f"{base}_nonagentic_score")

            # 2b. Agentic path
            if verbose:
                print(f"        agentic eval (max {max_tool_calls} tool calls) ...")
            try:
                aj: AgenticJudgment = run_agentic_guardrail(
                    provider=judge.provider,
                    guardrail_model=judge.model,
                    policy_text=policy_text,
                    rubric=rubric,
                    system_prompt=assistant_system_prompt,
                    user_message=scenario,
                    assistant_response=assistant_response,
                    max_tool_calls=max_tool_calls,
                    verbose=verbose,
                    logger=logger,
                    policy_label=f"{policy_label}[{judge.model_id}]",
                )
            except Exception as e:
                aj = AgenticJudgment(
                    valid=None,
                    score=None,
                    explanation=f"ERROR: {e}",
                    tool_calls_made=0,
                )
                if verbose:
                    print(f"        ERROR in agentic eval: {e}")

            out[f"{base}_agentic_valid"] = aj.valid
            out[f"{base}_agentic_score"] = aj.score
            out[f"{base}_agentic_explanation"] = aj.explanation
            out[f"{base}_agentic_tool_calls_made"] = aj.tool_calls_made
            out[f"{base}_agentic_sources_used"] = aj.sources_used
            out[f"{base}_agentic_tool_call_log"] = aj.tool_call_log
            out[f"{base}_agentic_url_checks"] = aj.url_checks
            out[f"{base}_agentic_claim_checks"] = aj.claim_checks
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
                    policy_label=f"{policy_label}[{judge.model_id}]",
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


def _extract_judge_rows(
    rows: List[Dict[str, Any]],
    judge: JudgeSpec,
    base_keys: set,
) -> List[Dict[str, Any]]:
    """
    Extract per-judge rows from the mega rows dict.

    Columns that belong to this judge (e.g. 'policy_claude_sonnet_4_6_nonagentic_score')
    are renamed to the single-judge format ('policy_nonagentic_score') so that
    visualize_results.py and other downstream tools work without changes.

    Base columns (scenario, assistant response, etc.) are carried through as-is.
    A 'guardrail_model' column is added with the judge's provider:model string.
    """
    marker = f"_{judge.label}_"
    result = []
    for row in rows:
        out: Dict[str, Any] = {}
        for key, val in row.items():
            if key in base_keys:
                out[key] = val
            elif marker in key:
                # "policy_gpt_5_nano_nonagentic_score" → "policy_nonagentic_score"
                out[key.replace(marker, "_", 1)] = val
        out["guardrail_model"] = judge.model_id
        result.append(out)
    return result


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
    assistant_provider = args.provider

    # ---- Build the judges list ---------------------------------------------
    # --guardrail-judges takes priority. Falls back to the legacy single-judge
    # --guardrail-provider / --guardrail-model flags.
    if args.guardrail_judges:
        try:
            judges: List[JudgeSpec] = [_parse_judge(s) for s in args.guardrail_judges]
        except argparse.ArgumentTypeError as e:
            parser.error(str(e))
    else:
        # Legacy single-judge mode: --guardrail-provider / --guardrail-model
        single_provider = args.guardrail_provider or args.provider
        single_model = args.guardrail_model
        judges = [_parse_judge(f"{single_provider}:{single_model}")]

    judge_labels = ", ".join(j.model_id for j in judges)
    print(f"Guardrail judges: {judge_labels}")

    # ---- Instantiate the non-agentic guardrail backend ---------------------
    # One shared instance is reused for all judges and rows.
    # For anyllm the model is passed at call time (via model_id), so a single
    # instance handles every judge without re-instantiation.
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
                judges=judges,
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
    # Base columns are non-judge-specific and included in every per-judge file.
    base_keys: set = set(rows_in[0].keys()) if rows_in else set()
    base_keys |= {
        "provider", "model", "assistant_system_prompt", "assistant_response",
        "guardrail_backend", "max_tool_calls_allowed", "error",
    }

    # Per-judge files: clean column names (no judge label), compatible with
    # visualize_results.py. Written to <output_prefix>_<judge_label>.[csv|json]
    print("\nPer-judge outputs:")
    for judge in judges:
        judge_rows = _extract_judge_rows(rows_out, judge, base_keys)
        judge_prefix = f"{args.output_prefix}_{judge.label}"
        j_csv, j_json = write_outputs(judge_rows, judge_prefix)
        print(f"  [{judge.model_id}]")
        print(f"    CSV  → {j_csv}")
        print(f"    JSON → {j_json}")

    # Mega file: all judges combined in one file with judge-namespaced columns.
    # Useful for cross-judge analysis and comparison scripts.
    csv_path, json_path = write_outputs(rows_out, f"{args.output_prefix}_all")
    print(f"\nMega file (all judges combined):")
    print(f"  CSV  → {csv_path}")
    print(f"  JSON → {json_path}")

    if log_dir:
        print(f"\nLogs → {log_dir}/")
    print(f"Rows processed: {len(rows_out)}")


if __name__ == "__main__":
    main()
