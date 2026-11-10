# Multilingual LLM Guardrail Evaluation

A research toolkit for studying how LLM safety guardrails behave across languages, and whether giving a guardrail agentic capabilities (web search, URL checks, domain-specific lookups) changes its judgments compared to a guardrail that relies only on built-in knowledge.

Scenarios from three high-stakes domains are sent to an assistant LLM. A guardrail judge scores each response twice against the same policy, once without tools and once with tools, so the effect of tool access can be compared row by row. The three domains are:
- **Humanitarian:** Iran asylum cases, in English and Farsi
- **Financial:** in English and Spanish
- **Cybersecurity/social engineering:** phishing, in English and Spanish

**Accepted at:**
- [ACM FAccT 2026 — Tutorial](https://blog.mozilla.ai/from-evaluation-to-guardrails-what-we-brought-to-acm-facct-2026/)
- [NeurIPS 2026 — Trustworthy AI for Good Workshop](https://trustworthy-ai-for-good.github.io/)

## Setup

Requires Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r agentic_guardrails/requirements_agentic.txt

cp .env_example .env
```

In `.env`, set `OPENAI_API_KEY` (assistant), `ANTHROPIC_API_KEY` (judge), and `TAVILY_API_KEY` (web search). Optional domain-tool keys (`RELIEFWEB_APPNAME`, `URLSCAN_API_KEY`) are described in `.env_example`.

## Run

These are the settings behind the results in `data/output/`: a `gpt-5.5` assistant, a `claude-opus-4-6` judge, Tavily search, and up to 8 tool calls per evaluation. Run from the project root.

**Humanitarian (English):**

```bash
python agentic_guardrails/run_agentic_comparison.py \
  --input data/iran_asylum_all_scenarios_en.csv \
  --output-prefix outputs/humanitarian_en \
  --guardrail anyllm \
  --provider openai --model gpt-5.5 \
  --guardrail-judges anthropic:claude-opus-4-6 \
  --assistant-system-prompt-file config/assistant_system_prompt.txt \
  --policy-files config/humanitarian_policy_explicit_tool_selection_binary.txt config/policy_generic.txt \
  --rubric-file config/rubric.txt \
  --web-search-tool tavily \
  --tool-group humanitarian \
  --specialized-tool-reserve 3 \
  --judge-granularity full-policy \
  --judge-mode both \
  --max-tool-calls 8 \
  --verbose
```

**Other domains and languages:** change `--input`, `--tool-group`, and the domain policy file:

| Domain | `--input` | `--tool-group` | Domain policy |
|---|---|---|---|
| Humanitarian | `data/iran_asylum_all_scenarios_{en,fa}.csv` | `humanitarian` | `config/humanitarian_policy_explicit_tool_selection_binary.txt` |
| Financial | `data/financial_scenarios_{en,es}.csv` | `financial` | `config/financial_policy_explicit_en_binary.txt` |
| Cybersecurity/social engineering | `data/cybersecurity_phishing_scenarios_{en,es}.csv` | `cybersecurity` | `config/policy_cybersecurity_explicit_binary.txt` |

Always pass `config/policy_generic.txt` as a second policy file, as in the humanitarian example.

Run `python agentic_guardrails/run_agentic_comparison.py --help` for all options.

## Results

The final outputs for all three domains are in `data/output/` (`*_guardrails_output.json`). Human annotation exports are in `data/output/annotators_analysis/`. The `annotation-platform/` directory contains the web app used for annotation.

For the full original documentation (pipeline details, output columns, tools, troubleshooting), see [README_old.md](README_old.md).
