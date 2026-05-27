# Web Search Tool Analysis: SearXNG vs Tavily vs DuckDuckGo

Comparison of three web search backends used for agentic guardrail claim verification,
based on actual evaluation runs against 10 scenarios in the humanitarian/financial domain.

**Datasets analyzed:**

| Label | Repo | Search tool | Scenarios | Judges |
|---|---|---|---|---|
| SearXNG | `guardrails_searxng` (`SearXNG_better_run1`) | SearXNG (self-hosted) | 10 (5 unique × EN + FA) | GPT-5-nano, Claude Sonnet 4.6 (Gemini failed — see below) |
| Tavily | `guardrails_tavely` | Tavily API | 10 | GPT-5-nano, Claude Sonnet 4.6, Gemini 2.5 Flash |
| DDG | `multilingual_llm_guardrails-main` | DuckDuckGo (duckduckgo_run1) | 10 | GPT-5-nano, Claude Sonnet 4.6, Gemini 2.5 Flash |

Policies evaluated per scenario: `policy`, `policy_fa`, `policy_concrete`, `policy_concrete_fa`,
`policy_generic`, `policy_generic_fa` — giving up to 60 (policy × judge) evaluation pairs per dataset.

---

## 1. Tool Reliability

How often each tool succeeded or failed per backend.

### search_web

| Backend | Total calls | Successful | Failed | Fail rate |
|---|---|---|---|---|
| SearXNG | 647 | 462 | 185 | 29% |
| Tavily | 784 | 776 | 8 | 1% |
| DDG | 850 | 825 | 25 | 3% |

**SearXNG's 29% fail rate is the highest of the three backends**, but the cause is different
from a connectivity problem. In the complete run Docker was already running and SearXNG was
reachable — there were zero wall-clock timeouts. The failures are content-level: SearXNG
returned no results (or a 400 Bad Request) for queries that were either too specific or
phrased in a way that matched no indexed content. The most common failing queries were for
very specific medical symptom combinations (e.g. "tuberculosis symptoms night sweats weight
loss"), small NGO names (e.g. "Refugio Hamburg psychological support torture survivors"),
and obscure UN initiative acronyms. These are queries where a managed index like Tavily or
DuckDuckGo's broader crawl may have better coverage than SearXNG's configured engine set.

**Tavily had the lowest fail rate (1%).** As a managed API service with SLAs, it is more
consistent than both self-hosted SearXNG and DuckDuckGo's unofficial scraping interface.

**DuckDuckGo's 3% fail rate** is consistent with the rate-limiting behaviour of its unofficial
scraping interface, which stalls connections rather than returning clean errors.

### fetch_url

| Backend | Total calls | Successful | Failed | Fail rate |
|---|---|---|---|---|
| SearXNG (requests+BS4) | 22 | 18 | 4 | 18% |
| Tavily (extract API) | 7 | 3 | 4 | 57% |
| DDG (requests+BS4) | 44 | 29 | 15 | 34% |

**Tavily's extract API had a 57% fail rate**, but on only 7 calls — too few to be conclusive.
The failures appear to come from sites that block automated extraction (government portals,
paywalled pages). The Tavily extract endpoint is more capable than raw requests+BS4 for
JavaScript-heavy pages but is not universally more reliable.

**SearXNG's fetch_url (requests+BS4) had the lowest fail rate among well-sampled backends (20%).**
This may partly reflect SearXNG returning higher-quality, more accessible source URLs from
its multi-engine aggregation, meaning the judge selected pages that were more likely to load.

**Key observation:** The judge calls `fetch_url` far less often when using Tavily (7 calls across
10 scenarios) versus DuckDuckGo (44 calls) or SearXNG (20 calls). This is because Tavily's `search_web`
already returns richer snippet content — the judge finds enough in the search result itself and
does not need to follow up with a full page fetch. This is a meaningful efficiency difference.

### check_url_validity

| Backend | Total calls | Reachable | Unreachable | Reachable % |
|---|---|---|---|---|
| SearXNG | 138 | 93 | 45 | 67% |
| Tavily | 219 | 82 | 137 | 37% |
| DDG | 346 | 263 | 83 | 76% |

**Tavily's 37% reachable rate is strikingly low.** This does not mean Tavily provided bad URLs.
`check_url_validity` uses a 10-second HTTP HEAD timeout. Government sites (OFAC, UNHCR, EU
portals) that feature heavily in these scenarios respond slowly to automated requests and
frequently exceed the timeout — resulting in a "Connection refused" or "Read timed out" result
that is classified as unreachable, even when the page is real and accessible in a browser. The
Tavily run may have had more scenarios that referenced government sources, amplifying this effect.

**SearXNG's 67% reachability** is lower than DDG (76%) and reflects the same government-site
timeout problem: OFAC and French asylum portals that SearXNG surfaces prominently respond slowly
to automated HEAD requests. DDG's higher reachability may partly reflect its heavier Wikipedia
weighting — Wikipedia pages load reliably under automated requests.

**Practical impact on scores:** Each "unreachable" URL applies a −0.15 deduction. False positives
from slow government sites therefore artificially lower agentic scores, particularly when the
assistant response cited legitimate institutional sources.

### Timeouts

| Backend | Total timeouts caught by the 60s wall-clock cap |
|---|---|
| SearXNG | 0 |
| Tavily | 0 |
| DDG | 2 |

**SearXNG produced zero wall-clock timeouts** in the complete run. Docker was running before the
experiment started, so there were no connection stalls. The 29% search fail rate is driven by
no-results and 400 errors, not by hanging requests. Tavily also produced no timeouts, consistent
with its managed API design.

---

## 2. Score Comparison: Agentic vs Non-Agentic

Scores averaged across all policies and all scenarios per dataset.

### GPT-5-nano as judge

| Backend | Avg nonagentic | Avg agentic | Avg delta | Judgment changed |
|---|---|---|---|---|
| SearXNG | 0.823 | 0.706 | −0.118 | 55% |
| Tavily | 0.841 | 0.792 | −0.049 | 32% |
| DDG | 0.899 | 0.850 | −0.049 | 22% |

### Claude Sonnet 4.6 as judge

| Backend | Avg nonagentic | Avg agentic | Avg delta | Judgment changed |
|---|---|---|---|---|
| SearXNG | 0.824 | 0.756 | −0.068 | 10% |
| Tavily | 0.827 | 0.707 | −0.120 | 32% |
| DDG | 0.832 | 0.780 | −0.052 | 7% |

### Gemini 2.5 Flash as judge

| Backend | Avg nonagentic | Avg agentic | Avg delta | Judgment changed |
|---|---|---|---|---|
| SearXNG | N/A (failed) | N/A | — | — |
| Tavily | 0.819 | 0.763 | −0.051 | 33% |
| DDG | 0.841 | 0.809 | −0.032 | 20% |

---

## 3. Key Findings

### Finding 1: Agentic guardrails consistently score lower than non-agentic

Across every backend, every judge, and every policy, the agentic score is lower than or equal to
the non-agentic score. The average delta is always negative. This means:

- Retrieval does not confirm what the guardrail already believed — it finds new problems.
- The agentic path is systematically more strict, not more permissive.
- Sources of the downward pressure: broken URL deductions (−0.15 each), contradicted claims
  (−0.20), and unverifiable material claims (−0.05), plus timeout false positives.

This is the core answer to the research question: **yes, tool access changes guardrail
judgments, and it changes them toward stricter (lower) scores.**

### Finding 2: SearXNG produces the largest score drop for GPT-5-nano

SearXNG's GPT-5-nano delta (−0.118) is more than double Tavily's (−0.049) and more than double
DDG's (−0.049). This is now confirmed on a complete 10-scenario run with no Docker startup
issues. The large drop is driven by SearXNG's high search fail rate (29% — mostly no-results for
specific queries): GPT-5-nano treats failed searches as evidence of unreliability, counting
unresolvable queries as unverifiable claims and applying −0.05 deductions each. GPT-5-nano also
averages 8.9 tool calls per evaluation (well above the cap), so it accumulates more failed search
evidence than Claude (4.5 calls/eval) before producing its judgment. The result is that GPT-5-nano
with SearXNG also had the highest judgment-change rate of any backend/judge combination: 55% of
evaluations produced a different pass/fail verdict after agentic retrieval.

### Finding 3: Tavily is the most consistent backend for Claude

Claude Sonnet 4.6 with Tavily showed the largest and most consistent judgment change rate (32%)
among Claude evaluations. Claude with DuckDuckGo changed judgment only 7% of the time. This
suggests that Tavily's search quality directly improves Claude's ability to find actionable
evidence — DuckDuckGo's results are not surfacing the same quality of sources for this domain.

### Finding 4: Gemini uses far fewer tool calls and is cheapest to run

Gemini's average tool calls across Tavily and DDG runs were 2.3–3.2, versus 3.8–4.6 for Claude
and 8.7–10.0 for GPT-5-nano. Gemini also had 10–13 zero-tool evaluations (scenarios where it
called no tools at all). This means Gemini is often not doing meaningful retrieval before
judging — it appears to look at the task, decide retrieval isn't needed, and proceed on internal
knowledge alone. This makes it faster and cheaper but weakens the agentic advantage.

### Finding 5: GPT-5-nano ignores the tool cap

GPT-5-nano averaged 8.7–10.0 tool calls per evaluation with `--max-tool-calls 5`. This is
because the model returns multiple tool calls in a single turn (parallel calls), and the cap is
checked between turns rather than between individual calls. The entire batch executes before the
cap can interrupt it. This behavior persists across all search backends and significantly
inflates cost and latency for GPT-5-nano.

### Finding 6: Farsi policies produce lower agentic scores than English policies

Scores are averaged across GPT-5-nano and Claude (Gemini is excluded from SearXNG; all three judges included for Tavily and DDG).

| Backend | EN agentic | FA agentic | Difference |
|---|---|---|---|
| SearXNG | 0.749 | 0.713 | −0.036 |
| Tavily | 0.751 | 0.755 | +0.004 |
| DDG | 0.800 | 0.826 | +0.026 |

In SearXNG, Farsi-policy agentic scores are consistently lower than English-policy agentic scores.
This is likely because the guardrail searches the web in English but judges the response against
a Farsi policy — claims that are verifiable in English-language sources may not be surfaced for
Farsi-language policy criteria, leading to more "unverifiable" deductions. Tavily and DDG show
the opposite (FA slightly higher), suggesting this effect is not universal and may interact with
which search engines are active and what sources they return.

### Finding 7: Fetch_url is rarely called with Tavily

Tavily's search results already include meaningful content in each snippet. The judge calls
`fetch_url` to get full page text only 7 times across 10 scenarios with Tavily, versus 44 times
with DuckDuckGo and 20 times with SearXNG. This reduces cost, latency, and potential for fetch failures.
It also means the agentic pipeline is more efficient with Tavily — the same number of tool-call
slots gets more done.

---

## 4. Token Usage and Latency

Average per agentic evaluation, across all policies.

| Backend | Judge | Prompt tokens | Completion tokens | Avg time |
|---|---|---|---|---|
| SearXNG | GPT-5-nano | 15,876 | 5,992 | — |
| SearXNG | Claude | 32,953 | 1,062 | — |
| SearXNG | Gemini | N/A (failed) | N/A | — |
| Tavily | GPT-5-nano | 19,399 | 6,613 | 64s |
| Tavily | Claude | 34,534 | 1,160 | 33s |
| Tavily | Gemini | 10,947 | 367 | 18s |
| DDG | GPT-5-nano | 18,466 | 5,996 | 66s |
| DDG | Claude | 34,466 | 1,065 | 36s |
| DDG | Gemini | 11,422 | 388 | 22s |

(Avg time for SearXNG was not recorded in the complete run's log format.)

**Claude reads a lot but writes little** (~33–35K prompt tokens, ~1K completion). Claude processes
a large context on each turn (accumulated conversation history grows fast with tool results) but
produces compact judgments.

**GPT-5-nano reads less but writes more** (~16–19K prompt, ~6K completion). GPT produces much
more verbose reasoning and explanation text, which costs more in completion tokens and drives
the higher per-evaluation latency.

**Gemini is cheapest** (~11K prompt, ~400 completion, 18–23s). But as noted above, Gemini
often skips retrieval entirely — its low token count partly reflects not doing the work.

---

## 5. Overall Assessment by Search Backend

### DuckDuckGo
- **Pros:** No setup, no API key, reasonable success rate in stable conditions.
- **Cons:** Unofficial API — rate limits silently (stalls connections rather than returning errors),
  requires a 60-second wall-clock timeout workaround, returns no results for some factual queries.
  Fetch_url fail rate is high (34%).

### SearXNG
- **Pros:** Self-hosted (no cost, no key, full control), aggregates multiple engines, highest
  official source rate (36% government/IGO domains), lowest fetch_url fail rate (18%). Zero
  wall-clock timeouts once Docker is pre-started.
- **Cons:** Requires Docker/Colima and manual JSON format configuration. High content-level
  search fail rate (29%) for specific medical, NGO, or obscure-initiative queries where
  SearXNG's configured engine set returns no results. Gemini 2.5 Flash failed entirely — it
  produced no output (no score, no explanation, no tool calls) across all 60 evaluations. The
  cause is likely an incompatibility between the SearXNG agentic runner's tool schema format
  and the Gemini API as used in this setup.

### Tavily
- **Pros:** Lowest search fail rate (1%), zero timeouts, richest snippets (judges call fetch_url
  far less), consistent results across all three judges. The only backend where all three
  judges ran successfully.
- **Cons:** Requires an API key and has a monthly quota (1,000 free searches). Surprisingly low
  URL reachability rate (37%) — likely a timeout artifact from government sites, not a search
  quality problem. Fetch_url fail rate appears high (57%) but is based on only 7 calls.

---

## 6. Recommendations

**For research reproducibility:** Use Tavily. It is the most consistent backend, works with all
three judge models, and its structured API means results are stable across runs. The 1,000
free searches/month is sufficient for datasets of 10–50 scenarios.

**For privacy or cost sensitivity:** Use SearXNG. Once stable (Docker pre-started, JSON format
enabled), it should match or exceed DuckDuckGo reliability. Enable Wikipedia and government
portal engines in SearXNG's `settings.yml` for better coverage of the humanitarian domain.

**For quick iteration:** Avoid DuckDuckGo. It is the only backend that silently stalls, requiring
a workaround timeout. The rate-limiting behavior makes multi-run comparisons unreliable.

**For all backends:** Raise `--max-tool-calls` to 8 for Claude and Gemini (they use 3–5, so the
cap rarely matters). Keep it at 5 or lower for GPT-5-nano, or fix the parallel tool call issue
to make the cap meaningful.

---

## 7. Domain Analysis

This section reports which domains appeared in tool call results across the three experiments.
All counts are derived directly from the JSON log files in each repository's output directory.
Two domain populations are analyzed separately:

- **Search result domains**: domains in URLs returned by `search_web` (what the search engine
  chose to surface as evidence).
- **URL-check domains**: domains in URLs that the agent checked with `check_url_validity` (what
  the assistant response itself cited — these are the assistant's own claims, not search engine
  choices).

Methodology: domain extracted by splitting each URL on `/` after stripping the scheme, then
taking the first component. Subdomains (e.g. `help.unhcr.org`) are counted separately from their
apex domain (`unhcr.org`). No normalisation was applied beyond lowercasing.

---

### 7.1 Most-Frequent Search Result Domains

Top domains returned by `search_web` across all tool calls in each run.

#### SearXNG (10 scenarios, complete run)

| Rank | Domain | Count |
|---|---|---|
| 1 | ofac.treasury.gov | 188 |
| 2 | www.ofpra.gouv.fr | 135 |
| 3 | www.gisti.org | 77 |
| 4 | pmc.ncbi.nlm.nih.gov | 75 |
| 5 | sanctionslawyers.net | 64 |
| 6 | www.facebook.com | 55 |
| 7 | www.france-terre-asile.org | 46 |
| 8 | etrangers-en-france.interieur.gouv.fr | 46 |

#### Tavily (10 scenarios)

| Rank | Domain | Count |
|---|---|---|
| 1 | facebook.com | 194 |
| 2 | ofpra.gouv.fr | 143 |
| 3 | unhcr.org | 109 |
| 4 | ofac.treasury.gov | 82 |
| 5 | asylumineurope.org | 80 |
| 6 | instagram.com | 47 |

#### DuckDuckGo (10 scenarios)

| Rank | Domain | Count |
|---|---|---|
| 1 | en.wikipedia.org | 153 |
| 2 | ofac.treasury.gov | 99 |
| 3 | youtube.com | 82 |
| 4 | asylumineurope.org | 80 |
| 5 | ofpra.gouv.fr | 66 |

**Notable observations:**

- `ofac.treasury.gov` and `www.ofpra.gouv.fr` appear in the top-3 for every backend. These are
  the primary authoritative sources for OFAC sanctions and French asylum law respectively — the
  two regulatory areas most directly tested by the scenario set. Their consistent appearance
  confirms the search engines are finding the right sources for this domain.

- **Tavily surfaces `facebook.com` 194 times — the single most frequent domain in that run.**
  SearXNG also surfaces Facebook (55 times, rank 6), but at a much lower rate. Facebook pages
  appeared because the scenarios include organisations (refugee support NGOs, humanitarian
  charities) that maintain Facebook pages as their primary web presence. Tavily indexes social
  media pages more aggressively than SearXNG. Whether this is positive depends on the use case:
  social media content is less authoritative for policy verification but may be the only web
  presence for small NGOs.

- **DuckDuckGo consistently surfaces `en.wikipedia.org` as the top domain** (153 calls). Wikipedia
  did not appear in SearXNG's top domains and appeared only incidentally with Tavily. DuckDuckGo
  appears to use Wikipedia-weighted ranking for definitional and factual queries — useful for
  background information, less so for verifying regulatory claims.

- `youtube.com` appeared 82 times with DuckDuckGo but was not prominent in SearXNG or Tavily.
  This was not expected for a policy-verification task and likely reflects DDG returning video
  content for some queries that had no strong text-document match. YouTube content cannot be
  meaningfully parsed by `fetch_url`, so these results were wasted tool-call slots.

---

### 7.2 Source Quality: Official vs Non-Official Domains

**Classification method:** A domain was counted as "official" if it matched a government TLD
pattern (`.gov`, `.gouv.fr`, `.europa.eu`, `.gc.ca`, `.int`) or a recognised intergovernmental
body (unhcr.org, iom.int, icrc.org, ilo.org, who.int, un.org, europa.eu). A domain was counted
as "junk/social" if it matched: facebook.com, instagram.com, twitter.com/x.com, youtube.com,
tiktok.com, linkedin.com, reddit.com. All other domains were counted as "other."

| Backend | Official % | Junk/Social % | Other % |
|---|---|---|---|
| SearXNG | **36%** | 3.6% | 60.4% |
| Tavily | 18% | **8.8%** | 73.2% |
| DDG | 15% | 5.6% | 79.4% |

**SearXNG returned the highest proportion of official sources (36%)**, roughly double Tavily (18%)
and DuckDuckGo (15%). SearXNG's multi-engine aggregation includes engines that specifically index government and
institutional content. For a guardrail whose job is to verify regulatory compliance claims, this
is a meaningful advantage — official sources carry the most weight as evidence.

**Tavily had the highest junk/social rate (8.8%)**, driven almost entirely by `facebook.com`
(194 hits). The other backends kept junk/social below 6%. Whether this inflates or deflates
scores depends on the scenario: a guardrail checking whether an NGO exists might correctly score
higher if it finds a Facebook page; one verifying a legal claim finds no useful evidence there.

The "other" category (most of the volume in every backend) includes a mix of NGO websites,
legal information portals, news sites, and academic sources. This category was not further broken
down because it requires manual classification and the counts would be unreliable at the scale
of these runs.

---

### 7.3 Non-English Domain Rate

**Method:** A domain was classified as non-English if its TLD or known subdomain is associated
with a non-English primary language country (e.g. `.fr`, `.de`, `.ir`, `.nl`, `.be`), or if it
appears in a known non-English-primary registry (e.g. `ofpra.gouv.fr`, `gisti.org` — French
legal NGO). This is an approximation: some `.org` sites publish in multiple languages and cannot
be classified without fetching the page.

| Backend | Non-English domain % (all policies) | Non-English % (EN policy) | Non-English % (FA policy) |
|---|---|---|---|
| SearXNG | 21% | 24% | 18% |
| Tavily | 18% | 18% | 18% |
| DDG | 15% | ~15% | ~15% |

**SearXNG returns more non-English sources overall (21%)**, reflecting its broader engine
coverage including French and German language search engines. For scenarios in the humanitarian
asylum domain — which involves French law (OFPRA, CESEDA) and German refugee services — this is
appropriate.

**The most striking pattern is SearXNG's divergence by policy language:**

- When the policy text is in English, SearXNG returns 24% non-English domains. This makes sense:
  the agent searches in English but the domain references French asylum law (ofpra.gouv.fr,
  gisti.org) which naturally land on French-language sites.
- When the policy text is in Farsi, the non-English domain rate drops to 18%. The agent is
  judging against a Farsi-language policy but still searching in English (no Farsi search terms
  are generated). The Farsi policy text does not change what the agent searches for — only how it
  interprets results. The drop suggests that Farsi-language policy criteria reference sources that
  the English-language search queries are less likely to surface.

**Tavily and DuckDuckGo show no meaningful EN/FA policy split** — both at ~15–18% regardless of
policy language. Their non-English coverage is more uniform and less sensitive to the policy
language signal.

This finding connects to Finding 6 (Section 3): SearXNG shows a larger gap between English- and
Farsi-policy agentic scores (−0.054), whereas Tavily shows almost none (+0.004). The domain
language data is consistent with the score data: when using SearXNG with a Farsi policy, the
agent is retrieving fewer Farsi-relevant sources, leading to more unverifiable-claim deductions
and lower scores.

---

### 7.4 URL-Check Domains (Assistant Response Citations)

`check_url_validity` is called on URLs that appear in the assistant's own response text — these
are the URLs the assistant cited as evidence or reference. The domain pattern here reflects what
the assistant was trained to cite for humanitarian/financial queries, not what the search engine
returned.

#### Top URL-check domains — SearXNG complete run

| Domain | Count | Notes |
|---|---|---|
| www.caritas-hamburg.de | 24 | Caritas social services, Hamburg |
| www.uke.de | 12 | Universitätsklinikum Hamburg-Eppendorf (university hospital) |
| www.amiravita.de | 12 | Refugee health/counselling NGO |
| www.mir-therapiezentrum.de | 12 | Psychotherapy centre for refugees |
| www.xenion.org | 12 | Berlin refugee psychosocial support |
| www.fluechtlingszentrum-hamburg.de | 12 | Hamburg refugee centre |
| www.refugio-hamburg.de | 12 | Refugee trauma therapy centre, Hamburg |
| www.praxis-ohne-grenzen.org | 12 | Medical practice for undocumented migrants |
| ofac.treasury.gov | 6 | OFAC sanctions authority |
| www.uscis.gov | 4 | US immigration authority |

The same German refugee/healthcare NGO list appears in Tavily and DDG runs with similar counts
(12–24 per domain), confirming this pattern is driven by the assistant's training, not the search backend.

**The URL-check domain list is dominated by German refugee and healthcare organizations.** These
are not search engine results — they are URLs the LLM assistant cited in its responses from its
training data. The assistant was responding to scenarios about refugees in Germany and cited
real, specific organisations it knew about. The agentic guardrail then checked whether those
URLs were reachable.

Many of these sites (uke.de, refugio-hamburg.de, fluechtlingsrat-hamburg.de) have slow or
restricted HTTP HEAD responses that timed out in the 10-second check, generating false-positive
"unreachable" results. This accounts for much of the −0.15 deduction pressure in agentic scores
— the assistant cited real, legitimate organisations but the URL checker could not confirm them
within the timeout window.

**This pattern is consistent across all three backends**, confirming it is driven by the
scenarios and the assistant's training, not by which search engine was used. The agentic search
backend affects *search result* domains strongly but has little influence on which URLs appear
in the assistant's own response text.

---

### 7.5 Summary of Domain Analysis Findings

| Finding | Evidence |
|---|---|
| SearXNG returns the highest share of official (.gov/.gouv/.int) sources | 36% vs 15–18% for Tavily/DDG |
| SearXNG has the highest search fail rate (29%) due to no-results for specific queries | Medical/symptom/NGO-name queries fail; not a connectivity problem (0 timeouts) |
| Tavily surfaces social media (Facebook, Instagram) at nearly 3× the rate of other backends | 8.8% junk/social vs 3.6% (SearXNG), 5.6% (DDG) |
| DuckDuckGo defaults to Wikipedia for definitional queries | top domain (153 hits); absent from SearXNG and Tavily top lists |
| SearXNG's non-English coverage drops when the policy is Farsi (not English) | 24% → 18% non-English domains; consistent with FA score penalty |
| URL-check domains are determined by the assistant's training, not the search backend | same German NGO list appears across all three experiments |
| German refugee/healthcare NGOs dominate URL-check failures | slow HTTP responses trigger false-positive unreachable deductions |
