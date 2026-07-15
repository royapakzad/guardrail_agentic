# Guardrail Annotation Platform

Hosted dashboard + annotation tool for guardrail evaluation results (Issue #41). Reads existing
`outputs/*.json` evaluation data (curated into `data/`), computes quantitative metrics
(score delta, flip rate, tool usage, latency, tokens, acronym checks, language flips), and lets
annotators submit structured qualitative findings against a fixed codebook, persisted in Postgres.

See `analysis_improvemnets/live_scenario_qualitative_quantitative_analysis.md` in the parent repo
for the research framing (RQs, the reflexive/codebook-TA methodology, §8.8's language-dimension
design) this app's metrics and annotation categories are built from.

## Data

`data/{humanitarian,financial,cybersecurity}/results.json` are **committed** (deliberately
overriding the parent repo's `.gitignore` for `outputs/`) — curated snapshots, not the full raw
`outputs/` directory. Each use case's raw schema is different (see `src/lib/adapters/`); a small
adapter per use case normalizes all three into one canonical `EvaluationRecord` shape before
anything else touches the data.

**Known gap:** cybersecurity currently has only 1 scenario logged (`outputs/three_domain_scenarios_v2/`)
versus 50 for humanitarian and 20 for financial. The dashboard flags this with a small-sample
warning rather than hiding it. Re-run `annotation-platform`'s data curation step (copy the relevant
files into `data/<use-case>/results.json`, merging cybersecurity's per-scenario dict files into a
list first) once a full cybersecurity batch run exists.

## Local development

```bash
npm install
npm run dev
```

Dashboards and scenario browsing work with no database configured. The annotation form's `POST
/api/annotations` (and gold-label endpoints) need `DATABASE_URL` set to a real Neon/Vercel Postgres
connection string — `@neondatabase/serverless`'s HTTP client only speaks to actual Neon endpoints,
not a plain local Postgres. To verify the SQL itself locally, apply `src/lib/db/schema.sql` to any
local Postgres and run the queries in `src/lib/db/queries.ts` by hand — that's how this schema was
verified during development.

## Deploy

1. Push this repo to GitHub (or point an existing Vercel project's **Root Directory** setting at
   `annotation-platform/` if deploying from the parent monorepo).
2. In Vercel: connect a Postgres database (Neon, via the Vercel Marketplace integration) — this
   sets `DATABASE_URL` automatically.
3. Run `src/lib/db/schema.sql` against the connected database once (Vercel's Neon integration
   dashboard has a SQL editor, or connect with `psql`).
4. Deploy. No auth is configured (name-field-only, per Issue #41's approved default) — keep the
   deployment URL unlisted if that matters for your use.

## Architecture notes

- `src/lib/adapters/` — one pure function per use case, `raw JSON → EvaluationRecord[]`. Ports two
  known pipeline quirks from the parent repo's `visualize_results.py`: recomputing `valid` from
  `score > 0.6` rather than trusting the stored field, and defensively JSON-parsing list-shaped
  fields in case they arrive as encoded strings.
- `src/lib/metrics/` — pure functions over `EvaluationRecord[]`, grouped by policy-variant label.
  `toolSelectionConsistency.ts` and `acronymChecks.ts` are explicitly descriptive/heuristic (no gold
  labels exist for this batch data) — see their file-level comments before treating their numbers
  as ground truth.
- `src/lib/db/` — two tables only (`annotations`, `gold_labels`); evaluation data itself is never
  duplicated into Postgres.
