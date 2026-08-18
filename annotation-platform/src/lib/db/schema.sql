-- Run once against the connected Postgres database (Vercel Postgres / Neon).
-- Evaluation data itself is never stored here — annotator-generated data, plus
-- metadata about uploaded result files (the file content lives in Vercel Blob,
-- not here — see src/lib/adapters/index.ts::getRecordsForDataset).

CREATE TABLE IF NOT EXISTS annotations (
  id SERIAL PRIMARY KEY,
  scenario_id TEXT NOT NULL,
  use_case TEXT NOT NULL,
  language TEXT NOT NULL,
  policy_label TEXT NOT NULL,
  annotator_name TEXT NOT NULL,
  evidence_source_type TEXT,
  deduction_reason_category TEXT,
  judgment_alignment_en TEXT,
  alignment_explanation_en TEXT,
  judgment_alignment_non_en TEXT,
  alignment_explanation_non_en TEXT,
  free_text TEXT,
  confidence TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS annotations_scenario_idx ON annotations (use_case, scenario_id);

CREATE TABLE IF NOT EXISTS gold_labels (
  id SERIAL PRIMARY KEY,
  scenario_id TEXT NOT NULL,
  use_case TEXT NOT NULL,
  language TEXT NOT NULL,
  gold_verdict TEXT NOT NULL,
  gold_notes TEXT,
  created_by TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (use_case, scenario_id, language)
);

CREATE TABLE IF NOT EXISTS datasets (
  id SERIAL PRIMARY KEY,
  use_case TEXT NOT NULL,
  filename TEXT NOT NULL,
  blob_url TEXT NOT NULL,
  uploaded_by TEXT NOT NULL,
  record_count INTEGER NOT NULL,
  uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS datasets_use_case_idx ON datasets (use_case, uploaded_at DESC);

-- Qualitative coding / thematic analysis (Issue #57) -----------------------
-- Standard HCI/CSCW thematic-analysis workflow: a shared, evolving codebook
-- (codes grouped into themes) that annotators apply to scenario text.
-- Multiple annotators can independently code the same scenario/field, which
-- is what the method needs for later reconciliation -- this schema doesn't
-- compute inter-coder agreement itself (see Issue #57's fast-follow note).

CREATE TABLE IF NOT EXISTS codebook_codes (
  id SERIAL PRIMARY KEY,
  use_case TEXT NOT NULL,
  name TEXT NOT NULL,
  definition TEXT NOT NULL,
  example_quote TEXT,
  theme TEXT,
  created_by TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (use_case, name)
);

CREATE INDEX IF NOT EXISTS codebook_codes_use_case_idx ON codebook_codes (use_case, theme);

CREATE TABLE IF NOT EXISTS code_applications (
  id SERIAL PRIMARY KEY,
  scenario_id TEXT NOT NULL,
  use_case TEXT NOT NULL,
  language TEXT NOT NULL,
  policy_label TEXT NOT NULL,
  annotator_name TEXT NOT NULL,
  code_id INTEGER NOT NULL REFERENCES codebook_codes(id) ON DELETE CASCADE,
  target_field TEXT NOT NULL,
  quote_text TEXT,
  note TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS code_applications_scenario_idx ON code_applications (use_case, scenario_id);
CREATE INDEX IF NOT EXISTS code_applications_code_idx ON code_applications (code_id);

-- Edit tracking for the platform UX redesign (annotations/code_applications
-- gain edit/delete UI; codebook_codes already had updated_at from the start).
-- Additive and idempotent -- safe to re-run against an existing database.
ALTER TABLE annotations ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();
ALTER TABLE code_applications ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();

-- Part 4 "Your review" redesign: replaced the agree/disagree-with-verdict and
-- evidentiary-attribution fields with a judgment on whether the annotator's
-- own read is more aligned with the agentic or non-agentic guardrail --
-- recorded once per language (English / non-English), since a scenario page
-- shows both language variants side by side and the annotator's read can
-- differ between them. Destructive -- drops any previously recorded answers
-- for the fields it replaces, including the single-language version of this
-- same judgment added earlier in the same redesign.
ALTER TABLE annotations ADD COLUMN IF NOT EXISTS judgment_alignment_en TEXT;
ALTER TABLE annotations ADD COLUMN IF NOT EXISTS alignment_explanation_en TEXT;
ALTER TABLE annotations ADD COLUMN IF NOT EXISTS judgment_alignment_non_en TEXT;
ALTER TABLE annotations ADD COLUMN IF NOT EXISTS alignment_explanation_non_en TEXT;
ALTER TABLE annotations DROP COLUMN IF EXISTS agrees_with_verdict;
ALTER TABLE annotations DROP COLUMN IF EXISTS disagreement_reason;
ALTER TABLE annotations DROP COLUMN IF EXISTS evidentiary_attribution_present;
ALTER TABLE annotations DROP COLUMN IF EXISTS judgment_alignment;
ALTER TABLE annotations DROP COLUMN IF EXISTS alignment_explanation;

-- Scope annotations/code_applications to the dataset they were made against
-- (bug fix). Before this, a saved review was keyed only by
-- (use_case, scenario_id, language, policy_label, annotator_name) -- with no
-- dataset in that key at all, uploading a NEW dataset that happens to reuse
-- the same scenario id and policy label (extremely common: every re-run of
-- the same scenario set produces the same ids) surfaced the OLD dataset's
-- saved annotations and qualitative codes on the new dataset's scenario page,
-- even though the underlying response/judge output could be completely
-- different now. TEXT, not an INTEGER FK to datasets(id), because a
-- DatasetId is either a real dataset row's id or the literal string "seed"
-- for the bundled sample data (see lib/datasetId.ts).
--
-- Existing rows get NULL here (we don't know what dataset they were made
-- against) and, going forward, deliberately fall out of the normal
-- dataset-scoped list queries -- surfacing them under an unrelated dataset
-- would just be re-committing the same bug, so this intentionally drops
-- visibility on old, unattributable reviews rather than guessing.
ALTER TABLE annotations ADD COLUMN IF NOT EXISTS dataset_id TEXT;
ALTER TABLE code_applications ADD COLUMN IF NOT EXISTS dataset_id TEXT;
CREATE INDEX IF NOT EXISTS annotations_dataset_idx ON annotations (use_case, scenario_id, dataset_id);
CREATE INDEX IF NOT EXISTS code_applications_dataset_idx ON code_applications (use_case, scenario_id, dataset_id);
