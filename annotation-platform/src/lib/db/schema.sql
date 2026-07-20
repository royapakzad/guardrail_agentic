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
  agrees_with_verdict BOOLEAN,
  disagreement_reason TEXT,
  evidence_source_type TEXT,
  deduction_reason_category TEXT,
  evidentiary_attribution_present BOOLEAN,
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
