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
