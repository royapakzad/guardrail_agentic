import "server-only";
import { getSql } from "./client";
import type { UseCase } from "@/lib/types";

export type Annotation = {
  id: number;
  scenario_id: string;
  use_case: string;
  language: string;
  policy_label: string;
  annotator_name: string;
  agrees_with_verdict: boolean | null;
  disagreement_reason: string | null;
  evidence_source_type: string | null;
  deduction_reason_category: string | null;
  evidentiary_attribution_present: boolean | null;
  free_text: string | null;
  confidence: string | null;
  created_at: string;
};

export type NewAnnotation = {
  scenarioId: string;
  useCase: UseCase;
  language: string;
  policyLabel: string;
  annotatorName: string;
  agreesWithVerdict: boolean | null;
  disagreementReason: string | null;
  evidenceSourceType: string | null;
  deductionReasonCategory: string | null;
  evidentiaryAttributionPresent: boolean | null;
  freeText: string | null;
  confidence: string | null;
};

export async function insertAnnotation(input: NewAnnotation): Promise<Annotation> {
  const sql = getSql();
  const rows = await sql`
    INSERT INTO annotations (
      scenario_id, use_case, language, policy_label, annotator_name,
      agrees_with_verdict, disagreement_reason, evidence_source_type,
      deduction_reason_category, evidentiary_attribution_present, free_text, confidence
    ) VALUES (
      ${input.scenarioId}, ${input.useCase}, ${input.language}, ${input.policyLabel}, ${input.annotatorName},
      ${input.agreesWithVerdict}, ${input.disagreementReason}, ${input.evidenceSourceType},
      ${input.deductionReasonCategory}, ${input.evidentiaryAttributionPresent}, ${input.freeText}, ${input.confidence}
    )
    RETURNING *
  `;
  return rows[0] as Annotation;
}

export async function listAnnotations(useCase: UseCase, scenarioId: string): Promise<Annotation[]> {
  const sql = getSql();
  const rows = await sql`
    SELECT * FROM annotations
    WHERE use_case = ${useCase} AND scenario_id = ${scenarioId}
    ORDER BY created_at DESC
  `;
  return rows as Annotation[];
}

export type GoldLabel = {
  id: number;
  scenario_id: string;
  use_case: string;
  language: string;
  gold_verdict: string;
  gold_notes: string | null;
  created_by: string;
  created_at: string;
  updated_at: string;
};

export type NewGoldLabel = {
  scenarioId: string;
  useCase: UseCase;
  language: string;
  goldVerdict: string;
  goldNotes: string | null;
  createdBy: string;
};

export async function upsertGoldLabel(input: NewGoldLabel): Promise<GoldLabel> {
  const sql = getSql();
  const rows = await sql`
    INSERT INTO gold_labels (scenario_id, use_case, language, gold_verdict, gold_notes, created_by)
    VALUES (${input.scenarioId}, ${input.useCase}, ${input.language}, ${input.goldVerdict}, ${input.goldNotes}, ${input.createdBy})
    ON CONFLICT (use_case, scenario_id, language)
    DO UPDATE SET gold_verdict = EXCLUDED.gold_verdict, gold_notes = EXCLUDED.gold_notes, updated_at = now()
    RETURNING *
  `;
  return rows[0] as GoldLabel;
}

export async function getGoldLabel(useCase: UseCase, scenarioId: string, language: string): Promise<GoldLabel | null> {
  const sql = getSql();
  const rows = await sql`
    SELECT * FROM gold_labels
    WHERE use_case = ${useCase} AND scenario_id = ${scenarioId} AND language = ${language}
    LIMIT 1
  `;
  return (rows[0] as GoldLabel) ?? null;
}

export type Dataset = {
  id: number;
  use_case: string;
  filename: string;
  blob_url: string;
  uploaded_by: string;
  record_count: number;
  uploaded_at: string;
};

export type NewDataset = {
  useCase: UseCase;
  filename: string;
  blobUrl: string;
  uploadedBy: string;
  recordCount: number;
};

export async function insertDataset(input: NewDataset): Promise<Dataset> {
  const sql = getSql();
  const rows = await sql`
    INSERT INTO datasets (use_case, filename, blob_url, uploaded_by, record_count)
    VALUES (${input.useCase}, ${input.filename}, ${input.blobUrl}, ${input.uploadedBy}, ${input.recordCount})
    RETURNING *
  `;
  return rows[0] as Dataset;
}

export async function listDatasets(useCase: UseCase): Promise<Dataset[]> {
  const sql = getSql();
  const rows = await sql`
    SELECT * FROM datasets
    WHERE use_case = ${useCase}
    ORDER BY uploaded_at DESC
  `;
  return rows as Dataset[];
}

export async function getDataset(id: number): Promise<Dataset | null> {
  const sql = getSql();
  const rows = await sql`
    SELECT * FROM datasets WHERE id = ${id} LIMIT 1
  `;
  return (rows[0] as Dataset) ?? null;
}

// ── Qualitative coding / thematic analysis (Issue #57) ─────────────────────

export type CodebookCode = {
  id: number;
  use_case: string;
  name: string;
  definition: string;
  example_quote: string | null;
  theme: string | null;
  created_by: string;
  created_at: string;
  updated_at: string;
};

export type NewCodebookCode = {
  useCase: UseCase;
  name: string;
  definition: string;
  exampleQuote: string | null;
  theme: string | null;
  createdBy: string;
};

export async function insertCodebookCode(input: NewCodebookCode): Promise<CodebookCode> {
  const sql = getSql();
  const rows = await sql`
    INSERT INTO codebook_codes (use_case, name, definition, example_quote, theme, created_by)
    VALUES (${input.useCase}, ${input.name}, ${input.definition}, ${input.exampleQuote}, ${input.theme}, ${input.createdBy})
    RETURNING *
  `;
  return rows[0] as CodebookCode;
}

export async function listCodebookCodes(useCase: UseCase): Promise<CodebookCode[]> {
  const sql = getSql();
  const rows = await sql`
    SELECT * FROM codebook_codes
    WHERE use_case = ${useCase}
    ORDER BY theme NULLS LAST, name
  `;
  return rows as CodebookCode[];
}

export type UpdateCodebookCode = {
  id: number;
  definition?: string;
  exampleQuote?: string | null;
  theme?: string | null;
};

export async function updateCodebookCode(input: UpdateCodebookCode): Promise<CodebookCode | null> {
  const sql = getSql();
  const rows = await sql`
    UPDATE codebook_codes
    SET
      definition = COALESCE(${input.definition ?? null}, definition),
      example_quote = CASE WHEN ${input.exampleQuote !== undefined} THEN ${input.exampleQuote ?? null} ELSE example_quote END,
      theme = CASE WHEN ${input.theme !== undefined} THEN ${input.theme ?? null} ELSE theme END,
      updated_at = now()
    WHERE id = ${input.id}
    RETURNING *
  `;
  return (rows[0] as CodebookCode) ?? null;
}

export type CodeApplication = {
  id: number;
  scenario_id: string;
  use_case: string;
  language: string;
  policy_label: string;
  annotator_name: string;
  code_id: number;
  target_field: string;
  quote_text: string | null;
  note: string | null;
  created_at: string;
};

export type CodeApplicationWithCode = CodeApplication & {
  code_name: string;
  code_theme: string | null;
};

export type NewCodeApplication = {
  scenarioId: string;
  useCase: UseCase;
  language: string;
  policyLabel: string;
  annotatorName: string;
  codeId: number;
  targetField: string;
  quoteText: string | null;
  note: string | null;
};

export async function insertCodeApplication(input: NewCodeApplication): Promise<CodeApplication> {
  const sql = getSql();
  const rows = await sql`
    INSERT INTO code_applications (
      scenario_id, use_case, language, policy_label, annotator_name,
      code_id, target_field, quote_text, note
    ) VALUES (
      ${input.scenarioId}, ${input.useCase}, ${input.language}, ${input.policyLabel}, ${input.annotatorName},
      ${input.codeId}, ${input.targetField}, ${input.quoteText}, ${input.note}
    )
    RETURNING *
  `;
  return rows[0] as CodeApplication;
}

export async function listCodeApplications(useCase: UseCase, scenarioId: string): Promise<CodeApplicationWithCode[]> {
  const sql = getSql();
  const rows = await sql`
    SELECT ca.*, cc.name AS code_name, cc.theme AS code_theme
    FROM code_applications ca
    JOIN codebook_codes cc ON cc.id = ca.code_id
    WHERE ca.use_case = ${useCase} AND ca.scenario_id = ${scenarioId}
    ORDER BY ca.created_at DESC
  `;
  return rows as CodeApplicationWithCode[];
}

export async function listCodeApplicationsForUseCase(useCase: UseCase): Promise<CodeApplicationWithCode[]> {
  const sql = getSql();
  const rows = await sql`
    SELECT ca.*, cc.name AS code_name, cc.theme AS code_theme
    FROM code_applications ca
    JOIN codebook_codes cc ON cc.id = ca.code_id
    WHERE ca.use_case = ${useCase}
    ORDER BY ca.created_at DESC
  `;
  return rows as CodeApplicationWithCode[];
}
