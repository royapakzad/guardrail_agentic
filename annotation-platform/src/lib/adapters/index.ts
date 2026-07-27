import "server-only";
import type { EvaluationRecord, UseCase } from "@/lib/types";
import { adaptHumanitarian } from "./humanitarian";
import { adaptFinancial } from "./financial";
import { adaptCybersecurity } from "./cybersecurity";
import { getDataset } from "@/lib/db/queries";
import { SEED_DATASET_ID, type DatasetId } from "@/lib/datasetId";

export { SEED_DATASET_ID, type DatasetId } from "@/lib/datasetId";

const ADAPTERS: Record<UseCase, (raw: unknown) => EvaluationRecord[]> = {
  humanitarian: adaptHumanitarian,
  financial: adaptFinancial,
  cybersecurity: adaptCybersecurity,
};

export const USE_CASES: UseCase[] = ["humanitarian", "financial", "cybersecurity"];

// No bundled sample data ships with the app -- every use case starts empty
// until a real batch run is uploaded. SEED_DATASET_ID is kept as the "no
// dataset selected yet" sentinel (see datasetSelection.ts) so an unset
// ?dataset= param still resolves to something rather than throwing.
function loadSeedRecords(): EvaluationRecord[] {
  return [];
}

/** Runs raw JSON through the right use case's adapter — used both by the seed
 * loader below and by the upload route to validate a file before storing it,
 * so "does this file parse" and "will this file render" are one check. */
export function adaptRawJson(useCase: UseCase, raw: unknown): EvaluationRecord[] {
  return ADAPTERS[useCase](raw);
}

export function getRecordCountForRawJson(useCase: UseCase, raw: unknown): number {
  return adaptRawJson(useCase, raw).length;
}

// Per-request-lifecycle cache — avoids re-fetching/re-parsing the same
// dataset twice within one page render (e.g. dashboard + nav both need it).
const cache = new Map<string, EvaluationRecord[]>();

export async function getRecordsForDataset(useCase: UseCase, datasetId: DatasetId): Promise<EvaluationRecord[]> {
  const cacheKey = `${useCase}:${datasetId}`;
  const cached = cache.get(cacheKey);
  if (cached) return cached;

  let records: EvaluationRecord[];
  if (datasetId === SEED_DATASET_ID) {
    records = loadSeedRecords();
  } else {
    const dataset = await getDataset(datasetId);
    if (!dataset || dataset.use_case !== useCase) {
      throw new Error(`Dataset ${datasetId} not found for use case ${useCase}`);
    }
    const res = await fetch(dataset.blob_url);
    if (!res.ok) throw new Error(`Could not fetch dataset ${datasetId} (HTTP ${res.status})`);
    const raw = await res.json();
    records = adaptRawJson(useCase, raw);
  }

  cache.set(cacheKey, records);
  return records;
}

/** The same scenario id can appear once per language (e.g. "IR01" in both the
 * en and fa result sets) — language disambiguates which record is meant. When
 * omitted, returns the first match. */
export async function getRecordByIdForDataset(
  useCase: UseCase,
  datasetId: DatasetId,
  id: string,
  language?: string
): Promise<EvaluationRecord | undefined> {
  const records = await getRecordsForDataset(useCase, datasetId);
  if (language) return records.find((r) => r.id === id && r.language === language);
  return records.find((r) => r.id === id);
}
