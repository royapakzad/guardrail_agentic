import "server-only";
import type { UseCase } from "@/lib/types";
import { SEED_DATASET_ID, type DatasetId } from "@/lib/adapters";
import { listDatasets } from "@/lib/db/queries";

/** Default dataset for a use case when the URL doesn't specify one: the most
 * recent upload if any exist, else the bundled seed sample. Falls back to
 * seed on any DB error too (e.g. no database connected yet in local dev) —
 * the app should never hard-fail just because uploads aren't configured. */
export async function resolveDefaultDatasetId(useCase: UseCase): Promise<DatasetId> {
  try {
    const datasets = await listDatasets(useCase);
    return datasets.length > 0 ? datasets[0].id : SEED_DATASET_ID;
  } catch {
    return SEED_DATASET_ID;
  }
}

/** Parses the `?dataset=` search param into a DatasetId, resolving to the
 * default (most recent upload, else seed) when absent. */
export async function resolveDatasetIdParam(useCase: UseCase, param: string | undefined): Promise<DatasetId> {
  if (!param || param === SEED_DATASET_ID) {
    return param === SEED_DATASET_ID ? SEED_DATASET_ID : resolveDefaultDatasetId(useCase);
  }
  const asNumber = Number(param);
  return Number.isFinite(asNumber) ? asNumber : resolveDefaultDatasetId(useCase);
}
