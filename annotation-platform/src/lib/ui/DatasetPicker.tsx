"use client";

import { useRouter } from "next/navigation";
import Link from "next/link";
import type { UseCase } from "@/lib/types";
import { SEED_DATASET_ID } from "@/lib/datasetId";

// Deliberately not importing Dataset's type from lib/db/queries (server-only
// module) — a minimal local shape keeps this client component's import graph
// free of the Postgres client. See lib/datasetId.ts for why.
type DatasetOption = { id: number; filename: string; uploaded_at: string; record_count: number; uploaded_by: string };

type Props = {
  useCase: UseCase;
  datasets: DatasetOption[];
  currentId: string; // "seed" or a numeric dataset id, as a string
  basePath: string; // e.g. `/${useCase}` or `/${useCase}/scenarios`
};

export function DatasetPicker({ useCase, datasets, currentId, basePath }: Props) {
  const router = useRouter();

  return (
    <div className="flex items-center gap-3 text-sm">
      <label className="text-slate-600">Dataset:</label>
      <select
        value={currentId}
        onChange={(e) => router.push(`${basePath}?dataset=${e.target.value}`)}
        className="rounded border border-slate-300 px-2 py-1 text-sm bg-white"
      >
        {datasets.map((d) => (
          <option key={d.id} value={d.id}>
            {d.filename} — {new Date(d.uploaded_at).toLocaleDateString()} ({d.record_count} scenarios, by {d.uploaded_by})
          </option>
        ))}
        <option value={SEED_DATASET_ID}>Sample data (bundled)</option>
      </select>
      <Link href={`/${useCase}/upload`} className="text-slate-500 underline hover:text-slate-800">
        upload a new run
      </Link>
    </div>
  );
}
