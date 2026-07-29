"use client";

import { useRouter } from "next/navigation";
import Link from "next/link";
import type { UseCase } from "@/lib/types";

// Deliberately not importing Dataset's type from lib/db/queries (server-only
// module) — a minimal local shape keeps this client component's import graph
// free of the Postgres client. See lib/datasetId.ts for why.
type DatasetOption = { id: number; filename: string; uploaded_at: string; record_count: number; uploaded_by: string };

type Props = {
  useCase: UseCase;
  datasets: DatasetOption[];
  currentId: string; // "seed" (no uploads yet) or a numeric dataset id, as a string
  basePath: string; // e.g. `/${useCase}` or `/${useCase}/scenarios`
};

// No bundled sample data ships with the app -- if nothing's been uploaded yet
// for this use case there's nothing to pick from, so show a prompt instead of
// a dropdown with a single dead-end "sample" option.
export function DatasetPicker({ useCase, datasets, currentId, basePath }: Props) {
  const router = useRouter();

  if (datasets.length === 0) {
    return (
      <div className="text-sm text-slate-600 dark:text-slate-300">
        No batch runs uploaded yet for this use case —{" "}
        <Link href={`/${useCase}/upload`} className="underline">upload one</Link> to see results here.
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3 text-sm">
      <label className="text-slate-600 dark:text-slate-300">Dataset:</label>
      <select
        value={currentId}
        onChange={(e) => router.push(`${basePath}?dataset=${e.target.value}`)}
        className="rounded border border-slate-300 px-2 py-1 text-sm bg-white dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
      >
        {datasets.map((d) => (
          <option key={d.id} value={d.id}>
            {d.filename} — {new Date(d.uploaded_at).toLocaleDateString()} ({d.record_count} scenarios, by {d.uploaded_by})
          </option>
        ))}
      </select>
      <Link href={`/${useCase}/upload`} className="text-slate-500 underline hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200">
        upload a new run
      </Link>
    </div>
  );
}
