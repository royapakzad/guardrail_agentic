"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

// Minimal local shape, same reasoning as DatasetPicker.tsx: keep this client
// component's import graph free of lib/db/queries (server-only, pulls in the
// Postgres client).
type DatasetOption = { id: number; filename: string; uploaded_at: string; record_count: number; uploaded_by: string };

export function DatasetList({ datasets }: { datasets: DatasetOption[] }) {
  const router = useRouter();
  const [pendingId, setPendingId] = useState<number | null>(null);
  const [confirmId, setConfirmId] = useState<number | null>(null);
  const [errorById, setErrorById] = useState<Record<number, string>>({});

  async function handleDelete(id: number) {
    setPendingId(id);
    setErrorById((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    try {
      const res = await fetch(`/api/datasets/${id}`, { method: "DELETE" });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error ?? `Delete failed (${res.status})`);
      }
      setConfirmId(null);
      router.refresh();
    } catch (err) {
      setErrorById((prev) => ({ ...prev, [id]: err instanceof Error ? err.message : "Delete failed" }));
    } finally {
      setPendingId(null);
    }
  }

  if (datasets.length === 0) {
    return <p className="text-sm text-slate-500 dark:text-slate-400">No uploads yet for this use case.</p>;
  }

  return (
    <div className="flex flex-col gap-2">
      {datasets.map((d) => (
        <div
          key={d.id}
          className="flex items-center justify-between gap-3 rounded-md border border-slate-200 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-900"
        >
          <div className="min-w-0">
            <div className="truncate font-medium text-slate-800 dark:text-slate-200" title={d.filename}>
              {d.filename}
            </div>
            <div className="text-xs text-slate-500 dark:text-slate-400">
              {new Date(d.uploaded_at).toLocaleString()} · {d.record_count} scenarios · by {d.uploaded_by}
            </div>
            {errorById[d.id] && <div className="mt-1 text-xs text-red-700 dark:text-red-400">{errorById[d.id]}</div>}
          </div>

          {confirmId === d.id ? (
            <div className="flex shrink-0 items-center gap-2 text-xs">
              <span className="text-amber-800 dark:text-amber-300">Delete this upload?</span>
              <button
                type="button"
                onClick={() => handleDelete(d.id)}
                disabled={pendingId === d.id}
                className="font-medium text-red-700 underline disabled:opacity-50 dark:text-red-400"
              >
                {pendingId === d.id ? "Deleting…" : "Delete anyway"}
              </button>
              <button type="button" onClick={() => setConfirmId(null)} className="text-slate-500 underline dark:text-slate-400">
                Cancel
              </button>
            </div>
          ) : (
            <button
              type="button"
              onClick={() => setConfirmId(d.id)}
              className="shrink-0 text-xs text-red-700 underline dark:text-red-400"
            >
              Delete
            </button>
          )}
        </div>
      ))}
    </div>
  );
}
