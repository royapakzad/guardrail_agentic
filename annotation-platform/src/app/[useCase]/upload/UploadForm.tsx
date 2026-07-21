"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import type { UseCase } from "@/lib/types";

export function UploadForm({ useCase }: { useCase: UseCase }) {
  const router = useRouter();
  const [status, setStatus] = useState<"idle" | "uploading" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setStatus("uploading");
    setErrorMessage(null);

    const form = e.currentTarget;
    const fd = new FormData(form);
    fd.set("useCase", useCase);

    try {
      const res = await fetch("/api/datasets", { method: "POST", body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.error ?? `Upload failed (${res.status})`);
      }
      router.push(`/${useCase}?dataset=${data.dataset.id}`);
    } catch (err) {
      setStatus("error");
      setErrorMessage(err instanceof Error ? err.message : "Upload failed");
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4 rounded-md border border-slate-200 bg-white p-5 max-w-lg dark:border-slate-700 dark:bg-slate-900">
      <div>
        <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Result file (JSON)</label>
        <input type="file" name="file" accept="application/json,.json" required className="w-full text-sm" />
        <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
          The output JSON produced by a batch guardrail evaluation run for this use case — same format as
          files in the parent repo&apos;s <code className="font-mono">outputs/</code> directory.
        </p>
      </div>
      <div>
        <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Your name</label>
        <input name="uploadedBy" required className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
      </div>
      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={status === "uploading"}
          className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-slate-300"
        >
          {status === "uploading" ? "Uploading…" : "Upload"}
        </button>
        {status === "error" && <span className="text-sm text-red-700 dark:text-red-400">{errorMessage}</span>}
      </div>
    </form>
  );
}
