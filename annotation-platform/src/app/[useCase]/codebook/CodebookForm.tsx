"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import type { UseCase } from "@/lib/types";

type Props = {
  useCase: UseCase;
};

export function CodebookForm({ useCase }: Props) {
  const router = useRouter();
  const [status, setStatus] = useState<"idle" | "submitting" | "done" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setStatus("submitting");
    setErrorMessage(null);

    const form = new FormData(e.currentTarget);
    const body = {
      useCase,
      name: form.get("name"),
      definition: form.get("definition"),
      exampleQuote: form.get("exampleQuote") || null,
      theme: form.get("theme") || null,
      createdBy: form.get("createdBy"),
    };

    try {
      const res = await fetch("/api/codebook", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error ?? `Request failed (${res.status})`);
      }
      setStatus("done");
      e.currentTarget.reset();
      router.refresh();
    } catch (err) {
      setStatus("error");
      setErrorMessage(err instanceof Error ? err.message : "Submission failed");
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-3 rounded-md border border-violet-200 bg-violet-50/40 p-4 dark:border-violet-800 dark:bg-violet-950/20">
      <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-200">Add a code</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Code name</label>
          <input name="name" required className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Theme</label>
          <input
            name="theme"
            placeholder="e.g. Evidence calibration"
            className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
          />
        </div>
      </div>
      <div>
        <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
          Definition (inclusion/exclusion criteria — when does this code apply?)
        </label>
        <textarea name="definition" required rows={2} className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
      </div>
      <div>
        <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Example quote</label>
        <textarea name="exampleQuote" rows={2} className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
      </div>
      <div>
        <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Your name</label>
        <input name="createdBy" required className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
      </div>
      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={status === "submitting"}
          className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-slate-300"
        >
          {status === "submitting" ? "Saving…" : "Add code"}
        </button>
        {status === "done" && <span className="text-sm font-medium text-emerald-700 dark:text-emerald-400">✓ Code added</span>}
        {status === "error" && <span className="text-sm text-red-700 dark:text-red-400">{errorMessage}</span>}
      </div>
    </form>
  );
}
