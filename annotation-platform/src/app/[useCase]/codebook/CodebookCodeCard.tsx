"use client";

import { useState } from "react";
import type { CodebookCode } from "@/lib/db/queries";

export function CodebookCodeCard({ code, applicationCount }: { code: CodebookCode; applicationCount: number }) {
  const [editing, setEditing] = useState(false);
  const [status, setStatus] = useState<"idle" | "submitting" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  async function handleSave(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setStatus("submitting");
    setErrorMessage(null);
    const form = new FormData(e.currentTarget);

    try {
      const res = await fetch(`/api/codebook/${code.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          definition: form.get("definition"),
          exampleQuote: form.get("exampleQuote") || null,
          theme: form.get("theme") || null,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error ?? `Request failed (${res.status})`);
      }
      window.location.reload();
    } catch (err) {
      setStatus("error");
      setErrorMessage(err instanceof Error ? err.message : "Save failed");
    }
  }

  if (editing) {
    return (
      <form onSubmit={handleSave} className="rounded-md border border-slate-200 bg-white p-3 flex flex-col gap-2 text-sm">
        <div className="font-mono font-medium text-slate-800">{code.name}</div>
        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1">Definition</label>
          <textarea name="definition" defaultValue={code.definition} rows={2} className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm" />
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1">Example quote</label>
          <textarea name="exampleQuote" defaultValue={code.example_quote ?? ""} rows={2} className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm" />
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1">Theme</label>
          <input name="theme" defaultValue={code.theme ?? ""} className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm" />
        </div>
        <div className="flex items-center gap-2">
          <button type="submit" disabled={status === "submitting"} className="rounded bg-slate-900 px-3 py-1 text-xs font-medium text-white hover:bg-slate-700 disabled:opacity-50">
            {status === "submitting" ? "Saving…" : "Save"}
          </button>
          <button type="button" onClick={() => setEditing(false)} className="rounded border border-slate-300 px-3 py-1 text-xs text-slate-600">
            Cancel
          </button>
          {status === "error" && <span className="text-xs text-red-700">{errorMessage}</span>}
        </div>
      </form>
    );
  }

  return (
    <div className="rounded-md border border-slate-200 bg-white p-3 text-sm">
      <div className="flex items-start justify-between gap-2">
        <div className="font-mono font-medium text-slate-800">{code.name}</div>
        <div className="flex items-center gap-2 shrink-0">
          <span className="rounded-full bg-sky-100 px-2 py-0.5 text-xs text-sky-800">{applicationCount} applied</span>
          <button type="button" onClick={() => setEditing(true)} className="text-xs text-sky-700 underline">
            Edit
          </button>
        </div>
      </div>
      <p className="mt-1 text-slate-700">{code.definition}</p>
      {code.example_quote && <p className="mt-1 text-slate-500 italic">“{code.example_quote}”</p>}
    </div>
  );
}
