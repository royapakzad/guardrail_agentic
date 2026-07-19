"use client";

import { useState } from "react";
import type { UseCase } from "@/lib/types";
import type { CodebookCode } from "@/lib/db/queries";
import { CODING_TARGET_FIELDS, CODING_TARGET_FIELD_LABELS } from "@/lib/annotationOptions";

type Props = {
  useCase: UseCase;
  scenarioId: string;
  language: string;
  policyLabel: string;
  codes: CodebookCode[];
};

export function QualitativeCodingForm({ useCase, scenarioId, language, policyLabel, codes }: Props) {
  const [status, setStatus] = useState<"idle" | "submitting" | "done" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setStatus("submitting");
    setErrorMessage(null);

    const form = new FormData(e.currentTarget);
    const body = {
      scenarioId,
      useCase,
      language,
      policyLabel,
      annotatorName: form.get("annotatorName"),
      codeId: form.get("codeId"),
      targetField: form.get("targetField"),
      quoteText: form.get("quoteText") || null,
      note: form.get("note") || null,
    };

    try {
      const res = await fetch("/api/code-applications", {
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
      window.location.reload();
    } catch (err) {
      setStatus("error");
      setErrorMessage(err instanceof Error ? err.message : "Submission failed");
    }
  }

  if (codes.length === 0) {
    return (
      <p className="text-sm text-slate-500">
        No codes in this use case&apos;s codebook yet — add some on the codebook page before applying codes here.
      </p>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-3 rounded-md border border-slate-200 bg-white p-4">
      <div>
        <label className="block text-xs font-medium text-slate-600 mb-1">Your name</label>
        <input name="annotatorName" required className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm" />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1">Code</label>
          <select name="codeId" required className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm">
            <option value="">—</option>
            {codes.map((c) => (
              <option key={c.id} value={c.id}>
                {c.theme ? `${c.theme} / ` : ""}{c.name}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1">Which text is this coding?</label>
          <select name="targetField" required className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm">
            <option value="">—</option>
            {CODING_TARGET_FIELDS.map((f) => (
              <option key={f} value={f}>{CODING_TARGET_FIELD_LABELS[f]}</option>
            ))}
          </select>
        </div>
      </div>
      <div>
        <label className="block text-xs font-medium text-slate-600 mb-1">
          Quote (paste the specific text this code applies to)
        </label>
        <textarea name="quoteText" rows={2} className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm" />
      </div>
      <div>
        <label className="block text-xs font-medium text-slate-600 mb-1">Note</label>
        <textarea name="note" rows={2} className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm" />
      </div>
      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={status === "submitting"}
          className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
        >
          {status === "submitting" ? "Saving…" : "Apply code"}
        </button>
        {status === "error" && <span className="text-sm text-red-700">{errorMessage}</span>}
      </div>
    </form>
  );
}
