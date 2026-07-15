"use client";

import { useState } from "react";
import { CONFIDENCE_LEVELS, DEDUCTION_REASON_CATEGORIES, EVIDENCE_SOURCE_TYPES } from "@/lib/annotationOptions";
import type { UseCase } from "@/lib/types";

type Props = {
  useCase: UseCase;
  scenarioId: string;
  language: string;
  policyLabel: string;
};

export function AnnotationForm({ useCase, scenarioId, language, policyLabel }: Props) {
  const [status, setStatus] = useState<"idle" | "submitting" | "done" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setStatus("submitting");
    setErrorMessage(null);

    const form = new FormData(e.currentTarget);
    const agreesRaw = form.get("agreesWithVerdict");
    const body = {
      scenarioId,
      useCase,
      language,
      policyLabel,
      annotatorName: form.get("annotatorName"),
      agreesWithVerdict: agreesRaw === "" ? null : agreesRaw === "true",
      disagreementReason: form.get("disagreementReason") || null,
      evidenceSourceType: form.get("evidenceSourceType") || null,
      deductionReasonCategory: form.get("deductionReasonCategory") || null,
      evidentiaryAttributionPresent:
        form.get("evidentiaryAttributionPresent") === "" ? null : form.get("evidentiaryAttributionPresent") === "true",
      freeText: form.get("freeText") || null,
      confidence: form.get("confidence") || null,
    };

    try {
      const res = await fetch("/api/annotations", {
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
    } catch (err) {
      setStatus("error");
      setErrorMessage(err instanceof Error ? err.message : "Submission failed");
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4 rounded-md border border-slate-200 bg-white p-5">
      <div>
        <label className="block text-xs font-medium text-slate-600 mb-1">Your name</label>
        <input name="annotatorName" required className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm" />
      </div>

      <fieldset>
        <legend className="text-xs font-medium text-slate-600 mb-1">Do you agree with the system&apos;s final verdict?</legend>
        <div className="flex gap-4 text-sm">
          <label className="flex items-center gap-1.5">
            <input type="radio" name="agreesWithVerdict" value="true" /> Agree
          </label>
          <label className="flex items-center gap-1.5">
            <input type="radio" name="agreesWithVerdict" value="false" /> Disagree
          </label>
          <label className="flex items-center gap-1.5">
            <input type="radio" name="agreesWithVerdict" value="" defaultChecked /> Not sure
          </label>
        </div>
      </fieldset>

      <div>
        <label className="block text-xs font-medium text-slate-600 mb-1">If you disagree, why?</label>
        <input name="disagreementReason" className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm" />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1">Evidence source type</label>
          <select name="evidenceSourceType" className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm">
            <option value="">—</option>
            {EVIDENCE_SOURCE_TYPES.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1">Deduction reason category</label>
          <select name="deductionReasonCategory" className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm">
            <option value="">—</option>
            {DEDUCTION_REASON_CATEGORIES.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>
      </div>

      <fieldset>
        <legend className="text-xs font-medium text-slate-600 mb-1">
          Does every evidentiary claim in the explanation have a matching tool call in the log? (Theme 3 check)
        </legend>
        <div className="flex gap-4 text-sm">
          <label className="flex items-center gap-1.5">
            <input type="radio" name="evidentiaryAttributionPresent" value="true" /> Yes
          </label>
          <label className="flex items-center gap-1.5">
            <input type="radio" name="evidentiaryAttributionPresent" value="false" /> No
          </label>
          <label className="flex items-center gap-1.5">
            <input type="radio" name="evidentiaryAttributionPresent" value="" defaultChecked /> N/A
          </label>
        </div>
      </fieldset>

      <div>
        <label className="block text-xs font-medium text-slate-600 mb-1">Free-text observations</label>
        <textarea name="freeText" rows={3} className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm" />
      </div>

      <div>
        <label className="block text-xs font-medium text-slate-600 mb-1">Your confidence in this annotation</label>
        <select name="confidence" className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm">
          <option value="">—</option>
          {CONFIDENCE_LEVELS.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </div>

      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={status === "submitting"}
          className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
        >
          {status === "submitting" ? "Submitting…" : "Submit annotation"}
        </button>
        {status === "done" && <span className="text-sm text-emerald-700">Saved.</span>}
        {status === "error" && <span className="text-sm text-red-700">{errorMessage}</span>}
      </div>
    </form>
  );
}
