"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import type { Annotation, CodeApplicationWithCode } from "@/lib/db/queries";
import {
  CONFIDENCE_LEVELS,
  DEDUCTION_REASON_CATEGORIES,
  EVIDENCE_SOURCE_TYPES,
  CODING_TARGET_FIELD_LABELS,
  type CodingTargetField,
} from "@/lib/annotationOptions";

type Props = {
  annotatorName: string;
  annotation: Annotation | null;
  codeApplications: CodeApplicationWithCode[];
};

export function SavedReviewCard({ annotatorName, annotation, codeApplications }: Props) {
  const latest = [annotation?.updated_at, ...codeApplications.map((a) => a.updated_at)]
    .filter((d): d is string => Boolean(d))
    .sort()
    .at(-1);

  return (
    <div className="rounded-md border border-slate-200 bg-white p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <span className="font-medium text-sm text-slate-800">{annotatorName}</span>
        {latest && <span className="text-xs text-slate-400">{new Date(latest).toLocaleString()}</span>}
      </div>

      {annotation && <AnnotationSection annotation={annotation} />}

      {codeApplications.length > 0 && (
        <div className="flex flex-col gap-2 border-t border-slate-100 pt-2">
          <div className="text-xs font-medium uppercase tracking-wide text-violet-800">Qualitative codes</div>
          {codeApplications.map((a) => (
            <CodeApplicationRow key={a.id} application={a} />
          ))}
        </div>
      )}
    </div>
  );
}

function AgreementBadge({ agrees }: { agrees: boolean | null }) {
  if (agrees === null) return <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600">not sure</span>;
  return agrees ? (
    <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs text-emerald-800">agrees</span>
  ) : (
    <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs text-amber-800">disagrees</span>
  );
}

function AnnotationSection({ annotation }: { annotation: Annotation }) {
  const router = useRouter();
  const [editing, setEditing] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [status, setStatus] = useState<"idle" | "submitting" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  async function handleSave(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setStatus("submitting");
    setErrorMessage(null);
    const form = new FormData(e.currentTarget);
    const agreesRaw = form.get("agreesWithVerdict");
    try {
      const res = await fetch(`/api/annotations/${annotation.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          agreesWithVerdict: agreesRaw === "" ? null : agreesRaw === "true",
          disagreementReason: form.get("disagreementReason") || null,
          evidenceSourceType: form.get("evidenceSourceType") || null,
          deductionReasonCategory: form.get("deductionReasonCategory") || null,
          evidentiaryAttributionPresent:
            form.get("evidentiaryAttributionPresent") === "" ? null : form.get("evidentiaryAttributionPresent") === "true",
          freeText: form.get("freeText") || null,
          confidence: form.get("confidence") || null,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error ?? `Request failed (${res.status})`);
      }
      setEditing(false);
      setStatus("idle");
      router.refresh();
    } catch (err) {
      setStatus("error");
      setErrorMessage(err instanceof Error ? err.message : "Save failed");
    }
  }

  async function handleDelete() {
    setStatus("submitting");
    setErrorMessage(null);
    try {
      const res = await fetch(`/api/annotations/${annotation.id}`, { method: "DELETE" });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error ?? `Request failed (${res.status})`);
      }
      router.refresh();
    } catch (err) {
      setStatus("error");
      setErrorMessage(err instanceof Error ? err.message : "Delete failed");
    }
  }

  if (editing) {
    return (
      <form onSubmit={handleSave} className="flex flex-col gap-3 rounded border border-slate-200 bg-slate-50 p-3 text-sm">
        <fieldset>
          <legend className="text-xs font-medium text-slate-600 mb-1">Agree with the verdict?</legend>
          <div className="flex gap-4 text-sm">
            <label className="flex items-center gap-1.5">
              <input type="radio" name="agreesWithVerdict" value="true" defaultChecked={annotation.agrees_with_verdict === true} /> Agree
            </label>
            <label className="flex items-center gap-1.5">
              <input type="radio" name="agreesWithVerdict" value="false" defaultChecked={annotation.agrees_with_verdict === false} /> Disagree
            </label>
            <label className="flex items-center gap-1.5">
              <input type="radio" name="agreesWithVerdict" value="" defaultChecked={annotation.agrees_with_verdict === null} /> Not sure
            </label>
          </div>
        </fieldset>
        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1">If you disagree, why?</label>
          <input name="disagreementReason" defaultValue={annotation.disagreement_reason ?? ""} className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Evidence source type</label>
            <select name="evidenceSourceType" defaultValue={annotation.evidence_source_type ?? ""} className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm">
              <option value="">—</option>
              {EVIDENCE_SOURCE_TYPES.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Deduction reason category</label>
            <select name="deductionReasonCategory" defaultValue={annotation.deduction_reason_category ?? ""} className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm">
              <option value="">—</option>
              {DEDUCTION_REASON_CATEGORIES.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>
        </div>
        <fieldset>
          <legend className="text-xs font-medium text-slate-600 mb-1">Evidentiary attribution present?</legend>
          <div className="flex gap-4 text-sm">
            <label className="flex items-center gap-1.5">
              <input type="radio" name="evidentiaryAttributionPresent" value="true" defaultChecked={annotation.evidentiary_attribution_present === true} /> Yes
            </label>
            <label className="flex items-center gap-1.5">
              <input type="radio" name="evidentiaryAttributionPresent" value="false" defaultChecked={annotation.evidentiary_attribution_present === false} /> No
            </label>
            <label className="flex items-center gap-1.5">
              <input type="radio" name="evidentiaryAttributionPresent" value="" defaultChecked={annotation.evidentiary_attribution_present === null} /> N/A
            </label>
          </div>
        </fieldset>
        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1">Free-text observations</label>
          <textarea name="freeText" defaultValue={annotation.free_text ?? ""} rows={3} className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm" />
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1">Confidence</label>
          <select name="confidence" defaultValue={annotation.confidence ?? ""} className="w-full max-w-xs rounded border border-slate-300 px-2 py-1.5 text-sm">
            <option value="">—</option>
            {CONFIDENCE_LEVELS.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
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
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2 flex-wrap">
        <AgreementBadge agrees={annotation.agrees_with_verdict} />
        {annotation.confidence && <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600">confidence: {annotation.confidence}</span>}
        {annotation.evidence_source_type && <span className="rounded-full bg-sky-100 px-2 py-0.5 text-xs text-sky-800">source: {annotation.evidence_source_type}</span>}
        {annotation.deduction_reason_category && <span className="rounded-full bg-sky-100 px-2 py-0.5 text-xs text-sky-800">reason: {annotation.deduction_reason_category}</span>}
        <span className="ml-auto flex items-center gap-2">
          <button type="button" onClick={() => setEditing(true)} className="text-xs text-sky-700 underline">Edit</button>
          {!confirmingDelete ? (
            <button type="button" onClick={() => setConfirmingDelete(true)} className="text-xs text-red-700 underline">Delete</button>
          ) : (
            <span className="text-xs">
              <button type="button" onClick={handleDelete} disabled={status === "submitting"} className="font-medium text-red-700 underline">Confirm</button>{" "}
              <button type="button" onClick={() => setConfirmingDelete(false)} className="text-slate-500 underline">Cancel</button>
            </span>
          )}
        </span>
      </div>
      {annotation.disagreement_reason && <p className="text-sm text-slate-700">Disagreement: {annotation.disagreement_reason}</p>}
      {annotation.free_text && <p className="text-sm text-slate-700 whitespace-pre-wrap">{annotation.free_text}</p>}
      {status === "error" && <p className="text-xs text-red-700">{errorMessage}</p>}
    </div>
  );
}

function CodeApplicationRow({ application }: { application: CodeApplicationWithCode }) {
  const router = useRouter();
  const [editing, setEditing] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [status, setStatus] = useState<"idle" | "submitting" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  async function handleSave(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setStatus("submitting");
    setErrorMessage(null);
    const form = new FormData(e.currentTarget);
    try {
      const res = await fetch(`/api/code-applications/${application.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          quoteText: form.get("quoteText") || null,
          note: form.get("note") || null,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error ?? `Request failed (${res.status})`);
      }
      setEditing(false);
      setStatus("idle");
      router.refresh();
    } catch (err) {
      setStatus("error");
      setErrorMessage(err instanceof Error ? err.message : "Save failed");
    }
  }

  async function handleDelete() {
    setStatus("submitting");
    setErrorMessage(null);
    try {
      const res = await fetch(`/api/code-applications/${application.id}`, { method: "DELETE" });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error ?? `Request failed (${res.status})`);
      }
      router.refresh();
    } catch (err) {
      setStatus("error");
      setErrorMessage(err instanceof Error ? err.message : "Delete failed");
    }
  }

  if (editing) {
    return (
      <form onSubmit={handleSave} className="rounded border border-violet-200 bg-violet-50/40 p-2 flex flex-col gap-2 text-sm">
        <div className="font-mono text-xs font-medium text-violet-900">
          {application.code_theme ? `${application.code_theme} / ` : ""}{application.code_name}
        </div>
        <textarea name="quoteText" defaultValue={application.quote_text ?? ""} rows={2} placeholder="Quote" className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm" />
        <textarea name="note" defaultValue={application.note ?? ""} rows={2} placeholder="Note" className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm" />
        <div className="flex items-center gap-2">
          <button type="submit" disabled={status === "submitting"} className="rounded bg-slate-900 px-3 py-1 text-xs font-medium text-white hover:bg-slate-700 disabled:opacity-50">
            {status === "submitting" ? "Saving…" : "Save"}
          </button>
          <button type="button" onClick={() => setEditing(false)} className="rounded border border-slate-300 px-3 py-1 text-xs text-slate-600">Cancel</button>
          {status === "error" && <span className="text-xs text-red-700">{errorMessage}</span>}
        </div>
      </form>
    );
  }

  return (
    <div className="rounded border border-violet-200 bg-violet-50/40 p-2 text-sm">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-xs font-medium text-violet-900">
          {application.code_theme ? `${application.code_theme} / ` : ""}{application.code_name}
        </span>
        <span className="flex items-center gap-2 shrink-0">
          <button type="button" onClick={() => setEditing(true)} className="text-xs text-sky-700 underline">Edit</button>
          {!confirmingDelete ? (
            <button type="button" onClick={() => setConfirmingDelete(true)} className="text-xs text-red-700 underline">Delete</button>
          ) : (
            <span className="text-xs">
              <button type="button" onClick={handleDelete} disabled={status === "submitting"} className="font-medium text-red-700 underline">Confirm</button>{" "}
              <button type="button" onClick={() => setConfirmingDelete(false)} className="text-slate-500 underline">Cancel</button>
            </span>
          )}
        </span>
      </div>
      <div className="mt-1 text-xs text-slate-500">
        {CODING_TARGET_FIELD_LABELS[application.target_field as CodingTargetField] ?? application.target_field}
      </div>
      {application.quote_text && <p className="mt-1 text-sm italic text-slate-700">&ldquo;{application.quote_text}&rdquo;</p>}
      {application.note && <p className="mt-1 text-sm text-slate-600">{application.note}</p>}
      {status === "error" && <p className="mt-1 text-xs text-red-700">{errorMessage}</p>}
    </div>
  );
}
