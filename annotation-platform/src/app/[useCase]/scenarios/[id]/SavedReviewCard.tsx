"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import type { Annotation, CodeApplicationWithCode } from "@/lib/db/queries";
import { CODING_TARGET_FIELD_LABELS, type CodingTargetField } from "@/lib/annotationOptions";

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
    <div className="rounded-md border border-slate-200 bg-white p-4 flex flex-col gap-3 dark:border-slate-700 dark:bg-slate-900">
      <div className="flex items-center justify-between gap-2">
        <span className="font-medium text-sm text-slate-800 dark:text-slate-200">{annotatorName}</span>
        {latest && <span className="text-xs text-slate-400 dark:text-slate-500">{new Date(latest).toLocaleString()}</span>}
      </div>

      {annotation && <AnnotationSection annotation={annotation} />}

      {codeApplications.length > 0 && (
        <div className="flex flex-col gap-2 border-t border-slate-100 pt-2 dark:border-slate-800">
          <div className="text-xs font-medium uppercase tracking-wide text-violet-800 dark:text-violet-400">Qualitative codes</div>
          {codeApplications.map((a) => (
            <CodeApplicationRow key={a.id} application={a} />
          ))}
        </div>
      )}
    </div>
  );
}

function AgreementBadge({ agrees }: { agrees: boolean | null }) {
  if (agrees === null) return <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600 dark:bg-slate-800 dark:text-slate-400">not sure</span>;
  return agrees ? (
    <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs text-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-300">agrees</span>
  ) : (
    <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs text-amber-800 dark:bg-amber-950/50 dark:text-amber-300">disagrees</span>
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
          evidentiaryAttributionPresent:
            form.get("evidentiaryAttributionPresent") === "" ? null : form.get("evidentiaryAttributionPresent") === "true",
          freeText: form.get("freeText") || null,
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
      <form onSubmit={handleSave} className="flex flex-col gap-3 rounded border border-slate-200 bg-slate-50 p-3 text-sm dark:border-slate-700 dark:bg-slate-800">
        <fieldset>
          <legend className="text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Agree with the verdict?</legend>
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
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">If you disagree, why?</label>
          <input name="disagreementReason" defaultValue={annotation.disagreement_reason ?? ""} className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        </div>
        <fieldset>
          <legend className="text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Evidentiary attribution present?</legend>
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
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Free-text observations</label>
          <textarea name="freeText" defaultValue={annotation.free_text ?? ""} rows={3} className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        </div>
        <div className="flex items-center gap-2">
          <button type="submit" disabled={status === "submitting"} className="rounded bg-slate-900 px-3 py-1 text-xs font-medium text-white hover:bg-slate-700 disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-slate-300">
            {status === "submitting" ? "Saving…" : "Save"}
          </button>
          <button type="button" onClick={() => setEditing(false)} className="rounded border border-slate-300 px-3 py-1 text-xs text-slate-600 dark:border-slate-600 dark:text-slate-400">
            Cancel
          </button>
          {status === "error" && <span className="text-xs text-red-700 dark:text-red-400">{errorMessage}</span>}
        </div>
      </form>
    );
  }

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2 flex-wrap">
        <AgreementBadge agrees={annotation.agrees_with_verdict} />
        {annotation.confidence && <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600 dark:bg-slate-800 dark:text-slate-400">confidence: {annotation.confidence}</span>}
        {annotation.evidence_source_type && <span className="rounded-full bg-sky-100 px-2 py-0.5 text-xs text-sky-800 dark:bg-sky-950/50 dark:text-sky-300">source: {annotation.evidence_source_type}</span>}
        {annotation.deduction_reason_category && <span className="rounded-full bg-sky-100 px-2 py-0.5 text-xs text-sky-800 dark:bg-sky-950/50 dark:text-sky-300">reason: {annotation.deduction_reason_category}</span>}
        <span className="ml-auto flex items-center gap-2">
          <button type="button" onClick={() => setEditing(true)} className="text-xs text-sky-700 dark:text-sky-400 underline">Edit</button>
          {!confirmingDelete ? (
            <button type="button" onClick={() => setConfirmingDelete(true)} className="text-xs text-red-700 dark:text-red-400 underline">Delete</button>
          ) : (
            <span className="text-xs">
              <button type="button" onClick={handleDelete} disabled={status === "submitting"} className="font-medium text-red-700 dark:text-red-400 underline">Confirm</button>{" "}
              <button type="button" onClick={() => setConfirmingDelete(false)} className="text-slate-500 dark:text-slate-400 underline">Cancel</button>
            </span>
          )}
        </span>
      </div>
      {annotation.disagreement_reason && <p className="text-sm text-slate-700 dark:text-slate-300">Disagreement: {annotation.disagreement_reason}</p>}
      {annotation.free_text && <p className="text-sm text-slate-700 dark:text-slate-300 whitespace-pre-wrap">{annotation.free_text}</p>}
      {status === "error" && <p className="text-xs text-red-700 dark:text-red-400">{errorMessage}</p>}
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
      <form onSubmit={handleSave} className="rounded border border-violet-200 bg-violet-50/40 p-2 flex flex-col gap-2 text-sm dark:border-violet-800 dark:bg-violet-950/20">
        <div className="font-mono text-xs font-medium text-violet-900 dark:text-violet-300">
          {application.code_theme ? `${application.code_theme} / ` : ""}{application.code_name}
        </div>
        <textarea name="quoteText" defaultValue={application.quote_text ?? ""} rows={2} placeholder="Quote" className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        <textarea name="note" defaultValue={application.note ?? ""} rows={2} placeholder="Note" className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        <div className="flex items-center gap-2">
          <button type="submit" disabled={status === "submitting"} className="rounded bg-slate-900 px-3 py-1 text-xs font-medium text-white hover:bg-slate-700 disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-slate-300">
            {status === "submitting" ? "Saving…" : "Save"}
          </button>
          <button type="button" onClick={() => setEditing(false)} className="rounded border border-slate-300 px-3 py-1 text-xs text-slate-600 dark:border-slate-600 dark:text-slate-400">Cancel</button>
          {status === "error" && <span className="text-xs text-red-700 dark:text-red-400">{errorMessage}</span>}
        </div>
      </form>
    );
  }

  return (
    <div className="rounded border border-violet-200 bg-violet-50/40 p-2 text-sm dark:border-violet-800 dark:bg-violet-950/20">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-xs font-medium text-violet-900 dark:text-violet-300">
          {application.code_theme ? `${application.code_theme} / ` : ""}{application.code_name}
        </span>
        <span className="flex items-center gap-2 shrink-0">
          <button type="button" onClick={() => setEditing(true)} className="text-xs text-sky-700 dark:text-sky-400 underline">Edit</button>
          {!confirmingDelete ? (
            <button type="button" onClick={() => setConfirmingDelete(true)} className="text-xs text-red-700 dark:text-red-400 underline">Delete</button>
          ) : (
            <span className="text-xs">
              <button type="button" onClick={handleDelete} disabled={status === "submitting"} className="font-medium text-red-700 dark:text-red-400 underline">Confirm</button>{" "}
              <button type="button" onClick={() => setConfirmingDelete(false)} className="text-slate-500 dark:text-slate-400 underline">Cancel</button>
            </span>
          )}
        </span>
      </div>
      <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">
        {CODING_TARGET_FIELD_LABELS[application.target_field as CodingTargetField] ?? application.target_field}
      </div>
      {application.quote_text && <p className="mt-1 text-sm italic text-slate-700 dark:text-slate-300">&ldquo;{application.quote_text}&rdquo;</p>}
      {application.note && <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">{application.note}</p>}
      {status === "error" && <p className="mt-1 text-xs text-red-700 dark:text-red-400">{errorMessage}</p>}
    </div>
  );
}
