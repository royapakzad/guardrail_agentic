"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import type { UseCase } from "@/lib/types";
import type { CodebookCode, CodeApplicationWithCode } from "@/lib/db/queries";

type Props = {
  code: CodebookCode;
  applications: CodeApplicationWithCode[];
  useCase?: UseCase;
};

export function CodebookCodeCard({ code, applications }: Props) {
  const router = useRouter();
  const [editing, setEditing] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [status, setStatus] = useState<"idle" | "submitting" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<{ count: number } | null>(null);

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
      setEditing(false);
      setStatus("idle");
      router.refresh();
    } catch (err) {
      setStatus("error");
      setErrorMessage(err instanceof Error ? err.message : "Save failed");
    }
  }

  async function handleDelete(confirmCascade: boolean) {
    setStatus("submitting");
    setErrorMessage(null);
    try {
      const res = await fetch(`/api/codebook/${code.id}`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirmCascade }),
      });
      if (res.status === 409) {
        const data = await res.json().catch(() => ({}));
        setDeleteConfirm({ count: data.applicationCount ?? applications.length });
        setStatus("idle");
        return;
      }
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
      <form onSubmit={handleSave} className="rounded-md border border-slate-200 bg-white p-3 flex flex-col gap-2 text-sm dark:border-slate-700 dark:bg-slate-900">
        <div className="font-mono font-medium text-slate-800 dark:text-slate-200">{code.name}</div>
        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Definition</label>
          <textarea name="definition" defaultValue={code.definition} rows={2} className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Example quote</label>
          <textarea name="exampleQuote" defaultValue={code.example_quote ?? ""} rows={2} className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Theme</label>
          <input name="theme" defaultValue={code.theme ?? ""} className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
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
    <div className="rounded-md border border-slate-200 bg-white p-3 text-sm dark:border-slate-700 dark:bg-slate-900">
      <div className="flex items-start justify-between gap-2">
        <div className="font-mono font-medium text-slate-800 dark:text-slate-200">{code.name}</div>
        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="rounded-full bg-violet-100 px-2 py-0.5 text-xs text-violet-800 hover:bg-violet-200 dark:bg-violet-900/40 dark:text-violet-300 dark:hover:bg-violet-900/60"
            title={applications.length > 0 ? "Click to see which scenarios used this code" : "Not applied to any scenario yet"}
          >
            {applications.length} applied{applications.length > 0 ? (expanded ? " ▲" : " ▼") : ""}
          </button>
          <button type="button" onClick={() => setEditing(true)} className="text-xs text-sky-700 dark:text-sky-400 underline">
            Edit
          </button>
          <button
            type="button"
            onClick={() => handleDelete(false)}
            disabled={status === "submitting"}
            className="text-xs text-red-700 dark:text-red-400 underline disabled:opacity-50"
          >
            Delete
          </button>
        </div>
      </div>
      <p className="mt-1 text-slate-700 dark:text-slate-300">{code.definition}</p>
      {code.example_quote && <p className="mt-1 text-slate-500 dark:text-slate-400 italic">&ldquo;{code.example_quote}&rdquo;</p>}
      {status === "error" && <p className="mt-1 text-xs text-red-700 dark:text-red-400">{errorMessage}</p>}

      {deleteConfirm && (
        <div className="mt-2 rounded border border-amber-300 bg-amber-50 p-2 text-xs text-amber-900 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-200">
          This code has been applied {deleteConfirm.count} time{deleteConfirm.count === 1 ? "" : "s"}. Deleting it will
          also delete those applications.{" "}
          <button type="button" onClick={() => handleDelete(true)} className="font-medium underline">
            Delete anyway
          </button>{" "}
          ·{" "}
          <button type="button" onClick={() => setDeleteConfirm(null)} className="underline">
            Cancel
          </button>
        </div>
      )}

      {expanded && applications.length > 0 && (
        <div className="mt-2 flex flex-col gap-1.5 border-t border-slate-100 pt-2 dark:border-slate-800">
          {applications.map((a) => (
            <div key={a.id} className="text-xs text-slate-600 dark:text-slate-400">
              <Link
                href={`/${a.use_case}/scenarios/${encodeURIComponent(a.scenario_id)}?lang=${a.language}`}
                className="font-medium text-sky-700 dark:text-sky-400 underline"
              >
                {a.scenario_id}
              </Link>{" "}
              · {a.annotator_name}
              {a.quote_text && <div className="italic">&ldquo;{a.quote_text}&rdquo;</div>}
              {a.note && <div>{a.note}</div>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
