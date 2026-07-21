"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import type { UseCase } from "@/lib/types";
import type { CodebookCode } from "@/lib/db/queries";
import {
  CONFIDENCE_LEVELS,
  DEDUCTION_REASON_CATEGORIES,
  EVIDENCE_SOURCE_TYPES,
  CODING_TARGET_FIELDS,
  CODING_TARGET_FIELD_LABELS,
} from "@/lib/annotationOptions";

type Props = {
  useCase: UseCase;
  scenarioId: string;
  language: string;
  policyLabel: string;
  codes: CodebookCode[];
};

const NEW_CODE_VALUE = "__new__";

type CodeRow = {
  key: number;
  codeId: string;
  targetField: string;
  quoteText: string;
  note: string;
  addingNew: boolean;
  newName: string;
  newTheme: string;
  newDefinition: string;
  newExampleQuote: string;
};

let rowKeySeq = 0;
function emptyRow(): CodeRow {
  return {
    key: rowKeySeq++,
    codeId: "",
    targetField: "",
    quoteText: "",
    note: "",
    addingNew: false,
    newName: "",
    newTheme: "",
    newDefinition: "",
    newExampleQuote: "",
  };
}

export function ScenarioReviewForm({ useCase, scenarioId, language, policyLabel, codes: initialCodes }: Props) {
  const router = useRouter();
  const [codes, setCodes] = useState<CodebookCode[]>(initialCodes);
  const [rows, setRows] = useState<CodeRow[]>([emptyRow()]);
  const [status, setStatus] = useState<"idle" | "submitting" | "done" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  function updateRow(key: number, patch: Partial<CodeRow>) {
    setRows((rs) => rs.map((r) => (r.key === key ? { ...r, ...patch } : r)));
  }

  function addRow() {
    setRows((rs) => [...rs, emptyRow()]);
  }

  function removeRow(key: number) {
    setRows((rs) => (rs.length === 1 ? rs : rs.filter((r) => r.key !== key)));
  }

  async function createInlineCode(row: CodeRow, annotatorName: string) {
    if (!row.newName.trim() || !row.newDefinition.trim()) {
      throw new Error("New code needs at least a name and a definition");
    }
    const res = await fetch("/api/codebook", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        useCase,
        name: row.newName,
        definition: row.newDefinition,
        exampleQuote: row.newExampleQuote || null,
        theme: row.newTheme || null,
        createdBy: annotatorName,
      }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error ?? `Could not create code (${res.status})`);
    }
    const data = await res.json();
    return data.code as CodebookCode;
  }

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setStatus("submitting");
    setErrorMessage(null);

    const form = new FormData(e.currentTarget);
    const annotatorName = String(form.get("annotatorName") ?? "").trim();
    const agreesRaw = form.get("agreesWithVerdict");

    try {
      if (!annotatorName) throw new Error("Your name is required");

      // Resolve any "+ Add a new code" rows into real codes first.
      const resolvedRows: CodeRow[] = [];
      for (const row of rows) {
        if (row.addingNew) {
          const newCode = await createInlineCode(row, annotatorName);
          setCodes((cs) => [...cs, newCode]);
          resolvedRows.push({ ...row, addingNew: false, codeId: String(newCode.id) });
        } else {
          resolvedRows.push(row);
        }
      }

      const annotationBody = {
        scenarioId,
        useCase,
        language,
        policyLabel,
        annotatorName,
        agreesWithVerdict: agreesRaw === "" ? null : agreesRaw === "true",
        disagreementReason: form.get("disagreementReason") || null,
        evidenceSourceType: form.get("evidenceSourceType") || null,
        deductionReasonCategory: form.get("deductionReasonCategory") || null,
        evidentiaryAttributionPresent:
          form.get("evidentiaryAttributionPresent") === "" ? null : form.get("evidentiaryAttributionPresent") === "true",
        freeText: form.get("freeText") || null,
        confidence: form.get("confidence") || null,
      };
      const annotationRes = await fetch("/api/annotations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(annotationBody),
      });
      if (!annotationRes.ok) {
        const data = await annotationRes.json().catch(() => ({}));
        throw new Error(data.error ?? `Could not save your review (${annotationRes.status})`);
      }

      for (const row of resolvedRows) {
        if (!row.codeId) continue;
        const res = await fetch("/api/code-applications", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            scenarioId,
            useCase,
            language,
            policyLabel,
            annotatorName,
            codeId: Number(row.codeId),
            targetField: row.targetField || "other",
            quoteText: row.quoteText || null,
            note: row.note || null,
          }),
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.error ?? `Could not save a qualitative code (${res.status})`);
        }
      }

      setStatus("done");
      setRows([emptyRow()]);
      e.currentTarget.reset();
      router.refresh();
    } catch (err) {
      setStatus("error");
      setErrorMessage(err instanceof Error ? err.message : "Save failed");
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-6 rounded-md border border-slate-200 bg-white p-5 dark:border-slate-700 dark:bg-slate-900">
      <div>
        <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Your name</label>
        <input name="annotatorName" required className="w-full max-w-xs rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
      </div>

      <div className="flex flex-col gap-4">
        <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-200">Your structured judgment</h3>

        <fieldset>
          <legend className="text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Do you agree with the system&apos;s final verdict?</legend>
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
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">If you disagree, why?</label>
          <input name="disagreementReason" className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Evidence source type</label>
            <select name="evidenceSourceType" className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100">
              <option value="">—</option>
              {EVIDENCE_SOURCE_TYPES.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Deduction reason category</label>
            <select name="deductionReasonCategory" className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100">
              <option value="">—</option>
              {DEDUCTION_REASON_CATEGORIES.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>
        </div>

        <fieldset>
          <legend className="text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
            Does every evidentiary claim in the explanation have a matching tool call in the log?
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
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Free-text observations</label>
          <textarea name="freeText" rows={3} className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Your confidence in this review</label>
          <select name="confidence" className="w-full max-w-xs rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100">
            <option value="">—</option>
            {CONFIDENCE_LEVELS.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="flex flex-col gap-3 rounded-md border border-violet-200 bg-violet-50/40 p-4 dark:border-violet-800 dark:bg-violet-950/20">
        <div>
          <h3 className="text-sm font-semibold text-violet-900 dark:text-violet-300">Qualitative codes</h3>
          <p className="text-xs text-violet-800/80 dark:text-violet-300/80 mt-0.5">
            Apply one or more codes from the codebook to specific text in this scenario. Don&apos;t see the right code?
            Choose &ldquo;+ Add a new code&rdquo; below to create one without leaving this page.
          </p>
        </div>

        {rows.map((row) => (
          <CodeRowFields
            key={row.key}
            row={row}
            codes={codes}
            showRemove={rows.length > 1}
            onChange={(patch) => updateRow(row.key, patch)}
            onRemove={() => removeRow(row.key)}
          />
        ))}

        <button
          type="button"
          onClick={addRow}
          className="self-start rounded-full border border-violet-300 px-3 py-1 text-xs font-medium text-violet-800 hover:bg-violet-100 dark:border-violet-700 dark:text-violet-300 dark:hover:bg-violet-900/40"
        >
          + Add another code
        </button>
      </div>

      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={status === "submitting"}
          className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
        >
          {status === "submitting" ? "Saving…" : "Save review"}
        </button>
        {status === "done" && <span className="text-sm font-medium text-emerald-700 dark:text-emerald-400">✓ Saved — see it below</span>}
        {status === "error" && <span className="text-sm text-red-700 dark:text-red-400">{errorMessage}</span>}
      </div>
    </form>
  );
}

function CodeRowFields({
  row,
  codes,
  showRemove,
  onChange,
  onRemove,
}: {
  row: CodeRow;
  codes: CodebookCode[];
  showRemove: boolean;
  onChange: (patch: Partial<CodeRow>) => void;
  onRemove: () => void;
}) {
  function handleCodeSelect(value: string) {
    if (value === NEW_CODE_VALUE) {
      onChange({ addingNew: true, codeId: "" });
    } else {
      onChange({ codeId: value, addingNew: false });
    }
  }

  return (
    <div className="rounded border border-violet-200 bg-white p-3 flex flex-col gap-2 dark:border-violet-800 dark:bg-slate-900">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Code</label>
          <select
            value={row.addingNew ? NEW_CODE_VALUE : row.codeId}
            onChange={(e) => handleCodeSelect(e.target.value)}
            className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
          >
            <option value="">— none for this row —</option>
            {codes.map((c) => (
              <option key={c.id} value={c.id}>
                {c.theme ? `${c.theme} / ` : ""}{c.name}
              </option>
            ))}
            <option value={NEW_CODE_VALUE}>+ Add a new code…</option>
          </select>
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Which text is this coding?</label>
          <select
            value={row.targetField}
            onChange={(e) => onChange({ targetField: e.target.value })}
            className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
          >
            <option value="">—</option>
            {CODING_TARGET_FIELDS.map((f) => (
              <option key={f} value={f}>{CODING_TARGET_FIELD_LABELS[f]}</option>
            ))}
          </select>
        </div>
      </div>

      {row.addingNew && (
        <div className="rounded border border-violet-300 bg-violet-50 p-2 flex flex-col gap-2 dark:border-violet-700 dark:bg-violet-950/30">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            <input
              placeholder="New code name"
              value={row.newName}
              onChange={(e) => onChange({ newName: e.target.value })}
              className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
            />
            <input
              placeholder="Theme (optional)"
              value={row.newTheme}
              onChange={(e) => onChange({ newTheme: e.target.value })}
              className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
            />
          </div>
          <textarea
            placeholder="Definition — when does this code apply?"
            value={row.newDefinition}
            onChange={(e) => onChange({ newDefinition: e.target.value })}
            rows={2}
            className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
          />
          <textarea
            placeholder="Example quote (optional)"
            value={row.newExampleQuote}
            onChange={(e) => onChange({ newExampleQuote: e.target.value })}
            rows={2}
            className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
          />
          <p className="text-xs text-violet-800/80 dark:text-violet-300/80">
            This code is created when you click &ldquo;Save review&rdquo; below.
          </p>
        </div>
      )}

      <div>
        <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Quote (paste the specific text this code applies to)</label>
        <textarea
          value={row.quoteText}
          onChange={(e) => onChange({ quoteText: e.target.value })}
          rows={2}
          className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
        />
      </div>
      <div>
        <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Note</label>
        <textarea
          value={row.note}
          onChange={(e) => onChange({ note: e.target.value })}
          rows={2}
          className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
        />
      </div>

      {showRemove && (
        <button type="button" onClick={onRemove} className="self-start text-xs text-red-700 dark:text-red-400 underline">
          Remove this code row
        </button>
      )}
    </div>
  );
}
