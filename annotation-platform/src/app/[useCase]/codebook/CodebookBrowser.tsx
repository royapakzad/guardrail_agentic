"use client";

import { useMemo, useState } from "react";
import { BarChart } from "@/lib/ui/BarChart";
import { CodebookCodeCard } from "./CodebookCodeCard";
import type { UseCase } from "@/lib/types";
import type { CodebookCode, CodeApplicationWithCode } from "@/lib/db/queries";

type Props = {
  useCase: UseCase;
  codes: CodebookCode[];
  applicationsByCode: Record<number, CodeApplicationWithCode[]>;
};

/** Client-side search/filter/theme-jump/frequency chart over the codebook --
 * a server component can't hold the search-box state, so the actual
 * filtering + rendering lives here once the page has fetched everything. */
export function CodebookBrowser({ useCase, codes, applicationsByCode }: Props) {
  const [search, setSearch] = useState("");

  const themes = useMemo(() => {
    return [...new Set(codes.map((c) => c.theme ?? "(no theme)"))].sort();
  }, [codes]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return codes;
    return codes.filter(
      (c) =>
        c.name.toLowerCase().includes(q) ||
        c.definition.toLowerCase().includes(q) ||
        (c.theme ?? "").toLowerCase().includes(q)
    );
  }, [codes, search]);

  const byTheme = useMemo(() => {
    const map = new Map<string, CodebookCode[]>();
    for (const c of filtered) {
      const key = c.theme ?? "(no theme)";
      const list = map.get(key);
      if (list) list.push(c);
      else map.set(key, [c]);
    }
    return map;
  }, [filtered]);

  const frequencyData = useMemo(
    () =>
      codes.map((c) => ({
        label: `${c.theme ? `${c.theme}/` : ""}${c.name}`,
        value: (applicationsByCode[c.id] ?? []).length,
      })),
    [codes, applicationsByCode]
  );

  if (codes.length === 0) {
    return <p className="text-sm text-slate-500 dark:text-slate-400">No codes yet — add the first one below.</p>;
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="rounded-md border border-violet-200 bg-violet-50 p-4 dark:border-violet-800 dark:bg-violet-950/20">
        <h2 className="text-sm font-semibold text-violet-900 dark:text-violet-300 mb-2">Code frequency</h2>
        <BarChart data={frequencyData} unitLabel="applications" />
      </div>

      <div className="flex flex-col gap-2">
        <input
          type="search"
          placeholder="Search codes by name, definition, or theme…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full rounded border border-slate-300 px-3 py-2 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
        />
        {themes.length > 1 && (
          <div className="flex flex-wrap gap-1.5">
            <span className="text-xs text-slate-500 dark:text-slate-400 self-center">Jump to:</span>
            {themes.map((t) => (
              <a
                key={t}
                href={`#theme-${encodeURIComponent(t)}`}
                className="rounded-full bg-violet-100 px-2.5 py-1 text-xs text-violet-800 hover:bg-violet-200 dark:bg-violet-900/40 dark:text-violet-300 dark:hover:bg-violet-900/60"
              >
                {t}
              </a>
            ))}
          </div>
        )}
      </div>

      {filtered.length === 0 ? (
        <p className="text-sm text-slate-500 dark:text-slate-400">No codes match &ldquo;{search}&rdquo;.</p>
      ) : (
        <div className="flex flex-col gap-6">
          {[...byTheme.entries()].map(([theme, themeCodes]) => (
            <div key={theme} id={`theme-${encodeURIComponent(theme)}`}>
              <h2 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2">{theme}</h2>
              <div className="flex flex-col gap-2">
                {themeCodes.map((c) => (
                  <CodebookCodeCard key={c.id} useCase={useCase} code={c} applications={applicationsByCode[c.id] ?? []} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
