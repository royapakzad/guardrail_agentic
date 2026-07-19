import { notFound } from "next/navigation";
import Link from "next/link";
import { USE_CASES } from "@/lib/adapters";
import { listCodebookCodes, listCodeApplicationsForUseCase } from "@/lib/db/queries";
import type { UseCase } from "@/lib/types";
import { CodebookForm } from "./CodebookForm";
import { CodebookCodeCard } from "./CodebookCodeCard";

function isUseCase(value: string): value is UseCase {
  return (USE_CASES as string[]).includes(value);
}

export default async function CodebookPage({ params }: { params: Promise<{ useCase: string }> }) {
  const { useCase: useCaseParam } = await params;
  if (!isUseCase(useCaseParam)) notFound();
  const useCase = useCaseParam;

  let codes: Awaited<ReturnType<typeof listCodebookCodes>> = [];
  let applications: Awaited<ReturnType<typeof listCodeApplicationsForUseCase>> = [];
  let dbError: string | null = null;
  try {
    [codes, applications] = await Promise.all([
      listCodebookCodes(useCase),
      listCodeApplicationsForUseCase(useCase),
    ]);
  } catch (err) {
    dbError = err instanceof Error ? err.message : "Could not load codebook";
  }

  const applicationCountByCode = new Map<number, number>();
  for (const a of applications) {
    applicationCountByCode.set(a.code_id, (applicationCountByCode.get(a.code_id) ?? 0) + 1);
  }

  const byTheme = new Map<string, typeof codes>();
  for (const c of codes) {
    const key = c.theme ?? "(no theme)";
    const list = byTheme.get(key);
    if (list) list.push(c);
    else byTheme.set(key, [c]);
  }

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight capitalize">{useCase} codebook</h1>
        <p className="mt-1 text-sm text-slate-600">
          The shared, evolving codebook for qualitative/thematic coding of this use case&apos;s scenarios —{" "}
          <Link href={`/${useCase}/scenarios`} className="underline">browse scenarios &amp; apply codes</Link>.
          Codes are grouped by theme; definitions can be revised as coding proceeds (standard thematic-analysis practice)
          rather than needing to be perfect up front.
        </p>
      </div>

      {dbError ? (
        <p className="text-sm text-amber-700 bg-amber-50 border border-amber-300 rounded px-3 py-2">
          Could not load codebook: {dbError}
        </p>
      ) : (
        <>
          {codes.length === 0 ? (
            <p className="text-sm text-slate-500">No codes yet — add the first one below.</p>
          ) : (
            <div className="flex flex-col gap-6">
              {[...byTheme.entries()].map(([theme, themeCodes]) => (
                <div key={theme}>
                  <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-2">{theme}</h2>
                  <div className="flex flex-col gap-2">
                    {themeCodes.map((c) => (
                      <CodebookCodeCard key={c.id} code={c} applicationCount={applicationCountByCode.get(c.id) ?? 0} />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          <CodebookForm useCase={useCase} />
        </>
      )}
    </div>
  );
}
