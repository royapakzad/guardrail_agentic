import { notFound } from "next/navigation";
import Link from "next/link";
import { USE_CASES } from "@/lib/adapters";
import { listCodebookCodes, listCodeApplicationsForUseCase } from "@/lib/db/queries";
import type { UseCase } from "@/lib/types";
import type { CodeApplicationWithCode } from "@/lib/db/queries";
import { UseCaseNav } from "@/lib/ui/UseCaseNav";
import { CodebookForm } from "./CodebookForm";
import { CodebookBrowser } from "./CodebookBrowser";

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

  const applicationsByCode: Record<number, CodeApplicationWithCode[]> = {};
  for (const a of applications) {
    (applicationsByCode[a.code_id] ??= []).push(a);
  }

  return (
    <div className="flex flex-col gap-6">
      <UseCaseNav useCase={useCase} />

      <div>
        <h1 className="text-2xl font-semibold tracking-tight capitalize">{useCase} codebook</h1>
        <p className="mt-1 text-sm text-slate-600">
          The shared, evolving codebook for qualitative/thematic coding of this use case&apos;s scenarios —{" "}
          <Link href={`/${useCase}/scenarios`} className="underline">browse scenarios &amp; apply codes</Link>, or read the{" "}
          <Link href={`/${useCase}/help`} className="underline">help guide</Link> for how coding works here.
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
          <CodebookBrowser useCase={useCase} codes={codes} applicationsByCode={applicationsByCode} />
          <CodebookForm useCase={useCase} />
        </>
      )}
    </div>
  );
}
