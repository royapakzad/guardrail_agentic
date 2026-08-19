import { notFound } from "next/navigation";
import Link from "next/link";
import { USE_CASES } from "@/lib/adapters";
import { resolveDatasetIdParam } from "@/lib/datasetSelection";
import { listCodebookCodes, listCodeApplicationsForUseCase, listDatasets } from "@/lib/db/queries";
import type { UseCase } from "@/lib/types";
import type { CodeApplicationWithCode } from "@/lib/db/queries";
import { UseCaseNav } from "@/lib/ui/UseCaseNav";
import { DatasetPicker } from "@/lib/ui/DatasetPicker";
import { CodebookForm } from "./CodebookForm";
import { CodebookBrowser } from "./CodebookBrowser";

function isUseCase(value: string): value is UseCase {
  return (USE_CASES as string[]).includes(value);
}

export default async function CodebookPage({
  params,
  searchParams,
}: {
  params: Promise<{ useCase: string }>;
  searchParams: Promise<{ dataset?: string }>;
}) {
  const { useCase: useCaseParam } = await params;
  const { dataset: datasetParam } = await searchParams;
  if (!isUseCase(useCaseParam)) notFound();
  const useCase = useCaseParam;

  const datasetId = await resolveDatasetIdParam(useCase, datasetParam);

  // Codes and themes are the shared, evolving codebook -- not scoped to a
  // dataset (kept visible regardless of which run is selected). Only which
  // *applications* count toward each code's frequency/badge is scoped to the
  // selected dataset, so switching datasets doesn't hide or invent codes,
  // just which annotations count against them.
  let codes: Awaited<ReturnType<typeof listCodebookCodes>> = [];
  let applications: Awaited<ReturnType<typeof listCodeApplicationsForUseCase>> = [];
  let dbError: string | null = null;
  try {
    [codes, applications] = await Promise.all([
      listCodebookCodes(useCase),
      listCodeApplicationsForUseCase(useCase, String(datasetId)),
    ]);
  } catch (err) {
    dbError = err instanceof Error ? err.message : "Could not load codebook";
  }
  const availableDatasets = await listDatasets(useCase).catch(() => []);

  const applicationsByCode: Record<number, CodeApplicationWithCode[]> = {};
  for (const a of applications) {
    (applicationsByCode[a.code_id] ??= []).push(a);
  }

  return (
    <div className="flex flex-col gap-6">
      <UseCaseNav useCase={useCase} datasetId={String(datasetId)} />

      <div className="flex items-baseline justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight capitalize dark:text-slate-100">{useCase} codebook</h1>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
            The shared, evolving codebook for qualitative/thematic coding of this use case&apos;s scenarios —{" "}
            <Link href={`/${useCase}/scenarios`} className="underline">browse scenarios &amp; apply codes</Link>, or read the{" "}
            <Link href={`/${useCase}/help`} className="underline">help guide</Link> for how coding works here.
            Codes and themes are shared across every dataset; the frequency chart and &ldquo;applied&rdquo; counts below
            reflect only the dataset selected on the right.
          </p>
        </div>
        <DatasetPicker useCase={useCase} datasets={availableDatasets} currentId={String(datasetId)} basePath={`/${useCase}/codebook`} />
      </div>

      {dbError ? (
        <p className="text-sm text-amber-700 bg-amber-50 border border-amber-300 rounded px-3 py-2 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-200">
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
