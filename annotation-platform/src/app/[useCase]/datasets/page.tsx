import { notFound } from "next/navigation";
import Link from "next/link";
import { USE_CASES } from "@/lib/adapters";
import { listDatasets } from "@/lib/db/queries";
import type { UseCase } from "@/lib/types";
import { DatasetList } from "./DatasetList";

function isUseCase(value: string): value is UseCase {
  return (USE_CASES as string[]).includes(value);
}

export function generateStaticParams() {
  return USE_CASES.map((useCase) => ({ useCase }));
}

export default async function ManageDatasetsPage({
  params,
}: {
  params: Promise<{ useCase: string }>;
}) {
  const { useCase: useCaseParam } = await params;
  if (!isUseCase(useCaseParam)) notFound();
  const useCase = useCaseParam;

  const datasets = await listDatasets(useCase).catch(() => []);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight capitalize dark:text-slate-100">Manage {useCase} uploads</h1>
        <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
          <Link href={`/${useCase}`} className="underline">back to dashboard</Link>
          {" · "}
          <Link href={`/${useCase}/upload`} className="underline">upload a new run</Link>
        </p>
      </div>
      <p className="text-sm text-slate-600 dark:text-slate-400">
        Deleting an upload only removes it from the dataset picker — it never touches annotations or gold
        labels already recorded against these scenarios.
      </p>
      <DatasetList datasets={datasets} />
    </div>
  );
}
