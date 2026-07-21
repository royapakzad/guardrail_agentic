import { notFound } from "next/navigation";
import Link from "next/link";
import { USE_CASES, getRecordsForDataset } from "@/lib/adapters";
import { resolveDatasetIdParam } from "@/lib/datasetSelection";
import { listDatasets } from "@/lib/db/queries";
import { DatasetPicker } from "@/lib/ui/DatasetPicker";
import { UseCaseNav } from "@/lib/ui/UseCaseNav";
import type { JudgePass, PolicyVariant, UseCase } from "@/lib/types";

function isUseCase(value: string): value is UseCase {
  return (USE_CASES as string[]).includes(value);
}

export function generateStaticParams() {
  return USE_CASES.map((useCase) => ({ useCase }));
}

export default async function ScenarioListPage({
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
  const [records, availableDatasets] = await Promise.all([
    getRecordsForDataset(useCase, datasetId),
    listDatasets(useCase).catch(() => []),
  ]);

  // The full criterion count for a policy should be the same on every
  // scenario that uses it -- if some record's judge only returned a subset
  // (a judge parsing/completeness issue, not a policy design difference),
  // the badge below flags it rather than silently showing a smaller "total".
  const expectedTotalByLabel = new Map<string, number>();
  for (const r of records) {
    for (const v of r.policyVariants) {
      const n = Math.max(v.nonagentic.criteriaVerdicts.length, v.agentic.criteriaVerdicts.length);
      expectedTotalByLabel.set(v.label, Math.max(expectedTotalByLabel.get(v.label) ?? 0, n));
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <UseCaseNav useCase={useCase} datasetId={String(datasetId)} />

      <div className="flex items-baseline justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight capitalize dark:text-slate-100">{useCase} scenarios</h1>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">{records.length} scenarios</p>
        </div>
        <DatasetPicker useCase={useCase} datasets={availableDatasets} currentId={String(datasetId)} basePath={`/${useCase}/scenarios`} />
      </div>

      <div className="overflow-x-auto rounded-md border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-900">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-left dark:border-slate-700 dark:bg-slate-800">
              <th className="px-3 py-2 font-medium text-slate-600 dark:text-slate-400">ID</th>
              <th className="px-3 py-2 font-medium text-slate-600 dark:text-slate-400">Lang</th>
              <th className="px-3 py-2 font-medium text-slate-600 dark:text-slate-400">Scenario</th>
              <th className="px-3 py-2 font-medium text-slate-600 dark:text-slate-400">Non-agentic</th>
              <th className="px-3 py-2 font-medium text-slate-600 dark:text-slate-400">Agentic</th>
            </tr>
          </thead>
          <tbody>
            {records.map((r) => (
              <tr key={`${r.id}-${r.language}`} className="border-b border-slate-100 last:border-0 hover:bg-slate-50 dark:border-slate-800 dark:hover:bg-slate-800">
                <td className="px-3 py-2 whitespace-nowrap">
                  <Link
                    href={`/${useCase}/scenarios/${encodeURIComponent(r.id)}?lang=${r.language}&dataset=${datasetId}`}
                    className="font-medium text-slate-900 hover:underline dark:text-slate-100"
                  >
                    {r.id}
                  </Link>
                </td>
                <td className="px-3 py-2 whitespace-nowrap uppercase text-xs text-slate-500 dark:text-slate-400">{r.language}</td>
                <td className="px-3 py-2 max-w-lg truncate" title={r.scenario}>
                  {r.scenario}
                </td>
                <td className="px-3 py-2">
                  <PolicyVariantBadges variants={r.policyVariants} pass="nonagentic" expectedTotalByLabel={expectedTotalByLabel} />
                </td>
                <td className="px-3 py-2">
                  <PolicyVariantBadges variants={r.policyVariants} pass="agentic" expectedTotalByLabel={expectedTotalByLabel} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PolicyVariantBadges({
  variants,
  pass,
  expectedTotalByLabel,
}: {
  variants: PolicyVariant[];
  pass: "nonagentic" | "agentic";
  expectedTotalByLabel: Map<string, number>;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {variants.map((v) => {
        const judged: JudgePass = v[pass];
        const total = judged.criteriaVerdicts.length;
        const compliant = judged.criteriaVerdicts.filter((c) => c.verdict === "COMPLIANT").length;
        const expected = expectedTotalByLabel.get(v.label) ?? total;
        const incomplete = total > 0 && total < expected;
        return (
          <span
            key={v.label}
            className={
              incomplete
                ? "inline-flex items-center gap-1 text-xs rounded-full bg-amber-100 px-2 py-0.5 tabular-nums text-amber-800 dark:bg-amber-950/50 dark:text-amber-300"
                : "inline-flex items-center gap-1 text-xs rounded-full bg-slate-100 px-2 py-0.5 tabular-nums text-slate-700 dark:bg-slate-800 dark:text-slate-300"
            }
            title={
              incomplete
                ? `${v.label}: judge only returned ${total} of the ${expected} criteria seen elsewhere for this policy — likely an incomplete judge response, not a policy difference.`
                : v.label
            }
          >
            {total > 0 ? `${compliant}/${total} compliant${incomplete ? " ⚠" : ""}` : "—"}
          </span>
        );
      })}
    </div>
  );
}
