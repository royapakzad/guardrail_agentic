import Link from "next/link";
import { USE_CASES, getRecordsForDataset } from "@/lib/adapters";
import { resolveDefaultDatasetId } from "@/lib/datasetSelection";

// Reads live "most recent upload" state from Postgres — must not be frozen as
// a static page at build time (see compare/page.tsx for the same issue).
export const dynamic = "force-dynamic";

const DESCRIPTIONS: Record<string, string> = {
  humanitarian: "Asylum, migration, and refugee-support scenarios.",
  financial: "Consumer finance and lending scenarios.",
  cybersecurity: "Phishing and social-engineering scenarios.",
};

export default async function Home() {
  const cards = await Promise.all(
    USE_CASES.map(async (useCase) => {
      const datasetId = await resolveDefaultDatasetId(useCase);
      const records = await getRecordsForDataset(useCase, datasetId);
      return { useCase, count: records.length, isSeed: datasetId === "seed" };
    })
  );

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight dark:text-slate-100">Use cases</h1>
        <p className="mt-1 text-slate-600 dark:text-slate-400 text-sm max-w-2xl">
          Pick a use case to see its quantitative dashboard, browse scenarios, and submit annotations. Each
          use case can load multiple uploaded result files — see{" "}
          <Link href="/compare" className="underline">Compare</Link> for a cross-use-case view.
        </p>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {cards.map(({ useCase, count, isSeed }) => (
          <Link
            key={useCase}
            href={`/${useCase}`}
            className="rounded-lg border border-slate-200 bg-white p-5 hover:border-slate-400 transition-colors dark:border-slate-700 dark:bg-slate-900 dark:hover:border-slate-500"
          >
            <div className="text-lg font-medium capitalize dark:text-slate-100">{useCase}</div>
            <div className="mt-1 text-sm text-slate-500 dark:text-slate-400">{DESCRIPTIONS[useCase]}</div>
            <div className="mt-3 text-xs text-slate-400 dark:text-slate-500 tabular-nums">
              {isSeed ? "No batch run uploaded yet" : `${count} scenarios loaded (latest upload)`}
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
