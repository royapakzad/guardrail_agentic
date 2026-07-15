import { notFound } from "next/navigation";
import Link from "next/link";
import { USE_CASES } from "@/lib/adapters";
import type { UseCase } from "@/lib/types";
import { UploadForm } from "./UploadForm";

function isUseCase(value: string): value is UseCase {
  return (USE_CASES as string[]).includes(value);
}

export function generateStaticParams() {
  return USE_CASES.map((useCase) => ({ useCase }));
}

export default async function UploadPage({
  params,
}: {
  params: Promise<{ useCase: string }>;
}) {
  const { useCase: useCaseParam } = await params;
  if (!isUseCase(useCaseParam)) notFound();
  const useCase = useCaseParam;

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight capitalize">Upload a {useCase} run</h1>
        <p className="mt-1 text-sm text-slate-600">
          <Link href={`/${useCase}`} className="underline">back to dashboard</Link>
        </p>
      </div>
      <UploadForm useCase={useCase} />
    </div>
  );
}
