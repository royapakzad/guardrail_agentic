"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { UseCase } from "@/lib/types";

/**
 * Persistent sub-navigation for everything scoped to one use case --
 * replaces the ad hoc, differently-worded inline links each page used to
 * carry on its own (dashboard said "browse scenarios & annotate", scenarios
 * said "back to dashboard", etc.), so orientation is the same everywhere.
 */
export function UseCaseNav({ useCase, datasetId }: { useCase: UseCase; datasetId?: string }) {
  const pathname = usePathname();

  const links: { href: string; label: string }[] = [
    { href: `/${useCase}`, label: "Dashboard" },
    { href: `/${useCase}/scenarios`, label: "Scenarios" },
    { href: `/${useCase}/codebook`, label: "Codebook" },
    { href: `/${useCase}/help`, label: "Help" },
  ];

  const exportHref = `/api/export?useCase=${useCase}${datasetId ? `&dataset=${encodeURIComponent(datasetId)}` : ""}`;

  return (
    <nav className="flex flex-wrap items-center gap-1.5 border-b border-slate-200 dark:border-slate-700 pb-3 text-sm">
      {links.map((l) => {
        const active = pathname === l.href;
        return (
          <Link
            key={l.href}
            href={l.href}
            aria-current={active ? "page" : undefined}
            className={
              active
                ? "rounded-full bg-slate-900 px-3 py-1 font-medium text-white dark:bg-slate-100 dark:text-slate-900"
                : "rounded-full px-3 py-1 text-slate-600 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800"
            }
          >
            {l.label}
          </Link>
        );
      })}
      <a
        href={exportHref}
        className="rounded-full px-3 py-1 text-slate-600 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800"
        title="Download all saved reviews (structured answers + qualitative codes + quantitative judge data) as a CSV"
      >
        ⬇ Export
      </a>
    </nav>
  );
}
