import { notFound } from "next/navigation";
import Link from "next/link";
import { USE_CASES } from "@/lib/adapters";
import { UseCaseNav } from "@/lib/ui/UseCaseNav";
import type { UseCase } from "@/lib/types";

function isUseCase(value: string): value is UseCase {
  return (USE_CASES as string[]).includes(value);
}

export function generateStaticParams() {
  return USE_CASES.map((useCase) => ({ useCase }));
}

export default async function HelpPage({ params }: { params: Promise<{ useCase: string }> }) {
  const { useCase: useCaseParam } = await params;
  if (!isUseCase(useCaseParam)) notFound();
  const useCase = useCaseParam;

  return (
    <div className="flex flex-col gap-8 max-w-3xl">
      <UseCaseNav useCase={useCase} />

      <div>
        <h1 className="text-2xl font-semibold tracking-tight capitalize">{useCase} help</h1>
        <p className="mt-1 text-sm text-slate-600">
          What this platform is for, and how to review a scenario end to end.
        </p>
      </div>

      <nav className="flex flex-wrap gap-1.5 text-xs">
        {[
          ["what-is-this", "What is this platform?"],
          ["reviewing-a-scenario", "Reviewing a scenario"],
          ["compliance", "What “compliant” means"],
          ["qualitative-coding", "Qualitative coding & the codebook"],
          ["dashboard", "The dashboard"],
          ["export", "Exporting your data"],
        ].map(([href, label]) => (
          <a key={href} href={`#${href}`} className="rounded-full bg-slate-100 px-2.5 py-1 text-slate-700 hover:bg-slate-200">
            {label}
          </a>
        ))}
      </nav>

      <HelpSection id="what-is-this" title="What is this platform?">
        <p>
          An automated judge evaluates each scenario twice — once <strong>non-agentic</strong> (reading the full policy,
          no tools) and once <strong>agentic</strong> (split by criterion, verifying claims with tools like URL checks,
          acronym lookups, and web search). This platform is where a human annotator reviews that judge&apos;s work:
          agree or disagree with its verdicts, note why, and apply qualitative codes to specific text — building a
          record that captures both what the judge decided and what a person thinks about that decision.
        </p>
      </HelpSection>

      <HelpSection id="reviewing-a-scenario" title="Reviewing a scenario">
        <p>
          Open any row from the <Link href={`/${useCase}/scenarios`} className="underline">scenarios</Link> list. Each
          scenario page is organized into three numbered steps:
        </p>
        <ol className="list-decimal list-inside flex flex-col gap-1 pl-2">
          <li><strong>The case</strong> — the scenario prompt and the assistant&apos;s response, side by side.</li>
          <li><strong>Judge evaluation</strong> — the automated judge&apos;s verdicts, explanations, and (for the agentic pass) its tool call log, URL checks, and claim checks. Read this before forming your own view.</li>
          <li>
            <strong>Your review</strong> — one form where you record your structured judgment (agree/disagree, evidence
            source, confidence, free-text notes) <em>and</em> apply qualitative codes to specific quotes, then save
            everything with a single &ldquo;Save review&rdquo; click. Saved reviews from every annotator appear above
            the form, each with Edit and Delete.
          </li>
        </ol>
        <p>
          If a scenario has more than one policy variant, use the pill buttons above step 1 to switch — your review is
          saved per variant, so you can review the same scenario under different policies independently.
        </p>
      </HelpSection>

      <HelpSection id="compliance" title="What &ldquo;compliant&rdquo; means">
        <p>
          Each policy criterion gets one of two verdicts: <strong>COMPLIANT</strong> (the response satisfies this
          criterion) or <strong>NOT FULLY COMPLIANT</strong> (it doesn&apos;t, fully). Criteria tagged with a blue
          &ldquo;tool&rdquo; badge are ones where the agentic pass could plausibly verify the claim with a tool and
          therefore land on a different verdict than the non-agentic pass — that&apos;s a <em>flip</em>. Criteria without
          that tag always carry the non-agentic verdict forward unchanged, by design.
        </p>
      </HelpSection>

      <HelpSection id="qualitative-coding" title="Qualitative coding & the codebook">
        <p>
          Qualitative coding here follows standard thematic-analysis practice: a <strong>code</strong> is a short label
          with a definition (when does it apply?), an optional example quote, and an optional <strong>theme</strong> that
          groups related codes. The <strong>codebook</strong> is the shared, evolving set of codes for this use case —
          it&apos;s expected to grow and be refined as coding proceeds (this is normal &ldquo;open coding&rdquo;
          practice, not a sign the codebook was incomplete to start).
        </p>
        <p>To apply a code while reviewing a scenario:</p>
        <ol className="list-decimal list-inside flex flex-col gap-1 pl-2">
          <li>In step 3 (&ldquo;Your review&rdquo;), pick a code from the dropdown, or choose <strong>&ldquo;+ Add a new code&hellip;&rdquo;</strong> if the right one doesn&apos;t exist yet — this opens an inline form to create it without leaving the page.</li>
          <li>Say which part of the response/explanation you&apos;re coding, and paste the specific quote.</li>
          <li>Add as many code rows as you need with &ldquo;+ Add another code&rdquo;, then save the whole review together.</li>
        </ol>
        <p>
          On the <Link href={`/${useCase}/codebook`} className="underline">codebook page</Link> you can search codes,
          jump between themes, see a frequency chart of how often each code has been used, and click a code&apos;s
          &ldquo;N applied&rdquo; badge to see exactly which scenarios used it. Codes can be edited or deleted there too
          — deleting a code that&apos;s already been applied will ask you to confirm, since it also removes those
          applications.
        </p>
      </HelpSection>

      <HelpSection id="dashboard" title="The dashboard">
        <p>
          The <Link href={`/${useCase}`} className="underline">dashboard</Link> aggregates the judge&apos;s own
          quantitative output across every scenario in the current dataset: compliance by criterion, verdict flip
          rates, tool usage, latency, token usage, domains touched during tool use, and — at the bottom — a frequency
          chart of every qualitative code annotators have applied. It reflects whichever dataset is selected in the
          picker at the top; upload a new batch run there to see fresh numbers.
        </p>
      </HelpSection>

      <HelpSection id="export" title="Exporting your data">
        <p>
          Click <strong>Export</strong> in the top navigation (on any page) to download a CSV of everything saved for
          this use case: one row per scenario × policy variant × annotator, joining your structured review and
          qualitative codes with the judge&apos;s own quantitative data for that row (compliance counts, tool calls
          made, judgment time, token totals for both passes). Nothing is duplicated into storage ahead of time — the
          export is built fresh from the live dataset and your saved reviews each time you download it.
        </p>
      </HelpSection>
    </div>
  );
}

function HelpSection({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="flex flex-col gap-3 scroll-mt-4">
      <h2 className="text-lg font-semibold text-slate-900 border-b border-slate-200 pb-2">{title}</h2>
      <div className="flex flex-col gap-3 text-sm text-slate-700 leading-relaxed">{children}</div>
    </section>
  );
}
