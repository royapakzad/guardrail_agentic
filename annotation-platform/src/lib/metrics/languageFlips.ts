import type { EvaluationRecord, UseCase } from "@/lib/types";
import { round } from "./flatten";

export type LanguageFlipRow = {
  id: string;
  enScore: number | null;
  otherScore: number | null;
  delta: number | null; // otherScore - enScore
  enValid: boolean | null;
  otherValid: boolean | null;
  flipped: boolean; // valid/invalid disagrees between languages
};

export type LanguageFlipSummary = {
  otherLanguage: string;
  pairedScenarioCount: number;
  meanAbsDelta: number | null;
  flippedCount: number;
  flipRate: number | null;
  rows: LanguageFlipRow[];
};

// Each use case names its language-paired policy prefix differently — see
// analysis_improvemnets/live_scenario_qualitative_quantitative_analysis.md §2.2
// for why the coverage itself is asymmetric (humanitarian=fa, financial=es).
// This picks the single most content-relevant policy pair per use case (the
// domain-specific explicit policy, not the shared generic fallback) since
// that's the pair language differences would actually show up in.
const PAIR_CONFIG: Partial<
  Record<UseCase, { otherLanguage: string; enLabel: (model: string | null) => string; otherLabel: (model: string | null) => string }>
> = {
  humanitarian: {
    otherLanguage: "fa",
    enLabel: (model) => `humanitarian_policy_explicit${model ? `_${model}` : ""}`,
    otherLabel: (model) => `humanitarian_policy_explicit_fa${model ? `_${model}` : ""}`,
  },
  financial: {
    otherLanguage: "es",
    enLabel: () => "financial_policy",
    otherLabel: () => "financial_policy_es",
  },
};

export function computeLanguageFlips(useCase: UseCase, records: EvaluationRecord[]): LanguageFlipSummary | null {
  const config = PAIR_CONFIG[useCase];
  if (!config) return null;

  const enById = new Map(records.filter((r) => r.language === "en").map((r) => [r.id, r]));
  const otherById = new Map(
    records.filter((r) => r.language === config.otherLanguage).map((r) => [r.id, r])
  );

  const rows: LanguageFlipRow[] = [];
  for (const [id, enRecord] of enById) {
    const otherRecord = otherById.get(id);
    if (!otherRecord) continue;

    // Try every judge-model suffix present on the en record's plain-language
    // variants, since humanitarian logs multiple judge models per scenario.
    const modelCandidates = [...new Set(enRecord.policyVariants.map((v) => v.judgeModel))];

    for (const model of modelCandidates) {
      const enLabel = config.enLabel(model);
      const otherLabel = config.otherLabel(model);
      const enVariant = enRecord.policyVariants.find((v) => v.label === enLabel);
      const otherVariant = otherRecord.policyVariants.find((v) => v.label === otherLabel);
      if (!enVariant || !otherVariant) continue;

      const enScore = enVariant.agentic.score ?? enVariant.nonagentic.score;
      const otherScore = otherVariant.agentic.score ?? otherVariant.nonagentic.score;
      const enValid = enVariant.agentic.valid ?? enVariant.nonagentic.valid;
      const otherValid = otherVariant.agentic.valid ?? otherVariant.nonagentic.valid;

      rows.push({
        id: model ? `${id} (${model})` : id,
        enScore,
        otherScore,
        delta: enScore !== null && otherScore !== null ? otherScore - enScore : null,
        enValid,
        otherValid,
        flipped: enValid !== null && otherValid !== null && enValid !== otherValid,
      });
    }
  }

  const deltas = rows.map((r) => r.delta).filter((d): d is number => d !== null);
  const flippedCount = rows.filter((r) => r.flipped).length;

  return {
    otherLanguage: config.otherLanguage,
    pairedScenarioCount: rows.length,
    meanAbsDelta: deltas.length > 0 ? round(deltas.reduce((a, b) => a + Math.abs(b), 0) / deltas.length) : null,
    flippedCount,
    flipRate: rows.length > 0 ? round(flippedCount / rows.length) : null,
    rows,
  };
}
