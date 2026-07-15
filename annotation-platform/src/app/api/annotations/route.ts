import { NextRequest, NextResponse } from "next/server";
import { insertAnnotation, listAnnotations } from "@/lib/db/queries";
import { dbErrorResponse } from "@/lib/db/apiError";
import { USE_CASES } from "@/lib/adapters";
import type { UseCase } from "@/lib/types";
import { CONFIDENCE_LEVELS, DEDUCTION_REASON_CATEGORIES, EVIDENCE_SOURCE_TYPES } from "@/lib/annotationOptions";

function isUseCase(value: unknown): value is UseCase {
  return typeof value === "string" && (USE_CASES as string[]).includes(value);
}

export async function GET(request: NextRequest) {
  const useCase = request.nextUrl.searchParams.get("useCase");
  const scenarioId = request.nextUrl.searchParams.get("scenarioId");

  if (!isUseCase(useCase) || !scenarioId) {
    return NextResponse.json({ error: "useCase and scenarioId query params are required" }, { status: 400 });
  }

  try {
    const annotations = await listAnnotations(useCase, scenarioId);
    return NextResponse.json({ annotations });
  } catch (error) {
    return dbErrorResponse(error);
  }
}

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const { scenarioId, useCase, language, policyLabel, annotatorName } = body;
  if (!scenarioId || !isUseCase(useCase) || !language || !policyLabel || !annotatorName) {
    return NextResponse.json(
      { error: "scenarioId, useCase, language, policyLabel, and annotatorName are required" },
      { status: 400 }
    );
  }

  if (body.evidenceSourceType && !EVIDENCE_SOURCE_TYPES.includes(body.evidenceSourceType)) {
    return NextResponse.json({ error: "Invalid evidenceSourceType" }, { status: 400 });
  }
  if (body.deductionReasonCategory && !DEDUCTION_REASON_CATEGORIES.includes(body.deductionReasonCategory)) {
    return NextResponse.json({ error: "Invalid deductionReasonCategory" }, { status: 400 });
  }
  if (body.confidence && !CONFIDENCE_LEVELS.includes(body.confidence)) {
    return NextResponse.json({ error: "Invalid confidence" }, { status: 400 });
  }

  try {
    const annotation = await insertAnnotation({
      scenarioId: String(scenarioId),
      useCase,
      language: String(language),
      policyLabel: String(policyLabel),
      annotatorName: String(annotatorName),
      agreesWithVerdict: typeof body.agreesWithVerdict === "boolean" ? body.agreesWithVerdict : null,
      disagreementReason: body.disagreementReason ? String(body.disagreementReason) : null,
      evidenceSourceType: body.evidenceSourceType ?? null,
      deductionReasonCategory: body.deductionReasonCategory ?? null,
      evidentiaryAttributionPresent:
        typeof body.evidentiaryAttributionPresent === "boolean" ? body.evidentiaryAttributionPresent : null,
      freeText: body.freeText ? String(body.freeText) : null,
      confidence: body.confidence ?? null,
    });

    return NextResponse.json({ annotation }, { status: 201 });
  } catch (error) {
    return dbErrorResponse(error);
  }
}
