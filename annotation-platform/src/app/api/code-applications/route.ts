import { NextRequest, NextResponse } from "next/server";
import { insertCodeApplication, listCodeApplications } from "@/lib/db/queries";
import { dbErrorResponse } from "@/lib/db/apiError";
import { USE_CASES } from "@/lib/adapters";
import type { UseCase } from "@/lib/types";

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
    const applications = await listCodeApplications(useCase, scenarioId);
    return NextResponse.json({ applications });
  } catch (error) {
    return dbErrorResponse(error);
  }
}

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const { scenarioId, useCase, language, policyLabel, annotatorName, codeId, targetField } = body;
  if (!scenarioId || !isUseCase(useCase) || !language || !policyLabel || !annotatorName || !codeId || !targetField) {
    return NextResponse.json(
      {
        error:
          "scenarioId, useCase, language, policyLabel, annotatorName, codeId, and targetField are required",
      },
      { status: 400 }
    );
  }

  const codeIdNum = Number(codeId);
  if (!Number.isInteger(codeIdNum)) {
    return NextResponse.json({ error: "codeId must be an integer" }, { status: 400 });
  }

  try {
    const application = await insertCodeApplication({
      scenarioId: String(scenarioId),
      useCase,
      language: String(language),
      policyLabel: String(policyLabel),
      annotatorName: String(annotatorName),
      codeId: codeIdNum,
      targetField: String(targetField),
      quoteText: body.quoteText ? String(body.quoteText) : null,
      note: body.note ? String(body.note) : null,
    });
    return NextResponse.json({ application }, { status: 201 });
  } catch (error) {
    return dbErrorResponse(error);
  }
}
