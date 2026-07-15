import { NextRequest, NextResponse } from "next/server";
import { getGoldLabel, upsertGoldLabel } from "@/lib/db/queries";
import { dbErrorResponse } from "@/lib/db/apiError";
import { USE_CASES } from "@/lib/adapters";
import type { UseCase } from "@/lib/types";

function isUseCase(value: unknown): value is UseCase {
  return typeof value === "string" && (USE_CASES as string[]).includes(value);
}

export async function GET(request: NextRequest) {
  const useCase = request.nextUrl.searchParams.get("useCase");
  const scenarioId = request.nextUrl.searchParams.get("scenarioId");
  const language = request.nextUrl.searchParams.get("language") ?? "en";

  if (!isUseCase(useCase) || !scenarioId) {
    return NextResponse.json({ error: "useCase and scenarioId query params are required" }, { status: 400 });
  }

  try {
    const goldLabel = await getGoldLabel(useCase, scenarioId, language);
    return NextResponse.json({ goldLabel });
  } catch (error) {
    return dbErrorResponse(error);
  }
}

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const { scenarioId, useCase, language, goldVerdict, createdBy } = body;
  if (!scenarioId || !isUseCase(useCase) || !language || !goldVerdict || !createdBy) {
    return NextResponse.json(
      { error: "scenarioId, useCase, language, goldVerdict, and createdBy are required" },
      { status: 400 }
    );
  }

  try {
    const goldLabel = await upsertGoldLabel({
      scenarioId: String(scenarioId),
      useCase,
      language: String(language),
      goldVerdict: String(goldVerdict),
      goldNotes: body.goldNotes ? String(body.goldNotes) : null,
      createdBy: String(createdBy),
    });

    return NextResponse.json({ goldLabel }, { status: 201 });
  } catch (error) {
    return dbErrorResponse(error);
  }
}
