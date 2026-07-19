import { NextRequest, NextResponse } from "next/server";
import { insertCodebookCode, listCodebookCodes } from "@/lib/db/queries";
import { dbErrorResponse } from "@/lib/db/apiError";
import { USE_CASES } from "@/lib/adapters";
import type { UseCase } from "@/lib/types";

function isUseCase(value: unknown): value is UseCase {
  return typeof value === "string" && (USE_CASES as string[]).includes(value);
}

export async function GET(request: NextRequest) {
  const useCase = request.nextUrl.searchParams.get("useCase");
  if (!isUseCase(useCase)) {
    return NextResponse.json({ error: "useCase query param is required" }, { status: 400 });
  }

  try {
    const codes = await listCodebookCodes(useCase);
    return NextResponse.json({ codes });
  } catch (error) {
    return dbErrorResponse(error);
  }
}

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const { useCase, name, definition, createdBy } = body;
  if (!isUseCase(useCase) || !name || !definition || !createdBy) {
    return NextResponse.json(
      { error: "useCase, name, definition, and createdBy are required" },
      { status: 400 }
    );
  }

  try {
    const code = await insertCodebookCode({
      useCase,
      name: String(name),
      definition: String(definition),
      exampleQuote: body.exampleQuote ? String(body.exampleQuote) : null,
      theme: body.theme ? String(body.theme) : null,
      createdBy: String(createdBy),
    });
    return NextResponse.json({ code }, { status: 201 });
  } catch (error) {
    return dbErrorResponse(error);
  }
}
