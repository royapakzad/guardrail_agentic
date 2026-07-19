import { NextRequest, NextResponse } from "next/server";
import { updateCodebookCode } from "@/lib/db/queries";
import { dbErrorResponse } from "@/lib/db/apiError";

// Codebooks are meant to evolve iteratively (the standard thematic-analysis
// method revises definitions as coding proceeds) -- this lets an annotator
// refine a code's definition/example/theme without deleting and re-creating
// it, which would orphan any code_applications already pointing at it.
export async function PATCH(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id: idParam } = await params;
  const id = Number(idParam);
  if (!Number.isInteger(id)) {
    return NextResponse.json({ error: "Invalid code id" }, { status: 400 });
  }

  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  try {
    const code = await updateCodebookCode({
      id,
      definition: body.definition ? String(body.definition) : undefined,
      exampleQuote: "exampleQuote" in body ? (body.exampleQuote ? String(body.exampleQuote) : null) : undefined,
      theme: "theme" in body ? (body.theme ? String(body.theme) : null) : undefined,
    });
    if (!code) {
      return NextResponse.json({ error: "Code not found" }, { status: 404 });
    }
    return NextResponse.json({ code });
  } catch (error) {
    return dbErrorResponse(error);
  }
}
