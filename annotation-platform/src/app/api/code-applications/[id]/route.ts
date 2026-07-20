import { NextRequest, NextResponse } from "next/server";
import { updateCodeApplication, deleteCodeApplication } from "@/lib/db/queries";
import { dbErrorResponse } from "@/lib/db/apiError";

export async function PATCH(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id: idParam } = await params;
  const id = Number(idParam);
  if (!Number.isInteger(id)) {
    return NextResponse.json({ error: "Invalid code application id" }, { status: 400 });
  }

  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  try {
    const application = await updateCodeApplication({
      id,
      quoteText: body.quoteText ? String(body.quoteText) : null,
      note: body.note ? String(body.note) : null,
    });
    if (!application) {
      return NextResponse.json({ error: "Code application not found" }, { status: 404 });
    }
    return NextResponse.json({ application });
  } catch (error) {
    return dbErrorResponse(error);
  }
}

export async function DELETE(_request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id: idParam } = await params;
  const id = Number(idParam);
  if (!Number.isInteger(id)) {
    return NextResponse.json({ error: "Invalid code application id" }, { status: 400 });
  }

  try {
    const deleted = await deleteCodeApplication(id);
    if (!deleted) {
      return NextResponse.json({ error: "Code application not found" }, { status: 404 });
    }
    return NextResponse.json({ ok: true });
  } catch (error) {
    return dbErrorResponse(error);
  }
}
