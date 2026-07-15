import { NextResponse } from "next/server";

/** Converts a thrown DB error into a clean JSON response instead of an opaque
 * 500 with no body — matters most for the "DATABASE_URL is not set" case,
 * which is expected until a real Postgres database is connected. */
export function dbErrorResponse(error: unknown) {
  const message = error instanceof Error ? error.message : "Unexpected database error";
  return NextResponse.json({ error: message }, { status: 500 });
}
