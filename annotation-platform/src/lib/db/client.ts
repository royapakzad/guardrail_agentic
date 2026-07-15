import "server-only";
import { neon, type NeonQueryFunction } from "@neondatabase/serverless";

// @neondatabase/serverless's neon() tagged-template client speaks Neon's HTTP
// query protocol — it only works against a real Neon/Vercel Postgres database,
// not a plain local Postgres instance. schema.sql was verified separately
// against a throwaway local Postgres container for SQL correctness; the HTTP
// transport itself is exercised at deploy time once a real database is wired up.
let cached: NeonQueryFunction<false, false> | null = null;

export function getSql(): NeonQueryFunction<false, false> {
  if (cached) return cached;
  const url = process.env.DATABASE_URL;
  if (!url) {
    throw new Error(
      "DATABASE_URL is not set. Connect a Postgres database (Vercel Postgres / Neon) in the Vercel project settings, or set DATABASE_URL locally for testing."
    );
  }
  cached = neon(url);
  return cached;
}
