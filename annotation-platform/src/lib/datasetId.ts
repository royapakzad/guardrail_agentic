// Client-safe: no server-only imports, no Node builtins. `lib/adapters/index.ts`
// (server-only) re-exports these; client components that need them (e.g.
// DatasetPicker) must import from here directly, not from lib/adapters,
// or the whole server module graph (fs, Postgres client) gets pulled into
// the browser bundle.
export const SEED_DATASET_ID = "seed" as const;
export type DatasetId = typeof SEED_DATASET_ID | number;
