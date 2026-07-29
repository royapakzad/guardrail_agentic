import "server-only";
import { put, type PutBlobResult } from "@vercel/blob";

/** Same pattern as lib/db/client.ts::getSql() — fail with a clear message up
 * front rather than letting the SDK's own (less specific) error surface. */
function requireBlobToken(): string {
  const token = process.env.BLOB_READ_WRITE_TOKEN;
  if (!token) {
    throw new Error(
      "BLOB_READ_WRITE_TOKEN is not set. Connect Vercel Blob storage in the Vercel project settings, or set BLOB_READ_WRITE_TOKEN locally for testing."
    );
  }
  return token;
}

export async function uploadDatasetFile(
  useCase: string,
  filename: string,
  content: string
): Promise<PutBlobResult> {
  const token = requireBlobToken();
  const pathname = `datasets/${useCase}/${Date.now()}-${filename}`;
  return put(pathname, content, {
    access: "public",
    contentType: "application/json",
    token,
  });
}
