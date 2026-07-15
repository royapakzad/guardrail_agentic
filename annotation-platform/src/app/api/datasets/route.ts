import { NextRequest, NextResponse } from "next/server";
import { getRecordCountForRawJson, USE_CASES } from "@/lib/adapters";
import { uploadDatasetFile } from "@/lib/blob";
import { insertDataset, listDatasets } from "@/lib/db/queries";
import { dbErrorResponse } from "@/lib/db/apiError";
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
    const datasets = await listDatasets(useCase);
    return NextResponse.json({ datasets });
  } catch (error) {
    return dbErrorResponse(error);
  }
}

export async function POST(request: NextRequest) {
  const form = await request.formData().catch(() => null);
  if (!form) {
    return NextResponse.json({ error: "Expected multipart/form-data" }, { status: 400 });
  }

  const useCase = form.get("useCase");
  const uploadedBy = form.get("uploadedBy");
  const file = form.get("file");

  if (!isUseCase(useCase) || typeof uploadedBy !== "string" || !uploadedBy.trim() || !(file instanceof File)) {
    return NextResponse.json({ error: "useCase, uploadedBy, and file are required" }, { status: 400 });
  }

  const text = await file.text();
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return NextResponse.json({ error: "File is not valid JSON" }, { status: 400 });
  }

  // Run it through the exact same adapter the dashboard will use — validates
  // "does this file parse into records for this use case" and "will this file
  // actually render" with one check, not two that can drift apart.
  let recordCount: number;
  try {
    recordCount = getRecordCountForRawJson(useCase, raw);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "File could not be parsed for this use case" },
      { status: 400 }
    );
  }
  if (recordCount === 0) {
    return NextResponse.json(
      { error: "File parsed but produced 0 scenario records — is this the right use case?" },
      { status: 400 }
    );
  }

  try {
    const blob = await uploadDatasetFile(useCase, file.name, text);
    const dataset = await insertDataset({
      useCase,
      filename: file.name,
      blobUrl: blob.url,
      uploadedBy: uploadedBy.trim(),
      recordCount,
    });
    return NextResponse.json({ dataset }, { status: 201 });
  } catch (error) {
    return dbErrorResponse(error);
  }
}
