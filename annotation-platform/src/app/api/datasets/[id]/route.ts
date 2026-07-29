import { NextRequest, NextResponse } from "next/server";
import { deleteDataset, getDataset } from "@/lib/db/queries";
import { deleteDatasetFile } from "@/lib/blob";
import { dbErrorResponse } from "@/lib/db/apiError";

// No FK from annotations/gold_labels/code_applications to datasets.id -- see
// deleteDataset()'s comment in lib/db/queries.ts. Deleting a dataset here is
// safe with no cascade/confirmation step: it only drops it from the picker
// list (and its backing blob), never annotator work.
export async function DELETE(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id: idParam } = await params;
  const id = Number(idParam);
  if (!Number.isInteger(id)) {
    return NextResponse.json({ error: "Invalid dataset id" }, { status: 400 });
  }

  try {
    const dataset = await getDataset(id);
    if (!dataset) {
      return NextResponse.json({ error: "Dataset not found" }, { status: 404 });
    }

    // Best-effort: the blob may already be gone (manual cleanup, expired
    // token, etc.) -- don't let that block removing the DB row the picker
    // actually reads from.
    try {
      await deleteDatasetFile(dataset.blob_url);
    } catch (error) {
      console.error(`Failed to delete blob for dataset ${id}:`, error);
    }

    await deleteDataset(id);
    return NextResponse.json({ ok: true });
  } catch (error) {
    return dbErrorResponse(error);
  }
}
