import { NextRequest, NextResponse } from "next/server";
import { updateAnnotation, deleteAnnotation } from "@/lib/db/queries";
import { dbErrorResponse } from "@/lib/db/apiError";
import {
  CONFIDENCE_LEVELS,
  DEDUCTION_REASON_CATEGORIES,
  EVIDENCE_SOURCE_TYPES,
  JUDGMENT_ALIGNMENT_OPTIONS,
} from "@/lib/annotationOptions";

export async function PATCH(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id: idParam } = await params;
  const id = Number(idParam);
  if (!Number.isInteger(id)) {
    return NextResponse.json({ error: "Invalid annotation id" }, { status: 400 });
  }

  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  if (body.evidenceSourceType && !EVIDENCE_SOURCE_TYPES.includes(body.evidenceSourceType)) {
    return NextResponse.json({ error: "Invalid evidenceSourceType" }, { status: 400 });
  }
  if (body.deductionReasonCategory && !DEDUCTION_REASON_CATEGORIES.includes(body.deductionReasonCategory)) {
    return NextResponse.json({ error: "Invalid deductionReasonCategory" }, { status: 400 });
  }
  if (body.confidence && !CONFIDENCE_LEVELS.includes(body.confidence)) {
    return NextResponse.json({ error: "Invalid confidence" }, { status: 400 });
  }
  if (body.judgmentAlignmentEn && !JUDGMENT_ALIGNMENT_OPTIONS.includes(body.judgmentAlignmentEn)) {
    return NextResponse.json({ error: "Invalid judgmentAlignmentEn" }, { status: 400 });
  }
  if (body.judgmentAlignmentNonEn && !JUDGMENT_ALIGNMENT_OPTIONS.includes(body.judgmentAlignmentNonEn)) {
    return NextResponse.json({ error: "Invalid judgmentAlignmentNonEn" }, { status: 400 });
  }

  try {
    const annotation = await updateAnnotation({
      id,
      evidenceSourceType: body.evidenceSourceType ?? null,
      deductionReasonCategory: body.deductionReasonCategory ?? null,
      judgmentAlignmentEn: body.judgmentAlignmentEn ?? null,
      alignmentExplanationEn: body.alignmentExplanationEn ? String(body.alignmentExplanationEn) : null,
      judgmentAlignmentNonEn: body.judgmentAlignmentNonEn ?? null,
      alignmentExplanationNonEn: body.alignmentExplanationNonEn ? String(body.alignmentExplanationNonEn) : null,
      freeText: body.freeText ? String(body.freeText) : null,
      confidence: body.confidence ?? null,
    });
    if (!annotation) {
      return NextResponse.json({ error: "Annotation not found" }, { status: 404 });
    }
    return NextResponse.json({ annotation });
  } catch (error) {
    return dbErrorResponse(error);
  }
}

export async function DELETE(_request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id: idParam } = await params;
  const id = Number(idParam);
  if (!Number.isInteger(id)) {
    return NextResponse.json({ error: "Invalid annotation id" }, { status: 400 });
  }

  try {
    const deleted = await deleteAnnotation(id);
    if (!deleted) {
      return NextResponse.json({ error: "Annotation not found" }, { status: 404 });
    }
    return NextResponse.json({ ok: true });
  } catch (error) {
    return dbErrorResponse(error);
  }
}
