# CLAUDE.md — Medical AI PACS (CT Chest) Project Rules

## Project identity
This repository implements a **product-style AI agent for PACS**:
- **Vision (MONAI)** produces structured, evidence-first outputs.
- **LLM (Upstage Solar Pro 3)** is **rewrite-only** (narrative polishing). It must NOT diagnose or infer.

This is a **product PoC** emphasizing safety, auditability, and integration readiness.

## Non-negotiable safety principles
1) **Evidence-first**
   - Every finding must be backed by Vision output evidence (slice references, mask paths, ids).
2) **Table-first**
   - Numeric measurements live **only in tables**.
   - LLM must NEVER modify tables or numbers.
3) **Validator-gated (fail-closed)**
   - If validation fails: return the **template report** (safe fallback).
4) **LLM cannot diagnose**
   - No definitive diagnosis, normal assertion, treatment recommendation.
   - Must include: **"DRAFT — Requires physician confirmation"**.

## Current architecture (Option 1)
Pipeline:
CT DICOM(zip) →
Preprocess (HU/spacing/orientation) →
Lung Segmentation →
Nodule Detection (heatmap) →
Candidate Processing →
Measurements + Evidence →
RAG (Solar Embedding + ChromaDB; patient isolation) →
Template-first report →
Solar Pro 3 rewrite-only →
Validator →
Return report + structured_ai_result

## Repository layout (canonical)
medical-ai-pacs/
- api/
  - main.py
  - schemas.py
- monai_pipeline/
  - preprocessing.py
  - lung_segmentation.py
  - nodule_detection.py
  - candidate_processor.py
  - evidence_generator.py
  - production_pipeline.py
- solar_integration/
  - embeddings.py
  - rag_system.py
  - templates.py
  - rewriter.py
  - validator.py
  - tracking.py
- config/settings.py
- scripts/
  - debug_lung_segmentation.py
  - (train/eval scripts)
- data/
  - chroma_db/
- models/
  - lung_seg/
  - nodule_det/

## GPU/compute constraints (important)
Environment: **RTX 3070 Ti (8GB VRAM)** single GPU.

Guidelines:
- 3D training uses **ROI patching** (e.g., 96^3 or 128^3), batch size 1–2, AMP enabled.
- Prefer **gradient accumulation** over increasing batch.
- Avoid full-volume training/inference where possible; use sliding window.

## Data assumptions
Primary datasets:
- Lung Seg: MSD Task06_Lung
- Nodule Det: LIDC-IDRI (subset ~200 cases)

Splits:
- patient-level split (train/val/test) with fixed seed recorded in config.

## Output schema expectations (must keep stable)
Structured AI Result must include:
- quality: slice_thickness_mm, coverage_score, artifact_score (proxy ok)
- nodules: candidate list with center_zyx, bbox_zyx, diameter_mm, volume_mm3, confidence, evidence
- findings: multi-label present/absent/uncertain with prob
- versioning: model_version, thresholds, pipeline_version

## RAG rules (patient isolation)
- If `patient_id` is missing → **no retrieval**
- Collections are separated per patient_id (or strict metadata filter)
- Prior comparison:
  1) rule-based lesion tracking first
  2) semantic search as helper only
- All retrieved items must be recorded in sources/audit

## LLM (Solar Pro 3) rules (rewrite-only)
- Only rewrite narrative text outside tables.
- Must preserve table blocks **byte-identical** (table-protected).
- Must not add/remove findings.
- Must not change any numbers.
- Must not infer diagnosis.

## Dev workflow expectations
- When fixing bugs: add a minimal test or a reproducible script.
- Prefer small, reviewable changes.
- Log critical invariants (shape/spacing/orientation alignment).

## Definition of done (for tasks)
- Runs on a clean venv with documented commands.
- Produces deterministic outputs with seed/config.
- Validator enforces policy (fail-closed).
- Includes at least one integration test in mock mode.
