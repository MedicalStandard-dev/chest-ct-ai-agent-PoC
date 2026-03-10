# SKILLS.md — Token-efficient Engineering Mode (Medical AI PACS)

## Primary goal
Minimize tokens while maximizing code correctness and safety.
Default to actionable diffs and commands over long explanations.

## Response format (strict)
1) **What to change** (1–5 bullets)
2) **Patch** (code blocks / file paths / diffs)
3) **How to run** (exact commands)
4) **How to verify** (expected log lines/metrics)

No long prose. No repetition.

## Token control rules
- Ask **no clarifying questions** if a safe default exists; choose defaults and proceed.
- Prefer checklists, not paragraphs.
- If output is long, provide:
  - file-level diffs
  - and a short rationale (<=5 bullets)

## Debug-first discipline (for ML)
Always validate these before “more training”:
- shape/spacing/orientation alignment (CT vs mask vs heatmap)
- label format (binary / one-hot / class index)
- tiny overfit test on 2–3 samples (train Dice should reach ~0.9 if pipeline is correct)
- visualization overlay (CT + GT + Pred)

## 3070 Ti (8GB) training defaults
- AMP: ON
- ROI: 96^3 (fallback 64^3 if OOM)
- batch: 1
- grad_accum: 2–8
- sliding window inference for large volumes
- dataloader workers: 2–4 (avoid CPU bottleneck)
- cache_rate: low or smart caching (avoid RAM blowups)

## Safety enforcement rules (never violate)
- LLM must remain rewrite-only; never let it modify tables/numbers/findings.
- If validator fails → output template report (fail-closed).
- Keep “DRAFT — Requires physician confirmation” in every report.

## Code style
- Python 3.10+
- Type hints everywhere
- Pydantic schemas stable
- Centralized config in config/settings.py
- Logging: concise + structured; include critical invariants

## Testing rules
- At least 1 integration test: mock vision + mock solar → end-to-end report.
- Add a smoke test for:
  - transform pipeline output shapes
  - validator table integrity
  - rag patient isolation

## Commit/PR discipline
- One PR = one theme (debug, training config, validator, rag).
- Include “How to reproduce” in PR description.

## Ready-to-use short commands (preferred)
When proposing commands, provide copy-paste blocks, e.g.:
- create venv, install deps
- run debug tool
- run train script
- run eval script
- run tests
