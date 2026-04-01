# Eval System Redesign

**Date:** 2026-04-02
**Status:** Approved
**Scope:** Rewrite eval_client.py scoring system: remove useless dimensions, add hard gates, add input photo comparison

## Problem

Current 7-dimension eval system has three categories of failure:

1. **Self-consistent but useless** (always high score, zero discrimination): `color_match`, `quality`, `decorations` (as currently defined)
2. **Counterproductive** (too strict for what GPT can actually verify): `text_render` exact-match requirement on stylized fonts
3. **Missing critical checks**: heart shape integrity (horns/wings), pure black background, character likeness compared to actual input photo (not just text description)

Result: scores drift 6-8 randomly, eval doesn't catch real problems (background not black, heart deformed, character doesn't resemble anchor) but fails on false positives (text that's actually correct).

## Design

### Remove entirely

| Dimension | Why |
|-----------|-----|
| `color_match` | Seedream follows prompt colors. Almost always 9. Zero discrimination. |
| `quality` | Seedream output is technically clean. Almost always 8-9. Zero discrimination. |
| `text_render` | GPT cannot reliably read stylized/cursive text. Becomes noise (always fails even when correct). Text presence check moved to `composition`. |

### Add: Hard gates (pass/fail, any FAIL = image rejected)

| Gate | Criteria |
|------|----------|
| `background_black` | Background must be solid black. Any visible gradient, glow, color cast, or non-black area → FAIL. |
| `heart_shape_clean` | Heart must be a clean rounded 3D shape. Any attached protrusions (horns, wings, tails, extra shapes growing from heart) → FAIL. Floating elements near but not attached to the heart are OK. |

### Keep: Scored dimensions (equal weight, 1-10 each)

| Dimension | What it evaluates | Key change from current |
|-----------|-------------------|------------------------|
| `character_likeness` | Compare generated character to the **input anchor photo** (sent as image). Check: gender match, hair style/color, facial features (beard, glasses, moles, skin tone), age impression, expression. | Currently compares to text description only. New: GPT does visual comparison against actual photo. |
| `heart_quality` | Heart material (candy/jelly, glossy surface, not balloon/plastic), size (occupies ~80% canvas), color from brand palette. | Refined from old `heart_carrier`. Shape check moved to hard gate. |
| `decoration_harmony` | Decorations should be restrained and unified: prefer single-color or minimal-color doodle/outline style. Penalize: too many different materials, too many colors, overcrowded, competing for attention with character. | Old dimension only checked "do decorations exist?" — always high. New: checks restraint and harmony. |
| `composition` | Character framed chest-up occupying ~70% of heart, text present at bottom (not oversized, not obscuring character), overall visual balance. | Absorbs text position check from removed `text_render`. No longer requires exact text content match. |

### Pass/fail logic

```
if background_black == FAIL or heart_shape_clean == FAIL:
    passed = False
    total_score = 0  (hard gate failure)
else:
    total_score = avg(character_likeness, heart_quality, decoration_harmony, composition)
    passed = total_score >= 8.0
```

No weights — all 4 scored dimensions are equally important. Total score is simple average.

### Input photo in eval

The eval currently receives:
- Good/bad reference images (for quality calibration)
- Generated image (to evaluate)
- `input_data` dict (text fields: text_output, brand_palette, photo_analysis, anchor_characterization)

**New**: also receives `anchor_photo_bytes` (the original input photo). This is added as an image in the GPT message with the label "Anchor photo — the character should resemble this person".

### Updated eval prompt

The system prompt changes from 7 scored dimensions to: 2 hard gates + 4 scored dimensions. The response JSON format changes to:

```json
{
  "hard_gates": {
    "background_black": true,
    "heart_shape_clean": true
  },
  "dimensions": {
    "character_likeness": 8.5,
    "heart_quality": 9.0,
    "decoration_harmony": 7.5,
    "composition": 8.0
  },
  "issues": ["list of specific problems"],
  "suggestion": "targeted fix suggestion"
}
```

`passed` and `total_score` are computed programmatically from this response, not by GPT.

### File changes

| File | Change |
|------|--------|
| `eval_client.py` | Rewrite `EVAL_SYSTEM_PROMPT`, update `evaluate()` to accept `anchor_photo`, update `_build_user_message()` to include anchor photo, update `_parse_response()` for new JSON format, remove `DIMENSION_WEIGHTS` |
| `orchestrator.py` | Pass `photo_bytes` through to `eval_client.evaluate()` |
| `tests/test_eval_client.py` | Update tests for new dimensions, hard gates, and anchor photo parameter |

### What stays the same

- `eval_store.py` — good/bad reference images unchanged
- `EvalResult` dataclass — same fields (passed, total_score, dimensions, issues, suggestion)
- Eval is still called per-image in the orchestrator pipeline
- Good/bad reference images still sent for quality calibration
