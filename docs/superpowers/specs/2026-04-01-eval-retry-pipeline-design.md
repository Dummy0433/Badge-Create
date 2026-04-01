# Badge Generation Eval & Retry Pipeline - Design Spec

## Overview

Add a quality evaluation loop after Seedream image generation. A multimodal LLM (GPT-5.4) evaluates generated images against structured criteria derived from the datamining input, reference good/bad examples, and fixed badge style requirements. Failed evaluations trigger targeted prompt adjustments and retry, with a reroll fallback.

## Scope

This spec covers:
- `eval_store.py` — eval reference library (good/bad example images + descriptions)
- `eval_client.py` — GPT-5.4 vision-based structured image evaluation
- `prompt_builder.py` — template-based prompt assembly from datamining JSON + LLM photo analysis
- `orchestrator.py` — generate → eval → retry/reroll loop

Out of scope (future work):
- LLM photo analysis step (extracting `photo_analysis` from anchor_photo) — for now, `photo_analysis` is provided manually or hardcoded
- Database-backed eval_store / reference_store
- Frontend integration of orchestrator (runs as a script/API, logs to console)

## Input Format

The system receives a datamining JSON. All fields pass through the pipeline unchanged and serve as both generation input and eval criteria.

```json
{
  "text_output": "Wells",
  "anchor_photo": "<base64 or file path>",
  "anchor_characterization": "This Jordanian anchor hosts an energetic and highly interactive 'Just Chatting' stream. He engages directly with his audience... signature blue hoodie...",
  "brand_palette": {
    "primary": {"name": "Gold", "hex": "#D4AF37"},
    "secondary": {"name": "Vibrant Blue", "hex": "#007BFF"},
    "tertiary": {"name": "Black", "hex": "#000000"}
  },
  "photo_analysis": {
    "gender": "male",
    "hair": "dark black side-parted short hair",
    "eyes": "deep brown eyes",
    "expression": "gentle smile",
    "skin_tone": "medium olive"
  }
}
```

`photo_analysis` will be provided manually for now. In the future, an LLM vision step will auto-extract it from `anchor_photo`.

## Component 1: prompt_builder.py

### Template Structure

The prompt is assembled from fixed template segments + variable slots filled from the input JSON.

**Fixed segments** (never change):
- Render style: `C4D Badge, 3D Pixar realistic cartoon style`
- Heart material: `thick and voluminous like soft candy or jelly, smooth glossy surface`
- Character composition: `positioned in front of the upper area of the heart from chest up, occupying 70% of the heart`
- Text material: `silver chrome sweep light effect, holographic iridescent metallic material`
- Lighting: `Warm side light from left, cool side light from right, soft front key light`
- Color style: `candy color palette, commercial art illustration style`

**Variable slots** (filled from input JSON):

| Slot | Source |
|------|--------|
| Heart color | `brand_palette.primary` |
| Background color | `brand_palette.tertiary` |
| Character appearance | `photo_analysis.*` |
| Character clothing | Extracted from `anchor_characterization` |
| Decoration elements | Derived from `anchor_characterization` themes |
| Text content | `text_output` |
| Text color tint | `brand_palette.primary` |

### Interface

```python
def build_prompt(input_data: dict) -> str:
    """Assemble full Seedream prompt from datamining input."""
    ...

def build_negative_prompt() -> str:
    """Return fixed negative prompt."""
    return "nsfw, balloon, inflatable, wrinkles, seams, strings, rope, flat, 2D, thin, metal frame, hard border, badge pin, deflated"
```

### Pre-check

Before returning the assembled prompt, validate:
1. `text_output` is non-empty and appears in the assembled prompt
2. `brand_palette` colors are referenced in the prompt
3. `photo_analysis` character traits are present in the prompt

Raise `PromptValidationError` if any check fails.

## Component 2: eval_store.py

Same pattern as `reference_store.py`: hardcoded now, database later.

### Data Structure

```python
@dataclass
class EvalReference:
    image_path: str
    description: str
    is_good: bool       # True = positive example, False = negative example
    score: float        # Expected score (e.g., 9.0 for good, 3.0 for bad)
    issues: list[str]   # Why it's bad (empty for good examples)
```

### Current Store

- 1 good example image + description + score (~9.0)
- 1 bad example image + description + score (~3.0) + issues list

Future: 3 good + 3 bad, randomly sampled from dataset.

### Interface

```python
def pick_eval_references(good_count: int = 1, bad_count: int = 1) -> tuple[list[EvalReference], list[EvalReference]]:
    """Return (good_refs, bad_refs) randomly sampled."""
    ...
```

## Component 3: eval_client.py

### GPT-5.4 Vision Evaluation

Sends to GPT-5.4:
1. The generated image
2. 1 good reference image + its description/score
3. 1 bad reference image + its description/score/issues
4. The original datamining input JSON (as eval criteria)
5. A structured eval system prompt

### Eval Dimensions (7)

| Dimension | What GPT-5.4 checks | Criteria source |
|-----------|---------------------|-----------------|
| `heart_carrier` | Heart shape, material, glossy candy feel | Fixed template |
| `character` | Person matches photo_analysis, correct positioning | `photo_analysis` |
| `decorations` | Style-appropriate floating elements | `anchor_characterization` |
| `text_render` | Text content = `text_output`, metallic material, bottom position | `text_output` |
| `color_match` | Heart/text/accent colors match brand_palette | `brand_palette` |
| `composition` | Overall layout, background = tertiary color, 70% character | Fixed template + `brand_palette.tertiary` |
| `quality` | No artifacts, no deformed limbs, no blur | General quality |

Each dimension scored 1-10. Total score = average of all dimensions.

### Response Format

```python
@dataclass
class EvalResult:
    passed: bool              # total_score >= 8.0
    total_score: float
    dimensions: dict[str, float]   # dimension_name → score
    issues: list[str]
    suggestion: str           # targeted fix suggestion for prompt adjustment
```

### Interface

```python
class EvalClient:
    def __init__(self, model: str = "gpt-5.4"):
        ...

    def evaluate(
        self,
        generated_image: bytes,
        input_data: dict,
        good_refs: list[EvalReference],
        bad_refs: list[EvalReference],
    ) -> EvalResult:
        ...
```

### OpenAI Configuration

- API key from `.env` file via `python-dotenv`
- Model: `gpt-5.4`
- Response format: JSON mode for structured output

## Component 4: orchestrator.py

### Pipeline Flow

```
Input JSON
    │
    ▼
prompt_builder.build_prompt()
    │
    ├─ Pre-check fails → raise PromptValidationError
    │
    ▼
Round 0: original prompt → Seedream generate → eval
    │
    ├─ passed (score ≥ 8) → return image
    │
    ├─ failed → LLM adjusts prompt based on suggestion
    │           (always patches ORIGINAL prompt, not previous adjustment)
    │
    ▼
Round 1: adjusted prompt → new seed → generate → eval
    │
    ├─ passed → return image
    │
    ├─ failed → LLM adjusts again (from original + accumulated feedback)
    │
    ▼
Round 2: adjusted prompt → new seed → generate → eval
    │
    ├─ passed → return image
    │
    ├─ failed → REROLL: LLM rewrites prompt from scratch
    │           (same input JSON, completely new expansion)
    │
    ▼
Round 3 (reroll): new prompt → new seed → generate → eval
    │
    ├─ passed → return image
    │
    └─ failed → return best image (highest score across all rounds)
```

### Prompt Adjustment Rules

- Adjustments are always based on: original prompt + eval suggestion
- The adjustment LLM receives: original prompt, failing dimensions with scores, suggestion text
- It outputs a modified prompt with targeted changes only to the failing areas
- Fixed template segments (render style, lighting, composition rules) must NOT be modified

### Reroll Rules

- Reroll means: call `prompt_builder.build_prompt()` again, but with a different system prompt that asks for a different creative expansion
- Same input JSON, same variable slots, but different wording/emphasis
- This is NOT a prompt adjustment — it's a fresh assembly

### Interface

```python
@dataclass
class OrchestrationResult:
    image: bytes
    score: float
    passed: bool
    rounds: int                    # how many attempts
    prompt_history: list[str]      # prompts used in each round
    eval_history: list[EvalResult] # eval results per round

class Orchestrator:
    def __init__(
        self,
        seedream_client: SeedreamClient,
        eval_client: EvalClient,
        max_retries: int = 2,
        max_rerolls: int = 1,
        pass_threshold: float = 8.0,
    ):
        ...

    def run(self, input_data: dict) -> OrchestrationResult:
        """Full pipeline: build prompt → generate → eval → retry/reroll."""
        ...
```

### Logging

All rounds logged to Python logger (INFO level):
- Round number, prompt used, seed
- Eval scores per dimension
- Issues and suggestions
- Final result (passed/failed, best score, total rounds)

No eval details exposed to frontend users.

## Seedream Generation Parameters

Use the tested optimal defaults for badge generation:

| Parameter | Value |
|-----------|-------|
| `guidance_scale` | 8.0 |
| `cfg_rescale_factor` | 0.0 |
| `single_edit_guidance_weight` | 2.0 |
| `single_edit_guidance_weight_image` | 1.0 |
| `cot_mode` | "enable" |
| `width` / `height` | 2048 |
| `force_single` | true |

Reference images (from `reference_store`) are injected as before when `use_refs=true`.

## File Structure

```
project/
├── prompt_builder.py      # Template-based prompt assembly + pre-check
├── eval_store.py          # Good/bad reference images for eval
├── eval_client.py         # GPT-5.4 vision eval client
├── orchestrator.py        # Generate → eval → retry/reroll loop
├── eval_references/       # Good/bad example images (gitignored)
├── .env                   # OPENAI_API_KEY
├── seedream_sdk.py        # (existing)
├── reference_store.py     # (existing) generation style references
├── server.py              # (existing, unchanged for now)
└── static/index.html      # (existing, unchanged for now)
```

## Dependencies

New:
- `openai>=1.0.0`
- `python-dotenv>=1.0.0`

## Error Handling

- `PromptValidationError` — pre-check fails, abort before generation
- `SeedreamAPIError` — Seedream call fails, counts as a failed round, retry with new seed
- `openai` errors — eval call fails, skip eval for that round, treat as failed
- All errors logged, never surfaced to end user
