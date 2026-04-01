# Unified Pipeline Architecture Design

**Date:** 2026-04-01
**Status:** Approved
**Scope:** Refactor server.py + orchestrator.py into 5 clean modules with unified retry logic

## Problem

CLI (`run_orchestrator.py`) and Web UI (`server.py`) implement the same pipeline with different logic:

- `orchestrator.py`: has retry/reroll but lacks batch, ref injection, concurrency
- `server.py` `/api/pipeline`: has batch, refs, concurrency but no retry

The prompt building chain (analyze -> assemble -> expand) is duplicated in both. Input preprocessing is embedded in orchestrator. Result: features diverge, bugs get fixed in one place but not the other.

## Design

### Module Boundaries

Five modules, each with a single responsibility:

```
① input_processor.py  →  ② prompt_builder.py  →  ③ generator.py  →  ④ eval_client.py
                                    ↑                                        |
                                    └──── ⑤ orchestrator.py (retry) ────────┘
```

#### Module 1: `input_processor.py` — Input Normalization

**Responsibility:** Collect and normalize input from any source (frontend form, external API, CLI). Ensure downstream modules always receive the same shape of data.

**Public API:**
```python
def process(raw: dict) -> dict:
    """Normalize raw input to standardized internal format.
    
    Accepts both nested datamining format and flat format.
    Validates required fields are present.
    Returns standardized dict with keys:
        text_output, anchor_photo, anchor_characterization,
        brand_palette, anchor_nickname, anchor_bio,
        community_type, slogan_lang
    """
```

**Source:** `preprocess_input()` moves here from `orchestrator.py`. Server.py stops importing it from orchestrator.

**Future:** Input prompt optimization (e.g., slogan cleanup, language detection) goes in this module.

#### Module 2: `prompt_builder.py` — Analyze + Assemble + Expand

**Responsibility:** Turn standardized input into a Seedream-ready prompt. All LLM prompt logic lives here.

**Public API:**
```python
def analyze_photo(client, photo_bytes: bytes) -> dict:
    """GPT vision: extract appearance features from anchor photo."""

def assemble_keywords(client, input_data: dict) -> dict:
    """GPT: map input fields to structured keywords JSON."""

def validate_keywords(keywords: dict) -> None:
    """Check assembled keywords JSON has all required fields.
    Raises PromptValidationError if invalid."""

def expand_prompt(client, keywords: dict) -> str:
    """GPT: expand keywords into full Seedream prompt."""

def build_prompt(client, input_data: dict, photo_bytes: bytes | None = None) -> tuple[dict, str]:
    """Full pipeline: analyze (if photo) -> assemble -> validate -> expand.
    Returns (keywords, prompt) so orchestrator can re-expand from keywords."""

def build_negative_prompt() -> str:
    """Return fixed negative prompt."""
```

**Key change:** `build_prompt()` returns `(keywords, prompt)` tuple. Orchestrator keeps the keywords so Level 2 retry can call `expand_prompt(keywords)` without redoing analysis + assembly.

**Validation:** `validate_keywords()` is new — checks the assembled JSON before expansion. Currently validation only happens after expansion (`validate_prompt()`); catching issues earlier is cheaper.

#### Module 3: `generator.py` — Image Generation

**Responsibility:** Take a prompt + images, call Seedream, return image bytes. Handles ref injection and PE construction.

**Public API:**
```python
class Generator:
    def __init__(self, seedream_client: SeedreamClient):
        ...

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int = -1,
        photo_bytes: bytes | None = None,
        refs: list[ReferenceImage] | None = None,
        width: int = 2048,
        height: int = 2048,
        **kwargs,
    ) -> GenerateResult:
        """Generate one image.
        
        Handles ref injection (prepend ref images before photo),
        PE construction (pre_llm_result), and Seedream API call.
        Returns GenerateResult with image bytes + metadata.
        """
```

**Source:** `_inject_references()` and `_build_ref_kwargs()` move here from `server.py`. `SeedreamClient` remains as the low-level HTTP wrapper; `Generator` is the higher-level interface that knows about refs and PE.

**`GenerateResult` dataclass:**
```python
@dataclass
class GenerateResult:
    image: bytes          # First generated image
    seed: int
    request_id: str
    llm_result: str       # VLM PE result
    raw_images: list[bytes]  # All images if multiple
```

#### Module 4: `eval_client.py` — Evaluation (unchanged)

Already isolated. No changes to public API.

```python
class EvalClient:
    def evaluate(self, generated_image, input_data, good_refs, bad_refs) -> EvalResult

class EvalResult:
    passed: bool          # score >= 8.0
    total_score: float
    dimensions: dict
    issues: list[str]
    suggestion: str
```

#### Module 5: `orchestrator.py` — Orchestration + Retry + Batch

**Responsibility:** Wire modules 1-4 together. Own the retry logic. Manage concurrency for batch.

**Public API:**
```python
@dataclass
class UnitResult:
    image: bytes
    score: float
    passed: bool
    seed: int
    prompt: str
    eval_result: EvalResult
    rounds: int              # How many attempts this unit took
    request_id: str

@dataclass 
class BatchResult:
    results: list[UnitResult]   # Sorted by score descending
    prompt: str                 # Initial expanded prompt
    keywords: dict              # For reference
    total: int
    success: int                # Units that passed threshold
    failed: int                 # Units that exhausted retries

class Orchestrator:
    def __init__(
        self,
        llm_client,                      # OpenAI client for prompt_builder calls
        generator: Generator,
        eval_client: EvalClient,
        max_retries: int = 2,        # Level 1: re-generate
        max_reexpands: int = 1,      # Level 2: re-expand + generate
        pass_threshold: float = 8.0,
        max_workers: int = 10,
    ):
        ...

    def run_batch(
        self,
        input_data: dict,
        photo_bytes: bytes | None = None,
        count: int = 1,
        use_refs: bool = True,
        ref_count: int = 2,
    ) -> BatchResult:
        """Main entry point for both single and batch.
        
        1. input_processor.process(input_data)
        2. prompt_builder.build_prompt() -> (keywords, prompt)
        3. ThreadPoolExecutor(count) x _run_single_unit()
        4. Sort results by score
        """

    def _run_single_unit(
        self, prompt, negative_prompt, keywords, photo_bytes, refs, input_data
    ) -> UnitResult:
        """One atomic pipeline unit with retry escalation.
        
        Round 0:   generate(prompt) -> eval
        Level 1:   generate(prompt, new_seed) -> eval  (up to max_retries)
        Level 2:   expand(keywords) -> generate -> eval (up to max_reexpands)
        Exhaust:   return best scoring result
        """
```

### Retry Escalation Detail

```
_run_single_unit(prompt, keywords, ...):

  best = None

  # Initial attempt
  image = generator.generate(prompt, seed=random)
  eval = eval_client.evaluate(image)
  if eval.passed → return
  best = track_best(image, eval)

  # Level 1: cheap retries (new seed, same prompt)
  for i in range(max_retries):          # default: 2
      image = generator.generate(prompt, seed=random)
      eval = eval_client.evaluate(image)
      if eval.passed → return
      best = track_best(image, eval)

  # Level 2: re-expand prompt from same keywords
  for i in range(max_reexpands):        # default: 1
      new_prompt = prompt_builder.expand_prompt(llm_client, keywords)
      image = generator.generate(new_prompt, seed=random)
      eval = eval_client.evaluate(image)
      if eval.passed → return
      best = track_best(image, eval)

  # All exhausted
  return best
```

Maximum cost per unit: 1 + 2 + 1 = 4 Seedream calls + 4 eval calls + 1 LLM expand call.
For batch of 10: worst case 40 Seedream + 40 eval + 10 expand.

### Server.py Changes

Server becomes a thin HTTP adapter:

```python
# server.py — only HTTP concerns

@app.post("/api/pipeline")
async def pipeline(
    input_json: str = Form(...),
    count: int = Form(10),
    use_refs: bool = Form(True),
    ref_count: int = Form(2),
    anchor_photo: UploadFile | None = File(None),
):
    photo_bytes = await anchor_photo.read() if anchor_photo else None
    input_data = json.loads(input_json)

    result = orchestrator.run_batch(
        input_data=input_data,
        photo_bytes=photo_bytes,
        count=count,
        use_refs=use_refs,
        ref_count=ref_count,
    )

    return _format_batch_response(result)
```

Existing endpoints `/api/generate`, `/api/generate_batch`, `/api/generate_sweep` can remain as lower-level direct-access endpoints for advanced usage, but `/api/pipeline` and `/api/pipeline_sweep` will call orchestrator.

### Sweep Mode

`/api/pipeline_sweep` calls `orchestrator.run_sweep()`:

```python
def run_sweep(
    self,
    input_data: dict,
    photo_bytes: bytes | None,
    param_combos: list[dict],  # [{guidance_scale: 8.0, cfg: 0.0, ...}, ...]
    use_refs: bool = True,
    ref_count: int = 2,
) -> BatchResult:
    """Like run_batch but each unit uses different generation params.
    Each combination still gets full retry escalation."""
```

### File Changes Summary

| File | Action |
|------|--------|
| `input_processor.py` | **New.** `preprocess_input()` from orchestrator.py + field validation |
| `prompt_builder.py` | **Modify.** Add `validate_keywords()`, update `build_prompt()` to return `(keywords, prompt)` |
| `generator.py` | **New.** `Generator` class with ref injection + PE construction from server.py |
| `eval_client.py` | **No change** |
| `eval_store.py` | **No change** |
| `reference_store.py` | **No change** |
| `seedream_sdk.py` | **No change** (stays as low-level HTTP client) |
| `orchestrator.py` | **Rewrite.** New `Orchestrator` with `run_batch()`, `run_sweep()`, retry escalation |
| `server.py` | **Simplify.** Remove business logic, thin HTTP adapter calling orchestrator |
| `run_orchestrator.py` | **Simplify.** Call `orchestrator.run_batch(count=1)` |
| `tests/` | **Update** to match new module boundaries |

### What Gets Deleted

- `server.py`: `_inject_references()`, `_build_ref_kwargs()`, prompt building in pipeline endpoints, ThreadPoolExecutor logic, eval logic
- `orchestrator.py`: `preprocess_input()`, `_adjust_prompt()`, `_reroll_prompt()`, `ADJUST_SYSTEM_PROMPT`, `REROLL_SYSTEM_PROMPT` (replaced by simpler re-expand)
- The entire "adjust prompt based on eval feedback" approach (replaced by retry escalation)

### What's NOT Changing

- `seedream_sdk.py` — low-level HTTP client, untouched
- `eval_client.py` — already clean
- `eval_store.py` — reference image management
- `reference_store.py` — few-shot references  
- `static/index.html` — frontend unchanged (same API contract for `/api/pipeline`)
- API response format for `/api/pipeline` — stays compatible so frontend works without changes
