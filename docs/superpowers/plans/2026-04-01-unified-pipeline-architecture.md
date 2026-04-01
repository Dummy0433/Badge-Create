# Unified Pipeline Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify CLI and Web UI into 5 clean modules with shared retry escalation logic.

**Architecture:** Extract input processing into `input_processor.py`, keep prompt logic in `prompt_builder.py`, create `generator.py` for Seedream+refs, rewrite `orchestrator.py` with retry escalation + batch + sweep, slim `server.py` to HTTP-only layer.

**Tech Stack:** Python 3.11+, FastAPI, pytest, unittest.mock

**Spec:** `docs/superpowers/specs/2026-04-01-unified-pipeline-architecture-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `input_processor.py` | **Create** | Normalize raw input from any source |
| `tests/test_input_processor.py` | **Create** | Tests for input normalization |
| `prompt_builder.py` | **Modify** | Add `validate_keywords()`, update `build_prompt()` return type |
| `tests/test_prompt_builder.py` | **Modify** | Add keyword validation tests, update `build_prompt` tests |
| `generator.py` | **Create** | `Generator` class: ref injection + PE + Seedream call |
| `tests/test_generator.py` | **Create** | Tests for Generator |
| `orchestrator.py` | **Rewrite** | Retry escalation + batch + sweep using modules 1-4 |
| `tests/test_orchestrator.py` | **Rewrite** | Tests for new orchestrator |
| `server.py` | **Simplify** | Thin HTTP adapter calling orchestrator |
| `tests/test_server.py` | **Modify** | Update pipeline tests to mock orchestrator |
| `run_orchestrator.py` | **Simplify** | Call `orchestrator.run_batch(count=1)` |

---

### Task 1: Create `input_processor.py`

Extract `preprocess_input()` from `orchestrator.py` into its own module with field validation.

**Files:**
- Create: `input_processor.py`
- Create: `tests/test_input_processor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_input_processor.py
import pytest
from input_processor import process, InputValidationError


NESTED_INPUT = {
    "slogan": "Diedriht",
    "community_type": "host",
    "slogan_lang": "Spanish",
    "anchor_info": {
        "anchor": {
            "nick_name": "Diedriht Mayorga",
            "bio_description": "Bienvenidos ami live",
        },
        "anchor_characterization": "Spanish-speaking anchor with playful style",
        "brand_palette": {
            "primary": {"name": "Black", "hex": "#000000"},
            "secondary": {"name": "Red", "hex": "#FF0000"},
            "tertiary": {"name": "White", "hex": "#FFFFFF"},
        },
    },
}

FLAT_INPUT = {
    "text_output": "Wells",
    "anchor_characterization": "interactive streamer",
    "brand_palette": {
        "primary": {"name": "Gold", "hex": "#D4AF37"},
        "secondary": {"name": "Blue", "hex": "#007BFF"},
        "tertiary": {"name": "Black", "hex": "#000000"},
    },
    "anchor_nickname": "Wells",
    "anchor_bio": "Just chatting",
    "community_type": "host",
    "slogan_lang": "English",
}


class TestProcessNestedInput:
    def test_flattens_nested_format(self):
        result = process(NESTED_INPUT)
        assert result["text_output"] == "Diedriht"
        assert result["anchor_characterization"] == "Spanish-speaking anchor with playful style"
        assert result["brand_palette"]["secondary"]["hex"] == "#FF0000"
        assert result["anchor_nickname"] == "Diedriht Mayorga"
        assert result["anchor_bio"] == "Bienvenidos ami live"
        assert result["community_type"] == "host"
        assert result["slogan_lang"] == "Spanish"

    def test_passes_through_flat_format(self):
        result = process(FLAT_INPUT)
        assert result["text_output"] == "Wells"
        assert result["anchor_characterization"] == "interactive streamer"
        assert result["anchor_nickname"] == "Wells"


class TestProcessValidation:
    def test_missing_slogan_and_text_output_raises(self):
        bad = {"anchor_info": {"anchor": {}, "anchor_characterization": "x", "brand_palette": {}}}
        with pytest.raises(InputValidationError, match="text_output"):
            process(bad)

    def test_missing_brand_palette_raises(self):
        bad = {"slogan": "Test", "anchor_info": {"anchor": {}, "anchor_characterization": "x"}}
        with pytest.raises(InputValidationError, match="brand_palette"):
            process(bad)

    def test_empty_slogan_raises(self):
        bad = {**NESTED_INPUT, "slogan": "  "}
        with pytest.raises(InputValidationError, match="text_output"):
            process(bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_input_processor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'input_processor'`

- [ ] **Step 3: Implement `input_processor.py`**

```python
# input_processor.py
"""Input normalization for badge generation pipeline.

Accepts raw input from frontend forms, external APIs, or CLI,
and normalizes it to a standardized internal format.
"""

import logging

logger = logging.getLogger(__name__)


class InputValidationError(Exception):
    """Raised when input data is missing required fields."""
    pass


def process(raw: dict) -> dict:
    """Normalize raw input to standardized internal format.

    Accepts both nested datamining format (with anchor_info)
    and flat internal format (with text_output).
    Validates required fields are present.
    """
    # Already in internal format
    if "text_output" in raw and "anchor_characterization" in raw:
        result = dict(raw)
    else:
        # Nested datamining format
        anchor_info = raw.get("anchor_info", {})
        anchor = anchor_info.get("anchor", {})
        result = {
            "text_output": raw.get("slogan", ""),
            "anchor_photo": raw.get("anchor_photo", ""),
            "anchor_characterization": anchor_info.get("anchor_characterization", ""),
            "brand_palette": anchor_info.get("brand_palette", {}),
            "anchor_nickname": anchor.get("nick_name", ""),
            "anchor_bio": anchor.get("bio_description", ""),
            "community_type": raw.get("community_type", ""),
            "slogan_lang": raw.get("slogan_lang", ""),
        }

    # Validate required fields
    text_output = result.get("text_output", "").strip()
    if not text_output:
        raise InputValidationError("text_output is empty or missing")
    result["text_output"] = text_output

    if not result.get("brand_palette"):
        raise InputValidationError("brand_palette is missing")

    logger.info("Input processed: text_output=%s", result["text_output"])
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_input_processor.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add input_processor.py tests/test_input_processor.py
git commit -m "feat: extract input_processor module from orchestrator"
```

---

### Task 2: Update `prompt_builder.py`

Add `validate_keywords()` and change `build_prompt()` to return `(keywords, prompt)`.

**Files:**
- Modify: `prompt_builder.py`
- Modify: `tests/test_prompt_builder.py`

- [ ] **Step 1: Write failing tests for `validate_keywords` and new `build_prompt` return type**

Add to `tests/test_prompt_builder.py`:

```python
from prompt_builder import validate_keywords


VALID_KEYWORDS = {
    "character": {
        "gender": "male",
        "hair": "dark hair",
        "eyes": "brown eyes",
        "expression": "smile",
        "clothing": "blue hoodie",
    },
    "heart_carrier": {
        "color_name": "Gold",
        "color_hex": "#D4AF37",
        "material": "soft candy or jelly",
    },
    "decorations": {
        "elements": ["music note", "chat bubble"],
    },
    "text": {
        "content": "Wells",
        "color_tint": "#007BFF",
    },
    "background_color": "#000000",
}


class TestValidateKeywords:
    def test_valid_keywords_passes(self):
        validate_keywords(VALID_KEYWORDS)  # Should not raise

    def test_missing_character_raises(self):
        bad = {**VALID_KEYWORDS}
        del bad["character"]
        with pytest.raises(PromptValidationError, match="character"):
            validate_keywords(bad)

    def test_missing_heart_carrier_raises(self):
        bad = {**VALID_KEYWORDS}
        del bad["heart_carrier"]
        with pytest.raises(PromptValidationError, match="heart_carrier"):
            validate_keywords(bad)

    def test_missing_text_raises(self):
        bad = {**VALID_KEYWORDS}
        del bad["text"]
        with pytest.raises(PromptValidationError, match="text"):
            validate_keywords(bad)

    def test_empty_text_content_raises(self):
        bad = {**VALID_KEYWORDS, "text": {"content": "", "color_tint": "#000"}}
        with pytest.raises(PromptValidationError, match="text.content"):
            validate_keywords(bad)


class TestBuildPromptReturnType:
    @patch("prompt_builder._get_client", return_value=None)
    def test_returns_keywords_and_prompt_tuple(self, _mock):
        keywords, prompt = build_prompt(SAMPLE_INPUT)
        assert isinstance(keywords, dict)
        assert isinstance(prompt, str)
        assert "C4D Badge" in prompt
        assert '"Wells"' in prompt
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `pytest tests/test_prompt_builder.py -v`
Expected: FAIL — `ImportError` for `validate_keywords`, tuple unpacking fails for `build_prompt`

- [ ] **Step 3: Add `validate_keywords()` to `prompt_builder.py`**

Add after `validate_prompt()` (after line 217):

```python
def validate_keywords(keywords: dict) -> None:
    """Validate assembled keywords JSON has all required fields.

    Raises PromptValidationError if critical fields are missing.
    """
    required_top = ["character", "heart_carrier", "text"]
    for key in required_top:
        if key not in keywords:
            raise PromptValidationError(f"Missing required keyword section: {key}")

    text_content = keywords.get("text", {}).get("content", "").strip()
    if not text_content:
        raise PromptValidationError("text.content is empty in assembled keywords")
```

- [ ] **Step 4: Update `build_prompt()` to return `(keywords, prompt)` tuple**

Replace the existing `build_prompt` function (lines 219-245) with:

```python
def build_prompt(input_data: dict, client: openai.OpenAI | None = None) -> tuple[dict, str]:
    """Full pipeline: assemble keywords → validate → expand → validate prompt.

    Returns (keywords, prompt) so caller can re-expand from keywords on retry.
    Falls back to template mode if no client provided and no env key.
    """
    text_output = input_data.get("text_output", "").strip()
    if not text_output:
        raise PromptValidationError("text_output is empty or missing")

    if not input_data.get("photo_analysis"):
        raise PromptValidationError("photo_analysis is missing")

    if not input_data.get("brand_palette"):
        raise PromptValidationError("brand_palette is missing")

    llm = client or _get_client()
    if llm:
        keywords = assemble_keywords(llm, input_data)
        validate_keywords(keywords)
        prompt = expand_prompt(llm, keywords)
        validate_prompt(prompt, input_data)
        return keywords, prompt

    # Fallback: template mode (for tests without OpenAI key)
    keywords = {
        "character": input_data.get("photo_analysis", {}),
        "heart_carrier": {
            "color_name": input_data["brand_palette"]["primary"]["name"],
            "color_hex": input_data["brand_palette"]["primary"]["hex"],
        },
        "text": {"content": text_output},
    }
    return keywords, _template_fallback(input_data)
```

- [ ] **Step 5: Update existing `TestBuildPrompt` tests to unpack tuple**

In `tests/test_prompt_builder.py`, update every test in `TestBuildPrompt` class.
Each call `build_prompt(SAMPLE_INPUT)` that previously returned a string now returns `(keywords, prompt)`.
Example — change each test method body from:

```python
prompt = build_prompt(SAMPLE_INPUT)
assert "C4D Badge" in prompt
```

to:

```python
_keywords, prompt = build_prompt(SAMPLE_INPUT)
assert "C4D Badge" in prompt
```

Apply this to all 8 methods in `TestBuildPrompt`: `test_contains_fixed_style`, `test_contains_text_output`, `test_contains_brand_colors`, `test_contains_character_traits`, `test_contains_lighting`, `test_missing_text_output_raises`, `test_missing_photo_analysis_raises`, `test_missing_brand_palette_raises`.

For the `raises` tests, the tuple unpacking doesn't happen because the exception is raised before the return.

- [ ] **Step 6: Run all prompt_builder tests**

Run: `pytest tests/test_prompt_builder.py -v`
Expected: All tests PASS (old + new)

- [ ] **Step 7: Commit**

```bash
git add prompt_builder.py tests/test_prompt_builder.py
git commit -m "feat: add validate_keywords, build_prompt returns (keywords, prompt)"
```

---

### Task 3: Create `generator.py`

Wrap `SeedreamClient` with ref injection and PE construction.

**Files:**
- Create: `generator.py`
- Create: `tests/test_generator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_generator.py
import json
from unittest.mock import MagicMock, patch
import pytest
from generator import Generator, GenerateResult
from reference_store import ReferenceImage
from seedream_sdk import SeedreamResponse


def _mock_ref(desc: str = "test ref description") -> ReferenceImage:
    ref = MagicMock(spec=ReferenceImage)
    ref.description = desc
    ref.load_bytes.return_value = b"fake-ref-image"
    return ref


def _make_seedream_response() -> SeedreamResponse:
    return SeedreamResponse(
        images=[b"fake-jpeg-data"],
        llm_result="rewritten prompt",
        request_id="req-123",
    )


class TestGeneratorBasic:
    def test_generate_t2i_no_images(self):
        sdk = MagicMock()
        sdk.generate.return_value = _make_seedream_response()
        gen = Generator(sdk)

        result = gen.generate(prompt="test prompt", negative_prompt="nsfw", seed=42)

        assert isinstance(result, GenerateResult)
        assert result.image == b"fake-jpeg-data"
        assert result.seed == 42
        assert result.request_id == "req-123"
        sdk.generate.assert_called_once()
        call_kw = sdk.generate.call_args.kwargs
        assert call_kw["prompt"] == "test prompt"
        assert call_kw["seed"] == 42
        assert call_kw["images"] is None

    def test_generate_with_photo(self):
        sdk = MagicMock()
        sdk.generate.return_value = _make_seedream_response()
        gen = Generator(sdk)

        result = gen.generate(
            prompt="test", negative_prompt="nsfw",
            photo_bytes=b"photo-data",
        )

        call_kw = sdk.generate.call_args.kwargs
        assert call_kw["images"] == [b"photo-data"]


class TestGeneratorRefInjection:
    def test_refs_prepended_before_photo(self):
        sdk = MagicMock()
        sdk.generate.return_value = _make_seedream_response()
        gen = Generator(sdk)

        refs = [_mock_ref("ref1 desc"), _mock_ref("ref2 desc")]
        gen.generate(
            prompt="test prompt", negative_prompt="nsfw",
            photo_bytes=b"photo-data", refs=refs,
        )

        call_kw = sdk.generate.call_args.kwargs
        images = call_kw["images"]
        assert len(images) == 3  # 2 refs + 1 photo
        assert images[0] == b"fake-ref-image"
        assert images[1] == b"fake-ref-image"
        assert images[2] == b"photo-data"

    def test_refs_build_pe_kwargs(self):
        sdk = MagicMock()
        sdk.generate.return_value = _make_seedream_response()
        gen = Generator(sdk)

        refs = [_mock_ref("desc A"), _mock_ref("desc B")]
        gen.generate(
            prompt="my prompt", negative_prompt="nsfw",
            photo_bytes=b"photo", refs=refs,
        )

        call_kw = sdk.generate.call_args.kwargs
        assert call_kw["use_pre_llm"] is False
        pe = json.loads(call_kw["pre_llm_result"])
        assert pe["input1"] == "desc A"
        assert pe["input2"] == "desc B"
        assert pe["output"] == "my prompt"
        assert pe["ratio"] == "1:1"

    def test_no_refs_no_pe(self):
        sdk = MagicMock()
        sdk.generate.return_value = _make_seedream_response()
        gen = Generator(sdk)

        gen.generate(prompt="test", negative_prompt="nsfw")

        call_kw = sdk.generate.call_args.kwargs
        assert "use_pre_llm" not in call_kw
        assert "pre_llm_result" not in call_kw


class TestGeneratorKwargsPassthrough:
    def test_extra_kwargs_forwarded(self):
        sdk = MagicMock()
        sdk.generate.return_value = _make_seedream_response()
        gen = Generator(sdk)

        gen.generate(
            prompt="test", negative_prompt="nsfw",
            guidance_scale=8.0, cfg_rescale_factor=0.0,
        )

        call_kw = sdk.generate.call_args.kwargs
        assert call_kw["guidance_scale"] == 8.0
        assert call_kw["cfg_rescale_factor"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'generator'`

- [ ] **Step 3: Implement `generator.py`**

```python
# generator.py
"""Image generation module wrapping Seedream SDK with reference injection."""

import json
import logging
from dataclasses import dataclass, field

from reference_store import ReferenceImage
from seedream_sdk import SeedreamClient

logger = logging.getLogger(__name__)


@dataclass
class GenerateResult:
    """Result of a single image generation."""

    image: bytes
    seed: int
    request_id: str
    llm_result: str = ""
    raw_images: list[bytes] = field(default_factory=list)


class Generator:
    """High-level image generator with reference injection and PE construction."""

    def __init__(self, seedream_client: SeedreamClient):
        self.seedream = seedream_client

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
        """Generate one image via Seedream.

        If refs are provided, prepends reference images before photo
        and builds PE (pre_llm_result) for style guidance.
        """
        images = None
        extra_kwargs = dict(kwargs)

        if refs:
            images = [ref.load_bytes() for ref in refs]
            if photo_bytes:
                images.append(photo_bytes)
            extra_kwargs.update(self._build_pe(prompt, refs))
        elif photo_bytes:
            images = [photo_bytes]

        result = self.seedream.generate(
            prompt=prompt,
            images=images,
            width=width,
            height=height,
            seed=seed,
            negative_prompt=negative_prompt,
            force_single=True,
            **extra_kwargs,
        )

        image = result.images[0] if result.images else b""
        return GenerateResult(
            image=image,
            seed=seed,
            request_id=result.request_id,
            llm_result=result.llm_result,
            raw_images=result.images,
        )

    def _build_pe(self, prompt: str, refs: list[ReferenceImage]) -> dict:
        """Build use_pre_llm + pre_llm_result kwargs from references."""
        ref_descs = ", ".join(f"input{i+1}" for i in range(len(refs)))
        pe = {}
        for i, ref in enumerate(refs, start=1):
            pe[f"input{i}"] = ref.description
        pe["edit"] = (
            f"Generate a C4D Badge 3D Pixar realistic cartoon style image, "
            f"following the exact visual style of {ref_descs} — 3D heart carrier, "
            f"character from chest up, metallic text at bottom, pure black background."
        )
        pe["output"] = prompt
        pe["ratio"] = "1:1"

        return {
            "use_pre_llm": False,
            "pre_llm_result": json.dumps(pe),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_generator.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add generator.py tests/test_generator.py
git commit -m "feat: create generator module with ref injection and PE construction"
```

---

### Task 4: Rewrite `orchestrator.py`

New `Orchestrator` with retry escalation, batch, and sweep. Uses modules 1-3.

**Files:**
- Rewrite: `orchestrator.py`
- Rewrite: `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing tests for single unit retry escalation**

```python
# tests/test_orchestrator.py
from unittest.mock import MagicMock, patch, call
import pytest
from orchestrator import Orchestrator, UnitResult, BatchResult
from eval_client import EvalResult
from generator import GenerateResult
from prompt_builder import PromptValidationError


SAMPLE_INPUT = {
    "text_output": "Wells",
    "anchor_characterization": "interactive streamer with signature blue hoodie",
    "brand_palette": {
        "primary": {"name": "Gold", "hex": "#D4AF37"},
        "secondary": {"name": "Vibrant Blue", "hex": "#007BFF"},
        "tertiary": {"name": "Black", "hex": "#000000"},
    },
    "photo_analysis": {
        "gender": "male",
        "hair": "dark black side-parted short hair",
        "eyes": "deep brown eyes",
        "expression": "gentle smile",
        "skin_tone": "medium olive",
    },
}

MOCK_KEYWORDS = {
    "character": {"gender": "male", "hair": "dark hair", "eyes": "brown",
                  "expression": "smile", "clothing": "blue hoodie"},
    "heart_carrier": {"color_name": "Gold", "color_hex": "#D4AF37",
                      "material": "soft candy"},
    "text": {"content": "Wells", "color_tint": "#007BFF"},
    "decorations": {"elements": ["music note"]},
    "background_color": "#000000",
}
MOCK_PROMPT = 'C4D Badge, 3D Pixar realistic cartoon style. heart "Wells" test'


def _eval(passed: bool, score: float) -> EvalResult:
    return EvalResult(
        passed=passed, total_score=score,
        dimensions={"heart_carrier": score, "character": score,
                     "decorations": score, "text_render": score,
                     "color_match": score, "composition": score,
                     "quality": score},
        issues=[] if passed else ["some issue"],
    )


def _gen_result(seed: int = 1) -> GenerateResult:
    return GenerateResult(
        image=b"fake-jpeg", seed=seed, request_id=f"req-{seed}",
    )


def _make_orchestrator(generator, eval_client, llm_client=None, **kwargs):
    return Orchestrator(
        llm_client=llm_client or MagicMock(),
        generator=generator,
        eval_client=eval_client,
        **kwargs,
    )


class TestSingleUnitPassFirst:
    @patch("orchestrator.pick_eval_references")
    @patch("orchestrator.build_prompt", return_value=(MOCK_KEYWORDS, MOCK_PROMPT))
    @patch("orchestrator.process")
    def test_pass_on_first_try(self, mock_process, mock_build, mock_pick_refs):
        mock_process.return_value = SAMPLE_INPUT
        mock_pick_refs.return_value = ([], [])

        gen = MagicMock()
        gen.generate.return_value = _gen_result()
        ev = MagicMock()
        ev.evaluate.return_value = _eval(True, 8.5)

        orch = _make_orchestrator(gen, ev)
        result = orch.run_batch(SAMPLE_INPUT, count=1)

        assert isinstance(result, BatchResult)
        assert len(result.results) == 1
        assert result.results[0].passed is True
        assert result.results[0].score == 8.5
        assert result.results[0].rounds == 1
        gen.generate.assert_called_once()


class TestSingleUnitRetryEscalation:
    @patch("orchestrator.pick_eval_references")
    @patch("orchestrator.build_prompt", return_value=(MOCK_KEYWORDS, MOCK_PROMPT))
    @patch("orchestrator.process")
    def test_level1_retry_new_seed(self, mock_process, mock_build, mock_pick_refs):
        """Fail first, pass on Level 1 retry (same prompt, new seed)."""
        mock_process.return_value = SAMPLE_INPUT
        mock_pick_refs.return_value = ([], [])

        gen = MagicMock()
        gen.generate.return_value = _gen_result()
        ev = MagicMock()
        ev.evaluate.side_effect = [_eval(False, 6.0), _eval(True, 8.5)]

        orch = _make_orchestrator(gen, ev, max_retries=2)
        result = orch.run_batch(SAMPLE_INPUT, count=1)

        assert result.results[0].passed is True
        assert result.results[0].rounds == 2
        assert gen.generate.call_count == 2

    @patch("orchestrator.expand_prompt", return_value="re-expanded prompt")
    @patch("orchestrator.pick_eval_references")
    @patch("orchestrator.build_prompt", return_value=(MOCK_KEYWORDS, MOCK_PROMPT))
    @patch("orchestrator.process")
    def test_level2_reexpand(self, mock_process, mock_build, mock_pick_refs, mock_expand):
        """Fail all Level 1 retries, pass on Level 2 re-expand."""
        mock_process.return_value = SAMPLE_INPUT
        mock_pick_refs.return_value = ([], [])

        gen = MagicMock()
        gen.generate.return_value = _gen_result()
        ev = MagicMock()
        # initial + 2 retries fail, re-expand passes
        ev.evaluate.side_effect = [
            _eval(False, 5.0), _eval(False, 5.5), _eval(False, 6.0),
            _eval(True, 8.5),
        ]

        orch = _make_orchestrator(gen, ev, max_retries=2, max_reexpands=1)
        result = orch.run_batch(SAMPLE_INPUT, count=1)

        assert result.results[0].passed is True
        assert result.results[0].rounds == 4
        mock_expand.assert_called_once()

    @patch("orchestrator.expand_prompt", return_value="re-expanded prompt")
    @patch("orchestrator.pick_eval_references")
    @patch("orchestrator.build_prompt", return_value=(MOCK_KEYWORDS, MOCK_PROMPT))
    @patch("orchestrator.process")
    def test_all_fail_returns_best(self, mock_process, mock_build, mock_pick_refs, mock_expand):
        """All retries and re-expands fail — return highest scoring result."""
        mock_process.return_value = SAMPLE_INPUT
        mock_pick_refs.return_value = ([], [])

        gen = MagicMock()
        gen.generate.return_value = _gen_result()
        ev = MagicMock()
        ev.evaluate.side_effect = [
            _eval(False, 5.0), _eval(False, 7.0), _eval(False, 6.0),
            _eval(False, 6.5),
        ]

        orch = _make_orchestrator(gen, ev, max_retries=2, max_reexpands=1)
        result = orch.run_batch(SAMPLE_INPUT, count=1)

        assert result.results[0].passed is False
        assert result.results[0].score == 7.0


class TestBatch:
    @patch("orchestrator.pick_eval_references")
    @patch("orchestrator.build_prompt", return_value=(MOCK_KEYWORDS, MOCK_PROMPT))
    @patch("orchestrator.process")
    def test_batch_runs_n_units(self, mock_process, mock_build, mock_pick_refs):
        mock_process.return_value = SAMPLE_INPUT
        mock_pick_refs.return_value = ([], [])

        gen = MagicMock()
        gen.generate.return_value = _gen_result()
        ev = MagicMock()
        ev.evaluate.return_value = _eval(True, 9.0)

        orch = _make_orchestrator(gen, ev)
        result = orch.run_batch(SAMPLE_INPUT, count=3)

        assert isinstance(result, BatchResult)
        assert result.total == 3
        assert len(result.results) == 3
        assert all(r.passed for r in result.results)

    @patch("orchestrator.pick_eval_references")
    @patch("orchestrator.build_prompt", return_value=(MOCK_KEYWORDS, MOCK_PROMPT))
    @patch("orchestrator.process")
    def test_batch_results_sorted_by_score(self, mock_process, mock_build, mock_pick_refs):
        mock_process.return_value = SAMPLE_INPUT
        mock_pick_refs.return_value = ([], [])

        gen = MagicMock()
        gen.generate.return_value = _gen_result()
        ev = MagicMock()
        ev.evaluate.side_effect = [_eval(True, 7.0), _eval(True, 9.0), _eval(True, 8.0)]

        orch = _make_orchestrator(gen, ev, max_retries=0, max_reexpands=0)
        result = orch.run_batch(SAMPLE_INPUT, count=3)

        scores = [r.score for r in result.results]
        assert scores == sorted(scores, reverse=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_orchestrator.py -v`
Expected: FAIL — `ImportError` for `UnitResult`, `BatchResult`, or `run_batch`

- [ ] **Step 3: Implement new `orchestrator.py`**

```python
# orchestrator.py
"""Orchestrator: unified pipeline with retry escalation, batch, and sweep."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from random import randint

from eval_client import EvalClient, EvalResult
from eval_store import pick_eval_references
from generator import Generator, GenerateResult
from input_processor import process
from prompt_builder import (
    analyze_photo, build_prompt, expand_prompt,
    build_negative_prompt, PromptValidationError,
)
from reference_store import pick_references

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output"


@dataclass
class UnitResult:
    """Result of one pipeline unit (possibly after retries)."""

    image: bytes
    score: float
    passed: bool
    seed: int
    prompt: str
    eval_result: EvalResult
    rounds: int
    request_id: str


@dataclass
class BatchResult:
    """Result of a batch run."""

    results: list[UnitResult] = field(default_factory=list)
    prompt: str = ""
    keywords: dict = field(default_factory=dict)
    total: int = 0
    success: int = 0
    failed: int = 0


class Orchestrator:
    """Unified pipeline: input → prompt → generate → eval → retry → batch."""

    def __init__(
        self,
        llm_client,
        generator: Generator,
        eval_client: EvalClient,
        max_retries: int = 2,
        max_reexpands: int = 1,
        pass_threshold: float = 8.0,
        max_workers: int = 10,
    ):
        self.llm_client = llm_client
        self.generator = generator
        self.eval_client = eval_client
        self.max_retries = max_retries
        self.max_reexpands = max_reexpands
        self.pass_threshold = pass_threshold
        self.max_workers = max_workers

    def run_batch(
        self,
        input_data: dict,
        photo_bytes: bytes | None = None,
        count: int = 1,
        use_refs: bool = True,
        ref_count: int = 2,
    ) -> BatchResult:
        """Main entry point for both single and batch generation.

        1. Normalize input
        2. Analyze photo (if provided)
        3. Build prompt (keywords + expand)
        4. Parallel N x _run_single_unit with retry escalation
        5. Sort results by score
        """
        input_data = process(input_data)

        if photo_bytes and not input_data.get("photo_analysis"):
            input_data["photo_analysis"] = analyze_photo(self.llm_client, photo_bytes)

        keywords, prompt = build_prompt(input_data, client=self.llm_client)
        negative_prompt = build_negative_prompt()
        logger.info("Prompt built (%d chars)", len(prompt))

        refs = pick_references(ref_count) if use_refs else []
        good_refs, bad_refs = pick_eval_references()

        def _run_unit(_idx: int) -> UnitResult:
            return self._run_single_unit(
                prompt=prompt,
                negative_prompt=negative_prompt,
                keywords=keywords,
                input_data=input_data,
                photo_bytes=photo_bytes,
                refs=refs,
                good_refs=good_refs,
                bad_refs=bad_refs,
            )

        results = []
        workers = min(count, self.max_workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_unit, i): i for i in range(count)}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error("Unit %d failed: %s", futures[future], e)

        results.sort(key=lambda r: r.score, reverse=True)
        success = sum(1 for r in results if r.passed)

        return BatchResult(
            results=results,
            prompt=prompt,
            keywords=keywords,
            total=count,
            success=success,
            failed=count - len(results),
        )

    def run_sweep(
        self,
        input_data: dict,
        photo_bytes: bytes | None = None,
        param_combos: list[dict] | None = None,
        use_refs: bool = True,
        ref_count: int = 2,
    ) -> BatchResult:
        """Like run_batch but each unit uses different generation params."""
        if not param_combos:
            return self.run_batch(input_data, photo_bytes, count=1,
                                  use_refs=use_refs, ref_count=ref_count)

        input_data = process(input_data)

        if photo_bytes and not input_data.get("photo_analysis"):
            input_data["photo_analysis"] = analyze_photo(self.llm_client, photo_bytes)

        keywords, prompt = build_prompt(input_data, client=self.llm_client)
        negative_prompt = build_negative_prompt()

        refs = pick_references(ref_count) if use_refs else []
        good_refs, bad_refs = pick_eval_references()

        def _run_unit(params: dict) -> UnitResult:
            return self._run_single_unit(
                prompt=prompt,
                negative_prompt=negative_prompt,
                keywords=keywords,
                input_data=input_data,
                photo_bytes=photo_bytes,
                refs=refs,
                good_refs=good_refs,
                bad_refs=bad_refs,
                extra_gen_kwargs=params,
            )

        results = []
        workers = min(len(param_combos), self.max_workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_unit, p): p for p in param_combos}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error("Sweep unit failed: %s", e)

        results.sort(key=lambda r: r.score, reverse=True)
        success = sum(1 for r in results if r.passed)

        return BatchResult(
            results=results,
            prompt=prompt,
            keywords=keywords,
            total=len(param_combos),
            success=success,
            failed=len(param_combos) - len(results),
        )

    def _run_single_unit(
        self,
        prompt: str,
        negative_prompt: str,
        keywords: dict,
        input_data: dict,
        photo_bytes: bytes | None,
        refs: list,
        good_refs: list,
        bad_refs: list,
        extra_gen_kwargs: dict | None = None,
    ) -> UnitResult:
        """One atomic unit with retry escalation.

        Round 0:   generate(prompt) → eval
        Level 1:   generate(prompt, new_seed) → eval  (up to max_retries)
        Level 2:   expand(keywords) → generate → eval (up to max_reexpands)
        Exhaust:   return best scoring result
        """
        gen_kwargs = {
            "guidance_scale": 8.0,
            "cfg_rescale_factor": 0.0,
            "single_edit_guidance_weight": 2.0,
            "single_edit_guidance_weight_image": 1.0,
        }
        if extra_gen_kwargs:
            gen_kwargs.update(extra_gen_kwargs)

        best: UnitResult | None = None
        current_prompt = prompt
        round_num = 0

        def _attempt(p: str) -> UnitResult:
            nonlocal round_num
            round_num += 1
            seed = randint(0, 2**31)
            gen_result = self.generator.generate(
                prompt=p,
                negative_prompt=negative_prompt,
                seed=seed,
                photo_bytes=photo_bytes,
                refs=refs if refs else None,
                **gen_kwargs,
            )
            self._save_image(gen_result.image, seed, round_num)
            eval_result = self.eval_client.evaluate(
                generated_image=gen_result.image,
                input_data=input_data,
                good_refs=good_refs,
                bad_refs=bad_refs,
            )
            logger.info("Round %d: seed=%d score=%.1f", round_num, seed, eval_result.total_score)
            return UnitResult(
                image=gen_result.image,
                score=eval_result.total_score,
                passed=eval_result.passed,
                seed=seed,
                prompt=p,
                eval_result=eval_result,
                rounds=round_num,
                request_id=gen_result.request_id,
            )

        def _track_best(unit: UnitResult) -> UnitResult:
            nonlocal best
            if best is None or unit.score > best.score:
                best = unit
            return unit

        # Initial attempt
        result = _track_best(_attempt(current_prompt))
        if result.passed:
            return result

        # Level 1: cheap retries (new seed, same prompt)
        for _ in range(self.max_retries):
            result = _track_best(_attempt(current_prompt))
            if result.passed:
                return result

        # Level 2: re-expand prompt from same keywords
        for _ in range(self.max_reexpands):
            current_prompt = expand_prompt(self.llm_client, keywords)
            logger.info("Level 2: re-expanded prompt (%d chars)", len(current_prompt))
            result = _track_best(_attempt(current_prompt))
            if result.passed:
                return result

        # All exhausted — return best
        logger.warning("All %d rounds exhausted, returning best (score=%.1f)",
                        round_num, best.score)
        best.rounds = round_num
        return best

    def _save_image(self, image: bytes, seed: int, round_num: int) -> None:
        """Save intermediate image for debugging."""
        if not image:
            return
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(OUTPUT_DIR, f"{ts}_r{round_num}_s{seed}.jpg")
        with open(path, "wb") as f:
            f.write(image)
        logger.info("Saved to %s", path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_orchestrator.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add orchestrator.py tests/test_orchestrator.py
git commit -m "feat: rewrite orchestrator with retry escalation, batch, and sweep"
```

---

### Task 5: Simplify `server.py`

Remove business logic, delegate to orchestrator.

**Files:**
- Modify: `server.py`
- Modify: `tests/test_server.py`

- [ ] **Step 1: Write failing test for new pipeline endpoint**

Add to `tests/test_server.py`:

```python
# Add to existing tests/test_server.py
from orchestrator import BatchResult, UnitResult
from eval_client import EvalResult


def _make_unit_result(score: float = 8.5) -> UnitResult:
    return UnitResult(
        image=b"\xff\xd8\xff\xe0fake-jpeg",
        score=score,
        passed=score >= 8.0,
        seed=12345,
        prompt="test prompt",
        eval_result=EvalResult(
            passed=score >= 8.0, total_score=score,
            dimensions={"heart_carrier": score}, issues=[],
        ),
        rounds=1,
        request_id="req-001",
    )


@pytest.mark.asyncio
async def test_pipeline_delegates_to_orchestrator():
    mock_batch_result = BatchResult(
        results=[_make_unit_result(9.0), _make_unit_result(7.5)],
        prompt="generated prompt",
        keywords={},
        total=2,
        success=1,
        failed=0,
    )

    with patch("server.orchestrator") as mock_orch:
        mock_orch.run_batch.return_value = mock_batch_result

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/pipeline",
                data={
                    "input_json": json.dumps({"slogan": "Test", "anchor_info": {
                        "anchor": {}, "anchor_characterization": "test",
                        "brand_palette": {"primary": {"name": "Red", "hex": "#F00"},
                                          "secondary": {"name": "Blue", "hex": "#00F"},
                                          "tertiary": {"name": "White", "hex": "#FFF"}},
                    }}),
                    "count": "2",
                },
                files=[("anchor_photo", ("test.jpg", b"\xff\xd8fake", "image/jpeg"))],
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert body["success"] == 1
        assert len(body["results"]) == 2
        assert body["results"][0]["eval"]["score"] == 9.0
        assert body["prompt"] == "generated prompt"
        mock_orch.run_batch.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_server.py::test_pipeline_delegates_to_orchestrator -v`
Expected: FAIL — `server.orchestrator` doesn't exist yet

- [ ] **Step 3: Rewrite `server.py`**

Replace the full content of `server.py` with:

```python
# server.py
"""FastAPI server — thin HTTP adapter for the badge generation pipeline."""

import base64
import itertools
import json
import logging
import os

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from eval_client import EvalClient
from generator import Generator
from orchestrator import Orchestrator, BatchResult
from seedream_sdk import SeedreamClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Seedream 4.5 Test UI")


# --- Initialize shared orchestrator ---
def _init_orchestrator() -> Orchestrator:
    import openai as _openai
    from dotenv import load_dotenv
    load_dotenv()
    llm_client = _openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return Orchestrator(
        llm_client=llm_client,
        generator=Generator(SeedreamClient()),
        eval_client=EvalClient(),
    )


orchestrator = _init_orchestrator()


def _format_batch_response(result: BatchResult) -> dict:
    """Convert BatchResult to JSON-serializable dict for frontend."""
    formatted_results = []
    for r in result.results:
        images_b64 = []
        if r.image:
            images_b64.append(
                f"data:image/jpeg;base64,{base64.b64encode(r.image).decode()}"
            )
        formatted_results.append({
            "seed": r.seed,
            "request_id": r.request_id,
            "images": images_b64,
            "eval": {
                "score": r.eval_result.total_score,
                "passed": r.eval_result.passed,
                "dimensions": r.eval_result.dimensions,
                "issues": r.eval_result.issues,
            },
        })

    return {
        "total": result.total,
        "success": result.success,
        "failed": result.failed,
        "prompt": result.prompt,
        "results": formatted_results,
        "errors": [],
    }


@app.post("/api/pipeline")
async def pipeline(
    input_json: str = Form(...),
    count: int = Form(10),
    use_refs: bool = Form(True),
    ref_count: int = Form(2),
    anchor_photo: UploadFile | None = File(None),
):
    """One-click pipeline: JSON+photo → prompt → batch generate → eval → ranked."""
    try:
        input_data = json.loads(input_json)
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

    photo_bytes = None
    if anchor_photo:
        photo_bytes = await anchor_photo.read()

    try:
        result = orchestrator.run_batch(
            input_data=input_data,
            photo_bytes=photo_bytes,
            count=count,
            use_refs=use_refs,
            ref_count=ref_count,
        )
    except Exception as e:
        logger.error("Pipeline error: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

    return _format_batch_response(result)


@app.post("/api/pipeline_sweep")
async def pipeline_sweep(
    input_json: str = Form(...),
    use_refs: bool = Form(True),
    ref_count: int = Form(2),
    guidance_scales: str = Form("8.0"),
    cfg_rescale_factors: str = Form("0.0"),
    edit_text_weights: str = Form("2.0"),
    edit_image_weights: str = Form("1.0"),
    anchor_photo: UploadFile | None = File(None),
):
    """Sweep pipeline: param combinations → each with full retry."""
    try:
        input_data = json.loads(input_json)
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

    photo_bytes = None
    if anchor_photo:
        photo_bytes = await anchor_photo.read()

    def _parse_floats(s: str) -> list[float]:
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    combos = []
    for gs, cfg, tw, iw in itertools.product(
        _parse_floats(guidance_scales),
        _parse_floats(cfg_rescale_factors),
        _parse_floats(edit_text_weights),
        _parse_floats(edit_image_weights),
    ):
        combos.append({
            "guidance_scale": gs,
            "cfg_rescale_factor": cfg,
            "single_edit_guidance_weight": tw,
            "single_edit_guidance_weight_image": iw,
        })

    try:
        result = orchestrator.run_sweep(
            input_data=input_data,
            photo_bytes=photo_bytes,
            param_combos=combos,
            use_refs=use_refs,
            ref_count=ref_count,
        )
    except Exception as e:
        logger.error("Sweep error: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

    return _format_batch_response(result)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
```

- [ ] **Step 4: Run all server tests**

Run: `pytest tests/test_server.py -v`
Expected: All tests PASS. The old `/api/generate` tests will fail because those endpoints are removed — delete `test_generate_t2i_success`, `test_generate_with_files`, `test_generate_missing_prompt` from the test file since those endpoints no longer exist.

- [ ] **Step 5: Clean up `tests/test_server.py`**

Remove the old generate endpoint tests. Keep only `test_pipeline_delegates_to_orchestrator` and any future pipeline tests. The final file:

```python
# tests/test_server.py
import json
from unittest.mock import patch, MagicMock
import pytest
from httpx import AsyncClient, ASGITransport
from server import app
from orchestrator import BatchResult, UnitResult
from eval_client import EvalResult


def _make_unit_result(score: float = 8.5) -> UnitResult:
    return UnitResult(
        image=b"\xff\xd8\xff\xe0fake-jpeg",
        score=score,
        passed=score >= 8.0,
        seed=12345,
        prompt="test prompt",
        eval_result=EvalResult(
            passed=score >= 8.0, total_score=score,
            dimensions={"heart_carrier": score}, issues=[],
        ),
        rounds=1,
        request_id="req-001",
    )


@pytest.mark.asyncio
async def test_pipeline_delegates_to_orchestrator():
    mock_batch_result = BatchResult(
        results=[_make_unit_result(9.0), _make_unit_result(7.5)],
        prompt="generated prompt",
        keywords={},
        total=2,
        success=1,
        failed=0,
    )

    with patch("server.orchestrator") as mock_orch:
        mock_orch.run_batch.return_value = mock_batch_result

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/pipeline",
                data={
                    "input_json": json.dumps({"slogan": "Test", "anchor_info": {
                        "anchor": {}, "anchor_characterization": "test",
                        "brand_palette": {"primary": {"name": "Red", "hex": "#F00"},
                                          "secondary": {"name": "Blue", "hex": "#00F"},
                                          "tertiary": {"name": "White", "hex": "#FFF"}},
                    }}),
                    "count": "2",
                },
                files=[("anchor_photo", ("test.jpg", b"\xff\xd8fake", "image/jpeg"))],
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert body["success"] == 1
        assert len(body["results"]) == 2
        assert body["results"][0]["eval"]["score"] == 9.0
        assert body["prompt"] == "generated prompt"
        mock_orch.run_batch.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_invalid_json():
    with patch("server.orchestrator"):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/pipeline",
                data={"input_json": "not-json", "count": "1"},
            )
        assert resp.status_code == 400
```

- [ ] **Step 6: Run all server tests**

Run: `pytest tests/test_server.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "refactor: simplify server.py to thin HTTP adapter over orchestrator"
```

---

### Task 6: Simplify `run_orchestrator.py`

Update CLI to use new orchestrator API.

**Files:**
- Modify: `run_orchestrator.py`

- [ ] **Step 1: Rewrite `run_orchestrator.py`**

```python
# run_orchestrator.py
"""CLI script to run the badge generation pipeline."""

import json
import logging
import os
import sys

import openai
from dotenv import load_dotenv

from eval_client import EvalClient
from generator import Generator
from orchestrator import Orchestrator
from seedream_sdk import SeedreamClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

SAMPLE_INPUT = {
    "slogan": "Wells",
    "community_type": "host",
    "slogan_lang": "English",
    "anchor_info": {
        "anchor": {
            "nick_name": "Wells",
            "bio_description": "Just chatting streamer",
        },
        "anchor_characterization": (
            "This Jordanian anchor hosts an energetic and highly interactive "
            "'Just Chatting' stream. He engages directly with his audience. "
            "Visually, he presents a casual style, frequently seen in a "
            "signature blue hoodie."
        ),
        "brand_palette": {
            "primary": {"name": "Gold", "hex": "#D4AF37"},
            "secondary": {"name": "Vibrant Blue", "hex": "#007BFF"},
            "tertiary": {"name": "Black", "hex": "#000000"},
        },
    },
}


def main():
    input_data = SAMPLE_INPUT
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            input_data = json.load(f)

    llm_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    orch = Orchestrator(
        llm_client=llm_client,
        generator=Generator(SeedreamClient()),
        eval_client=EvalClient(),
    )

    # Load photo if referenced in input
    photo_bytes = None
    photo_path = input_data.get("anchor_photo", "")
    if photo_path and os.path.isfile(photo_path):
        with open(photo_path, "rb") as f:
            photo_bytes = f.read()

    count = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    result = orch.run_batch(input_data, photo_bytes=photo_bytes, count=count)

    print(f"\n{'='*60}")
    print(f"Total: {result.total} | Success: {result.success} | Failed: {result.failed}")
    print(f"Prompt: {result.prompt[:150]}...")

    for i, r in enumerate(result.results):
        print(f"\n--- Result {i+1} ---")
        print(f"Score: {r.score:.1f} | Passed: {r.passed} | Rounds: {r.rounds} | Seed: {r.seed}")
        if r.eval_result.dimensions:
            print(f"Dimensions: {r.eval_result.dimensions}")
        if r.eval_result.issues:
            print(f"Issues: {r.eval_result.issues}")

        if r.image:
            out_path = f"output/cli_result_{i}_{r.seed}.jpg"
            os.makedirs("output", exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(r.image)
            print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify imports work**

Run: `python -c "from orchestrator import Orchestrator; from generator import Generator; from input_processor import process; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add run_orchestrator.py
git commit -m "refactor: simplify CLI to use new orchestrator API"
```

---

### Task 7: Run full test suite and fix any issues

**Files:**
- All test files

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass. If any fail, diagnose and fix.

- [ ] **Step 2: Verify test count**

The expected test breakdown:
- `test_input_processor.py`: 5 tests
- `test_prompt_builder.py`: ~10 tests (8 old + 2 new)
- `test_generator.py`: 6 tests
- `test_orchestrator.py`: 6 tests
- `test_server.py`: 2 tests
- `test_eval_client.py`: unchanged
- `test_eval_store.py`: unchanged
- `test_seedream_sdk.py`: unchanged

- [ ] **Step 3: Commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: resolve test issues from pipeline unification"
```

- [ ] **Step 4: Verify server starts**

Run: `python -c "from server import app; print('Server imports OK')"`
Expected: `Server imports OK`

---

## Execution Dependency Graph

```
Task 1 (input_processor) ──┐
                            ├── Task 4 (orchestrator) ── Task 5 (server) ── Task 7 (verify)
Task 2 (prompt_builder) ───┤                                                     │
                            │                                                     │
Task 3 (generator) ────────┘                           Task 6 (CLI) ─────────────┘
```

Tasks 1, 2, 3 are independent and can run in parallel.
Task 4 depends on all three.
Tasks 5 and 6 depend on Task 4.
Task 7 depends on all.
