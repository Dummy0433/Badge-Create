# Eval & Retry Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a quality evaluation loop (GPT-5.4 vision) after Seedream image generation with structured scoring, targeted prompt adjustment, and retry/reroll logic.

**Architecture:** Four new modules — `prompt_builder.py` (template assembly + pre-check), `eval_store.py` (good/bad reference images), `eval_client.py` (GPT-5.4 structured eval), `orchestrator.py` (generate → eval → retry/reroll loop). Each is independently testable. The orchestrator wires them together with the existing `SeedreamClient`.

**Tech Stack:** Python 3.14, OpenAI SDK (`openai`), `python-dotenv`, pytest, existing `seedream_sdk.py`

---

### Task 1: Dependencies & Configuration

**Files:**
- Modify: `requirements.txt`
- Create: `.env.example`

- [ ] **Step 1: Add new dependencies to requirements.txt**

```
requests>=2.31.0
fastapi>=0.110.0
uvicorn>=0.29.0
python-multipart>=0.0.9
pytest>=8.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
openai>=1.0.0
python-dotenv>=1.0.0
```

- [ ] **Step 2: Create .env.example**

```
OPENAI_API_KEY=sk-your-key-here
```

- [ ] **Step 3: Add .env and eval_references/ to .gitignore**

Append to `.gitignore`:
```
.env
eval_references/
```

- [ ] **Step 4: Install dependencies**

Run: `pip install -r requirements.txt`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .env.example .gitignore
git commit -m "chore: add openai and python-dotenv dependencies"
```

---

### Task 2: prompt_builder.py — Template Assembly & Pre-check

**Files:**
- Create: `prompt_builder.py`
- Create: `tests/test_prompt_builder.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_prompt_builder.py
import pytest
from prompt_builder import build_prompt, build_negative_prompt, PromptValidationError


SAMPLE_INPUT = {
    "text_output": "Wells",
    "anchor_characterization": (
        "This Jordanian anchor hosts an energetic and highly interactive "
        "'Just Chatting' stream. He engages directly with his audience, "
        "responding to song requests, comments, and playful banter. "
        "Visually, he presents a casual style, frequently seen in a "
        "signature blue hoodie."
    ),
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


class TestBuildPrompt:
    def test_contains_fixed_style(self):
        prompt = build_prompt(SAMPLE_INPUT)
        assert "C4D Badge" in prompt
        assert "3D Pixar realistic cartoon style" in prompt
        assert "candy color palette" in prompt

    def test_contains_text_output(self):
        prompt = build_prompt(SAMPLE_INPUT)
        assert '"Wells"' in prompt

    def test_contains_brand_colors(self):
        prompt = build_prompt(SAMPLE_INPUT)
        # Primary color used for heart
        assert "Gold" in prompt or "#D4AF37" in prompt
        # Tertiary used for background
        assert "#000000" in prompt or "Black" in prompt

    def test_contains_character_traits(self):
        prompt = build_prompt(SAMPLE_INPUT)
        assert "dark black side-parted short hair" in prompt
        assert "deep brown eyes" in prompt
        assert "gentle smile" in prompt

    def test_contains_lighting(self):
        prompt = build_prompt(SAMPLE_INPUT)
        assert "Warm side light from left" in prompt

    def test_missing_text_output_raises(self):
        bad_input = {**SAMPLE_INPUT, "text_output": ""}
        with pytest.raises(PromptValidationError, match="text_output"):
            build_prompt(bad_input)

    def test_missing_photo_analysis_raises(self):
        bad_input = {**SAMPLE_INPUT}
        del bad_input["photo_analysis"]
        with pytest.raises(PromptValidationError, match="photo_analysis"):
            build_prompt(bad_input)

    def test_missing_brand_palette_raises(self):
        bad_input = {**SAMPLE_INPUT}
        del bad_input["brand_palette"]
        with pytest.raises(PromptValidationError, match="brand_palette"):
            build_prompt(bad_input)


class TestBuildNegativePrompt:
    def test_contains_key_terms(self):
        neg = build_negative_prompt()
        assert "nsfw" in neg
        assert "balloon" in neg
        assert "inflatable" in neg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prompt_builder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'prompt_builder'`

- [ ] **Step 3: Implement prompt_builder.py**

```python
# prompt_builder.py
"""Template-based prompt assembly for badge generation."""


class PromptValidationError(Exception):
    """Raised when assembled prompt fails pre-checks."""
    pass


_TEMPLATE = (
    'C4D Badge, 3D Pixar realistic cartoon style. '
    'A large plump rounded solid 3D heart shape as the background carrier, '
    'the heart is thick and voluminous like soft candy or jelly, '
    'smooth glossy surface with gradient {heart_color} color, '
    'NOT a balloon NOT inflatable, no wrinkles no seams no strings, '
    'just a clean solid puffy 3D heart object. '
    'A {gender} character with {hair}, {eyes}, {expression}, '
    'wearing {clothing}, '
    'the character is positioned in front of the upper area of the heart '
    'from chest up, occupying 70% of the heart. '
    '{decorations} floating around the heart in the air as decorative elements. '
    '3D bold thick retro cursive gradient text "{text_output}" at the bottom '
    'of the heart, partially extending beyond the heart edge, '
    'the text has silver chrome sweep light effect, '
    'holographic iridescent metallic material. '
    'Warm side light from left, cool side light from right, '
    'soft front key light. '
    'Pure {bg_color} background, candy color palette, '
    'commercial art illustration style.'
)


def _extract_clothing(characterization: str) -> str:
    """Extract clothing description from anchor characterization."""
    lower = characterization.lower()
    if "hoodie" in lower:
        # Find the phrase around "hoodie"
        for sentence in characterization.split("."):
            if "hoodie" in sentence.lower():
                # Extract color + hoodie mention
                words = sentence.strip().split()
                for i, w in enumerate(words):
                    if "hoodie" in w.lower():
                        start = max(0, i - 2)
                        return " ".join(words[start:i + 1]).strip("(), ")
    if "jacket" in lower:
        return "a casual jacket"
    return "a casual top"


def _extract_decorations(characterization: str) -> str:
    """Derive decoration elements from anchor characterization themes."""
    lower = characterization.lower()
    elements = []
    if any(w in lower for w in ["song", "music", "sing"]):
        elements.append("music notes")
    if any(w in lower for w in ["chat", "interactive", "audience", "comment"]):
        elements.append("chat bubbles")
    if any(w in lower for w in ["stream", "broadcast", "host"]):
        elements.append("small star icons")
    elements.append("hand-drawn graffiti-style outline hearts and love doodles")
    return "Small " + ", ".join(elements)


def build_prompt(input_data: dict) -> str:
    """Assemble full Seedream prompt from datamining input.

    Raises PromptValidationError if required fields are missing.
    """
    # Validate required fields
    text_output = input_data.get("text_output", "").strip()
    if not text_output:
        raise PromptValidationError("text_output is empty or missing")

    photo = input_data.get("photo_analysis")
    if not photo:
        raise PromptValidationError("photo_analysis is missing")

    palette = input_data.get("brand_palette")
    if not palette:
        raise PromptValidationError("brand_palette is missing")

    characterization = input_data.get("anchor_characterization", "")

    # Extract variable slots
    heart_color = f"{palette['primary']['name']} (#{palette['primary']['hex'].lstrip('#')})"
    bg_color = palette["tertiary"]["hex"]
    clothing = _extract_clothing(characterization)
    decorations = _extract_decorations(characterization)

    prompt = _TEMPLATE.format(
        heart_color=heart_color,
        gender=photo.get("gender", ""),
        hair=photo.get("hair", ""),
        eyes=photo.get("eyes", ""),
        expression=photo.get("expression", ""),
        clothing=clothing,
        decorations=decorations,
        text_output=text_output,
        bg_color=bg_color,
    )

    # Post-check: text_output must appear in final prompt
    if text_output not in prompt:
        raise PromptValidationError(
            f"text_output '{text_output}' not found in assembled prompt"
        )

    return prompt


def build_negative_prompt() -> str:
    """Return the fixed negative prompt for badge generation."""
    return (
        "nsfw, balloon, inflatable, wrinkles, seams, strings, rope, "
        "flat, 2D, thin, metal frame, hard border, badge pin, deflated"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prompt_builder.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prompt_builder.py tests/test_prompt_builder.py
git commit -m "feat: add prompt_builder with template assembly and pre-check"
```

---

### Task 3: eval_store.py — Eval Reference Library

**Files:**
- Create: `eval_store.py`
- Create: `tests/test_eval_store.py`
- Create: `eval_references/` directory (images provided by user)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_eval_store.py
from eval_store import EvalReference, pick_eval_references, get_all_eval_references


class TestEvalStore:
    def test_eval_reference_fields(self):
        refs = get_all_eval_references()
        assert len(refs) >= 2
        for ref in refs:
            assert isinstance(ref, EvalReference)
            assert ref.image_path
            assert ref.description
            assert isinstance(ref.is_good, bool)
            assert 0 <= ref.score <= 10

    def test_good_refs_have_high_score(self):
        refs = get_all_eval_references()
        good = [r for r in refs if r.is_good]
        assert len(good) >= 1
        for r in good:
            assert r.score >= 8.0
            assert len(r.issues) == 0

    def test_bad_refs_have_low_score(self):
        refs = get_all_eval_references()
        bad = [r for r in refs if not r.is_good]
        assert len(bad) >= 1
        for r in bad:
            assert r.score < 6.0
            assert len(r.issues) > 0

    def test_pick_eval_references(self):
        good, bad = pick_eval_references(good_count=1, bad_count=1)
        assert len(good) == 1
        assert len(bad) == 1
        assert good[0].is_good is True
        assert bad[0].is_good is False

    def test_load_bytes(self):
        refs = get_all_eval_references()
        for ref in refs:
            data = ref.load_bytes()
            assert isinstance(data, bytes)
            assert len(data) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval_store'`

- [ ] **Step 3: Create eval_references directory and placeholder images**

The user will provide a good and bad example image. For now, create the directory and copy placeholder images from existing references:

```bash
mkdir -p eval_references
# User will provide actual images — use existing refs as temporary placeholders
cp references/ref_01_gelik.png eval_references/good_01.png
cp references/ref_01_gelik.png eval_references/bad_01.png
```

NOTE: Ask user to provide actual good and bad example images to replace these placeholders.

- [ ] **Step 4: Implement eval_store.py**

```python
# eval_store.py
"""Eval reference library for image quality assessment.

Stores good and bad example images with descriptions, scores, and issues.
Currently hardcoded. Will be replaced with database-backed dynamic dataset.
"""

import os
import random
from dataclasses import dataclass, field

EVAL_REFERENCES_DIR = os.path.join(os.path.dirname(__file__), "eval_references")


@dataclass
class EvalReference:
    """A reference image for eval with quality metadata."""

    image_path: str
    description: str
    is_good: bool
    score: float
    issues: list[str] = field(default_factory=list)

    def load_bytes(self) -> bytes:
        with open(self.image_path, "rb") as f:
            return f.read()


_GOOD_REFERENCES = [
    EvalReference(
        image_path=os.path.join(EVAL_REFERENCES_DIR, "good_01.png"),
        description=(
            "Excellent badge: 3D Pixar style, plump glossy heart carrier with "
            "correct gradient color, character positioned chest-up at 70%, "
            "clean metallic text at bottom, appropriate floating decorations, "
            "pure black background, candy color palette throughout."
        ),
        is_good=True,
        score=9.0,
        issues=[],
    ),
]

_BAD_REFERENCES = [
    EvalReference(
        image_path=os.path.join(EVAL_REFERENCES_DIR, "bad_01.png"),
        description=(
            "Poor badge: heart shape looks like a deflated balloon with wrinkles, "
            "character is too small and off-center, text is blurry and unreadable, "
            "background is not pure black, colors do not match brand palette."
        ),
        is_good=False,
        score=3.0,
        issues=[
            "Heart shape has wrinkles and seams like a balloon",
            "Character occupies less than 40% of heart",
            "Text is blurry and partially cut off",
            "Background has gradient instead of pure black",
            "Colors do not match brand palette",
        ],
    ),
]


def get_all_eval_references() -> list[EvalReference]:
    return list(_GOOD_REFERENCES) + list(_BAD_REFERENCES)


def pick_eval_references(
    good_count: int = 1, bad_count: int = 1
) -> tuple[list[EvalReference], list[EvalReference]]:
    """Return (good_refs, bad_refs) randomly sampled."""
    good_count = min(good_count, len(_GOOD_REFERENCES))
    bad_count = min(bad_count, len(_BAD_REFERENCES))
    return (
        random.sample(_GOOD_REFERENCES, good_count),
        random.sample(_BAD_REFERENCES, bad_count),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_eval_store.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add eval_store.py tests/test_eval_store.py
git commit -m "feat: add eval_store with good/bad reference images"
```

---

### Task 4: eval_client.py — GPT-5.4 Vision Eval

**Files:**
- Create: `eval_client.py`
- Create: `tests/test_eval_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_eval_client.py
import json
from unittest.mock import patch, MagicMock
import pytest
from eval_client import EvalClient, EvalResult
from eval_store import EvalReference


def _make_ref(is_good: bool) -> EvalReference:
    return EvalReference(
        image_path="dummy.png",
        description="test ref",
        is_good=is_good,
        score=9.0 if is_good else 3.0,
        issues=[] if is_good else ["bad quality"],
    )


MOCK_EVAL_RESPONSE = {
    "passed": True,
    "total_score": 8.5,
    "dimensions": {
        "heart_carrier": 9,
        "character": 8,
        "decorations": 8,
        "text_render": 9,
        "color_match": 8,
        "composition": 9,
        "quality": 8.5,
    },
    "issues": [],
    "suggestion": "",
}

MOCK_FAIL_RESPONSE = {
    "passed": False,
    "total_score": 5.5,
    "dimensions": {
        "heart_carrier": 7,
        "character": 6,
        "decorations": 5,
        "text_render": 3,
        "color_match": 6,
        "composition": 7,
        "quality": 4.5,
    },
    "issues": ["Text 'Wells' rendered as 'Wels'", "Artifacts on heart edge"],
    "suggestion": "Emphasize exact text spelling in quotes, add 'clean edges' to prompt",
}

INPUT_DATA = {
    "text_output": "Wells",
    "brand_palette": {
        "primary": {"name": "Gold", "hex": "#D4AF37"},
        "secondary": {"name": "Vibrant Blue", "hex": "#007BFF"},
        "tertiary": {"name": "Black", "hex": "#000000"},
    },
    "photo_analysis": {"gender": "male", "hair": "dark hair"},
    "anchor_characterization": "interactive streamer",
}


class TestEvalClient:
    @patch("eval_client.openai.OpenAI")
    def test_evaluate_pass(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(MOCK_EVAL_RESPONSE)))]
        )

        eval_client = EvalClient()
        result = eval_client.evaluate(
            generated_image=b"fake-image",
            input_data=INPUT_DATA,
            good_refs=[_make_ref(True)],
            bad_refs=[_make_ref(False)],
        )

        assert isinstance(result, EvalResult)
        assert result.passed is True
        assert result.total_score == 8.5
        assert result.dimensions["text_render"] == 9
        assert len(result.issues) == 0

    @patch("eval_client.openai.OpenAI")
    def test_evaluate_fail(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(MOCK_FAIL_RESPONSE)))]
        )

        eval_client = EvalClient()
        result = eval_client.evaluate(
            generated_image=b"fake-image",
            input_data=INPUT_DATA,
            good_refs=[_make_ref(True)],
            bad_refs=[_make_ref(False)],
        )

        assert result.passed is False
        assert result.total_score == 5.5
        assert "Wels" in result.issues[0]
        assert result.suggestion != ""

    @patch("eval_client.openai.OpenAI")
    def test_system_prompt_contains_dimensions(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(MOCK_EVAL_RESPONSE)))]
        )

        eval_client = EvalClient()
        eval_client.evaluate(
            generated_image=b"fake-image",
            input_data=INPUT_DATA,
            good_refs=[_make_ref(True)],
            bad_refs=[_make_ref(False)],
        )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", [])
        system_msg = messages[0]["content"]
        assert "heart_carrier" in system_msg
        assert "text_render" in system_msg
        assert "color_match" in system_msg

    @patch("eval_client.openai.OpenAI")
    def test_input_data_in_user_message(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(MOCK_EVAL_RESPONSE)))]
        )

        eval_client = EvalClient()
        eval_client.evaluate(
            generated_image=b"fake-image",
            input_data=INPUT_DATA,
            good_refs=[_make_ref(True)],
            bad_refs=[_make_ref(False)],
        )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", [])
        # Find user message with text content containing input_data
        user_texts = []
        for msg in messages:
            if msg["role"] == "user":
                for part in msg["content"]:
                    if isinstance(part, dict) and part.get("type") == "text":
                        user_texts.append(part["text"])
        full_text = " ".join(user_texts)
        assert "Wells" in full_text
        assert "Gold" in full_text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval_client'`

- [ ] **Step 3: Implement eval_client.py**

```python
# eval_client.py
"""GPT-5.4 vision-based structured image evaluation for badge generation."""

import base64
import json
import logging
import os
from dataclasses import dataclass, field

import openai
from dotenv import load_dotenv

from eval_store import EvalReference

load_dotenv()

logger = logging.getLogger(__name__)

EVAL_SYSTEM_PROMPT = """\
You are a quality evaluator for AI-generated badge images. You evaluate a generated \
image against specific criteria and reference examples.

Score each dimension from 1-10:
- heart_carrier: Is there a large plump 3D heart shape? Correct material (candy/jelly, \
glossy surface)? Correct color from brand palette?
- character: Does the person match the described appearance (hair, eyes, expression)? \
Positioned chest-up, occupying ~70% of the heart?
- decorations: Are there appropriate floating decorative elements matching the \
anchor's personality/themes?
- text_render: Does the text content EXACTLY match the required text_output value? \
Is it metallic material? Positioned at the bottom?
- color_match: Do the heart color, text accents, and decorations match the \
brand_palette colors provided?
- composition: Overall layout correct? Background is the specified color? \
Character properly framed?
- quality: No artifacts, deformed limbs, blurry areas, or visual glitches?

IMPORTANT for text_render: The text in the image MUST exactly match the \
"text_output" value from the criteria. Any misspelling or missing characters \
is a critical failure (score ≤ 3).

Respond with ONLY valid JSON in this exact format:
{
  "passed": <bool, true if average score >= 8.0>,
  "total_score": <float, average of all dimension scores>,
  "dimensions": {
    "heart_carrier": <float>,
    "character": <float>,
    "decorations": <float>,
    "text_render": <float>,
    "color_match": <float>,
    "composition": <float>,
    "quality": <float>
  },
  "issues": [<list of specific problems found, empty if none>],
  "suggestion": "<targeted prompt adjustment suggestion to fix issues, empty if passed>"
}"""


@dataclass
class EvalResult:
    """Structured evaluation result."""

    passed: bool
    total_score: float
    dimensions: dict[str, float] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    suggestion: str = ""


class EvalClient:
    """Client for evaluating generated badge images using GPT-5.4 vision."""

    def __init__(self, model: str = "gpt-5.4"):
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def evaluate(
        self,
        generated_image: bytes,
        input_data: dict,
        good_refs: list[EvalReference],
        bad_refs: list[EvalReference],
    ) -> EvalResult:
        """Evaluate a generated image against criteria and references."""
        user_content = self._build_user_message(
            generated_image, input_data, good_refs, bad_refs
        )

        logger.info("Calling GPT-5.4 eval: text_output=%s", input_data.get("text_output"))

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            max_tokens=1000,
        )

        return self._parse_response(response)

    def _build_user_message(
        self,
        generated_image: bytes,
        input_data: dict,
        good_refs: list[EvalReference],
        bad_refs: list[EvalReference],
    ) -> list[dict]:
        """Build multimodal user message with images and criteria."""
        content = []

        # Criteria from input data
        criteria = {
            "text_output": input_data.get("text_output", ""),
            "brand_palette": input_data.get("brand_palette", {}),
            "photo_analysis": input_data.get("photo_analysis", {}),
            "anchor_characterization": input_data.get("anchor_characterization", ""),
        }
        content.append({
            "type": "text",
            "text": f"## Evaluation Criteria\n{json.dumps(criteria, indent=2)}",
        })

        # Good reference(s)
        for ref in good_refs:
            content.append({
                "type": "text",
                "text": f"## Good Example (score: {ref.score})\n{ref.description}",
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(ref.load_bytes()).decode()}"
                },
            })

        # Bad reference(s)
        for ref in bad_refs:
            content.append({
                "type": "text",
                "text": (
                    f"## Bad Example (score: {ref.score})\n{ref.description}\n"
                    f"Issues: {', '.join(ref.issues)}"
                ),
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(ref.load_bytes()).decode()}"
                },
            })

        # Generated image to evaluate
        content.append({
            "type": "text",
            "text": "## Image to Evaluate\nScore this image against the criteria above:",
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64.b64encode(generated_image).decode()}"
            },
        })

        return content

    def _parse_response(self, response) -> EvalResult:
        """Parse GPT response into EvalResult."""
        text = response.choices[0].message.content
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.error("Failed to parse eval response: %s", text)
            return EvalResult(passed=False, total_score=0.0, suggestion="Eval parse error")

        return EvalResult(
            passed=data.get("passed", False),
            total_score=data.get("total_score", 0.0),
            dimensions=data.get("dimensions", {}),
            issues=data.get("issues", []),
            suggestion=data.get("suggestion", ""),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_client.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add eval_client.py tests/test_eval_client.py
git commit -m "feat: add eval_client with GPT-5.4 vision structured evaluation"
```

---

### Task 5: orchestrator.py — Generate → Eval → Retry/Reroll Loop

**Files:**
- Create: `orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_orchestrator.py
from unittest.mock import MagicMock, patch
import pytest
from orchestrator import Orchestrator, OrchestrationResult
from eval_client import EvalResult
from prompt_builder import PromptValidationError
from seedream_sdk import SeedreamResponse


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


def _make_eval_result(passed: bool, score: float, suggestion: str = "") -> EvalResult:
    return EvalResult(
        passed=passed,
        total_score=score,
        dimensions={"heart_carrier": score, "character": score, "decorations": score,
                     "text_render": score, "color_match": score, "composition": score,
                     "quality": score},
        issues=[] if passed else ["some issue"],
        suggestion=suggestion,
    )


def _make_seedream_response() -> SeedreamResponse:
    return SeedreamResponse(
        images=[b"fake-jpeg-data"],
        llm_result="",
        request_id="test-123",
    )


class TestOrchestratorPassOnFirstTry:
    def test_returns_on_first_pass(self):
        seedream = MagicMock()
        seedream.generate.return_value = _make_seedream_response()

        eval_client = MagicMock()
        eval_client.evaluate.return_value = _make_eval_result(True, 8.5)

        orch = Orchestrator(seedream_client=seedream, eval_client=eval_client)
        result = orch.run(SAMPLE_INPUT)

        assert isinstance(result, OrchestrationResult)
        assert result.passed is True
        assert result.rounds == 1
        assert len(result.eval_history) == 1
        seedream.generate.assert_called_once()


class TestOrchestratorRetry:
    @patch("orchestrator.openai.OpenAI")
    def test_retries_on_failure_then_passes(self, mock_openai_cls):
        mock_adj_client = MagicMock()
        mock_openai_cls.return_value = mock_adj_client
        mock_adj_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="adjusted prompt text"))]
        )

        seedream = MagicMock()
        seedream.generate.return_value = _make_seedream_response()

        eval_client = MagicMock()
        eval_client.evaluate.side_effect = [
            _make_eval_result(False, 6.0, "fix text rendering"),
            _make_eval_result(True, 8.5),
        ]

        orch = Orchestrator(seedream_client=seedream, eval_client=eval_client)
        result = orch.run(SAMPLE_INPUT)

        assert result.passed is True
        assert result.rounds == 2
        assert len(result.prompt_history) == 2

    @patch("orchestrator.openai.OpenAI")
    def test_rerolls_after_max_retries(self, mock_openai_cls):
        mock_adj_client = MagicMock()
        mock_openai_cls.return_value = mock_adj_client
        mock_adj_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="rerolled prompt text"))]
        )

        seedream = MagicMock()
        seedream.generate.return_value = _make_seedream_response()

        eval_client = MagicMock()
        eval_client.evaluate.side_effect = [
            _make_eval_result(False, 5.0, "fix A"),
            _make_eval_result(False, 5.5, "fix B"),
            _make_eval_result(False, 6.0, "fix C"),
            _make_eval_result(True, 8.5),
        ]

        orch = Orchestrator(
            seedream_client=seedream, eval_client=eval_client,
            max_retries=2, max_rerolls=1,
        )
        result = orch.run(SAMPLE_INPUT)

        assert result.passed is True
        assert result.rounds == 4


class TestOrchestratorFallback:
    @patch("orchestrator.openai.OpenAI")
    def test_returns_best_on_all_failures(self, mock_openai_cls):
        mock_adj_client = MagicMock()
        mock_openai_cls.return_value = mock_adj_client
        mock_adj_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="adjusted prompt"))]
        )

        seedream = MagicMock()
        seedream.generate.return_value = _make_seedream_response()

        eval_client = MagicMock()
        eval_client.evaluate.side_effect = [
            _make_eval_result(False, 5.0, "fix A"),
            _make_eval_result(False, 7.0, "fix B"),
            _make_eval_result(False, 6.0, "fix C"),
            _make_eval_result(False, 6.5, "fix D"),
        ]

        orch = Orchestrator(
            seedream_client=seedream, eval_client=eval_client,
            max_retries=2, max_rerolls=1,
        )
        result = orch.run(SAMPLE_INPUT)

        assert result.passed is False
        assert result.score == 7.0  # best score across all rounds


class TestOrchestratorValidation:
    def test_invalid_input_raises(self):
        orch = Orchestrator(
            seedream_client=MagicMock(), eval_client=MagicMock()
        )
        with pytest.raises(PromptValidationError):
            orch.run({"text_output": ""})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orchestrator'`

- [ ] **Step 3: Implement orchestrator.py**

```python
# orchestrator.py
"""Orchestrator: generate → eval → retry/reroll pipeline."""

import logging
import os
from dataclasses import dataclass, field
from random import randint

import openai
from dotenv import load_dotenv

from eval_client import EvalClient, EvalResult
from eval_store import pick_eval_references
from prompt_builder import build_prompt, build_negative_prompt
from seedream_sdk import SeedreamClient, SeedreamAPIError

load_dotenv()

logger = logging.getLogger(__name__)

ADJUST_SYSTEM_PROMPT = """\
You are a prompt engineer. You receive an original image generation prompt and \
evaluation feedback about what went wrong. Your job is to adjust the prompt to \
fix the specific issues mentioned.

Rules:
- Only modify parts of the prompt related to the failing dimensions
- Do NOT change: render style (C4D Badge, 3D Pixar), lighting, composition rules, \
candy color palette, or commercial art illustration style
- Keep the overall structure intact
- Return ONLY the adjusted prompt text, nothing else"""

REROLL_SYSTEM_PROMPT = """\
You are a prompt engineer. You receive structured data about a badge to generate \
and must write a fresh, creative prompt for a Seedream image generation model.

The badge style is FIXED:
- C4D Badge, 3D Pixar realistic cartoon style
- Large plump 3D heart shape carrier (candy/jelly material, glossy)
- Character positioned chest-up, occupying 70% of heart
- Metallic text at bottom with chrome sweep light effect
- Floating decorative elements around the heart
- Candy color palette, commercial art illustration style
- Specific lighting: warm left, cool right, soft front key light

Fill in the variable parts from the data provided. Write a DIFFERENT creative \
expansion than a previous attempt — vary the wording and emphasis while keeping \
all required elements. Return ONLY the prompt text."""


@dataclass
class OrchestrationResult:
    """Result of the full orchestration pipeline."""

    image: bytes
    score: float
    passed: bool
    rounds: int
    prompt_history: list[str] = field(default_factory=list)
    eval_history: list[EvalResult] = field(default_factory=list)


class Orchestrator:
    """Generate → eval → retry/reroll pipeline."""

    def __init__(
        self,
        seedream_client: SeedreamClient,
        eval_client: EvalClient,
        max_retries: int = 2,
        max_rerolls: int = 1,
        pass_threshold: float = 8.0,
    ):
        self.seedream = seedream_client
        self.eval = eval_client
        self.max_retries = max_retries
        self.max_rerolls = max_rerolls
        self.pass_threshold = pass_threshold
        self.adj_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def run(self, input_data: dict) -> OrchestrationResult:
        """Full pipeline: build prompt → generate → eval → retry/reroll."""
        original_prompt = build_prompt(input_data)
        negative_prompt = build_negative_prompt()

        all_images: list[bytes] = []
        all_scores: list[float] = []
        all_prompts: list[str] = []
        all_evals: list[EvalResult] = []

        # Phase 1: original + retries
        current_prompt = original_prompt
        for attempt in range(1 + self.max_retries):
            image, eval_result = self._generate_and_eval(
                current_prompt, negative_prompt, input_data, attempt
            )
            all_images.append(image)
            all_scores.append(eval_result.total_score)
            all_prompts.append(current_prompt)
            all_evals.append(eval_result)

            if eval_result.passed:
                logger.info("Round %d: PASSED (score=%.1f)", attempt, eval_result.total_score)
                return OrchestrationResult(
                    image=image, score=eval_result.total_score, passed=True,
                    rounds=len(all_evals), prompt_history=all_prompts,
                    eval_history=all_evals,
                )

            logger.info(
                "Round %d: FAILED (score=%.1f) issues=%s",
                attempt, eval_result.total_score, eval_result.issues,
            )

            # Adjust prompt for next retry (based on ORIGINAL prompt)
            if attempt < self.max_retries:
                current_prompt = self._adjust_prompt(
                    original_prompt, eval_result
                )

        # Phase 2: reroll
        for reroll in range(self.max_rerolls):
            rerolled_prompt = self._reroll_prompt(input_data)
            image, eval_result = self._generate_and_eval(
                rerolled_prompt, negative_prompt, input_data,
                len(all_evals),
            )
            all_images.append(image)
            all_scores.append(eval_result.total_score)
            all_prompts.append(rerolled_prompt)
            all_evals.append(eval_result)

            if eval_result.passed:
                logger.info(
                    "Reroll %d: PASSED (score=%.1f)", reroll, eval_result.total_score
                )
                return OrchestrationResult(
                    image=image, score=eval_result.total_score, passed=True,
                    rounds=len(all_evals), prompt_history=all_prompts,
                    eval_history=all_evals,
                )

            logger.info(
                "Reroll %d: FAILED (score=%.1f)", reroll, eval_result.total_score
            )

        # All attempts failed — return best
        best_idx = all_scores.index(max(all_scores))
        logger.warning(
            "All %d rounds failed. Returning best (score=%.1f)",
            len(all_evals), all_scores[best_idx],
        )
        return OrchestrationResult(
            image=all_images[best_idx], score=all_scores[best_idx], passed=False,
            rounds=len(all_evals), prompt_history=all_prompts,
            eval_history=all_evals,
        )

    def _generate_and_eval(
        self, prompt: str, negative_prompt: str, input_data: dict, round_num: int
    ) -> tuple[bytes, EvalResult]:
        """Generate image with Seedream and evaluate it."""
        seed = randint(0, 2**31)
        logger.info("Round %d: generating (seed=%d)", round_num, seed)

        try:
            result = self.seedream.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                guidance_scale=8.0,
                cfg_rescale_factor=0.0,
                single_edit_guidance_weight=2.0,
                single_edit_guidance_weight_image=1.0,
            )
            image = result.images[0] if result.images else b""
        except SeedreamAPIError as e:
            logger.error("Round %d: Seedream error: %s", round_num, e)
            image = b""

        if not image:
            return b"", EvalResult(
                passed=False, total_score=0.0, suggestion="Generation failed"
            )

        good_refs, bad_refs = pick_eval_references()
        eval_result = self.eval.evaluate(
            generated_image=image,
            input_data=input_data,
            good_refs=good_refs,
            bad_refs=bad_refs,
        )

        logger.info(
            "Round %d: eval score=%.1f dims=%s",
            round_num, eval_result.total_score, eval_result.dimensions,
        )
        return image, eval_result

    def _adjust_prompt(self, original_prompt: str, eval_result: EvalResult) -> str:
        """Use LLM to make targeted adjustments to the original prompt."""
        feedback = (
            f"Scores: {eval_result.dimensions}\n"
            f"Issues: {eval_result.issues}\n"
            f"Suggestion: {eval_result.suggestion}"
        )

        response = self.adj_client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "system", "content": ADJUST_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Original prompt:\n{original_prompt}\n\n"
                    f"Evaluation feedback:\n{feedback}\n\n"
                    f"Return the adjusted prompt:"
                )},
            ],
            max_tokens=2000,
        )
        adjusted = response.choices[0].message.content.strip()
        logger.info("Prompt adjusted: %s", adjusted[:200])
        return adjusted

    def _reroll_prompt(self, input_data: dict) -> str:
        """Use LLM to generate a completely fresh prompt from input data."""
        response = self.adj_client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "system", "content": REROLL_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Generate a badge prompt using this data:\n"
                    f"{__import__('json').dumps(input_data, indent=2)}\n\n"
                    f"Return ONLY the prompt text:"
                )},
            ],
            max_tokens=2000,
        )
        rerolled = response.choices[0].message.content.strip()
        logger.info("Prompt rerolled: %s", rerolled[:200])
        return rerolled
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_orchestrator.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add orchestrator with generate-eval-retry-reroll pipeline"
```

---

### Task 6: Integration — Run Script & Smoke Test

**Files:**
- Create: `run_orchestrator.py`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Create run script**

```python
# run_orchestrator.py
"""CLI script to run the badge generation orchestration pipeline."""

import json
import logging
import sys

from eval_client import EvalClient
from orchestrator import Orchestrator
from seedream_sdk import SeedreamClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

SAMPLE_INPUT = {
    "text_output": "Wells",
    "anchor_characterization": (
        "This Jordanian anchor hosts an energetic and highly interactive "
        "'Just Chatting' stream. He engages directly with his audience, "
        "responding to song requests, comments, and playful banter with "
        "quick-witted humor. Visually, he presents a casual style, "
        "frequently seen in a signature blue hoodie."
    ),
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


def main():
    input_data = SAMPLE_INPUT
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            input_data = json.load(f)

    orch = Orchestrator(
        seedream_client=SeedreamClient(),
        eval_client=EvalClient(),
    )

    result = orch.run(input_data)

    print(f"\n{'='*60}")
    print(f"Result: {'PASSED' if result.passed else 'FAILED'}")
    print(f"Score: {result.score:.1f}")
    print(f"Rounds: {result.rounds}")
    print(f"Image size: {len(result.image)} bytes")

    if result.image:
        out_path = "output/orchestrator_result.jpg"
        with open(out_path, "wb") as f:
            f.write(result.image)
        print(f"Saved to: {out_path}")

    for i, (prompt, ev) in enumerate(zip(result.prompt_history, result.eval_history)):
        print(f"\n--- Round {i} ---")
        print(f"Score: {ev.total_score:.1f} | Passed: {ev.passed}")
        print(f"Dimensions: {ev.dimensions}")
        if ev.issues:
            print(f"Issues: {ev.issues}")
        print(f"Prompt: {prompt[:150]}...")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS (existing 9 + new ~23)

- [ ] **Step 3: Update CLAUDE.md with orchestrator docs**

Add to the Commands section:
```
python3 run_orchestrator.py                    # Run orchestration with sample input
python3 run_orchestrator.py input.json         # Run with custom input JSON
```

Add a new section documenting the eval pipeline architecture and dimensions.

- [ ] **Step 4: Commit**

```bash
git add run_orchestrator.py CLAUDE.md
git commit -m "feat: add orchestrator run script and update docs"
```

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Push to remote**

```bash
git push
```

---

Plan complete and saved to `docs/superpowers/plans/2026-04-01-eval-retry-pipeline.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session, batch execution with checkpoints

Which approach?