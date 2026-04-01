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
You are a quality evaluator for AI-generated badge images.

## Hard Gates (pass/fail)

Check these FIRST. If either fails, the image is rejected regardless of scores.

- background_black: Is the background SOLID BLACK? Any visible gradient, glow, \
color cast, or non-black area → false.
- heart_shape_clean: Is the heart a CLEAN rounded 3D shape? Any attached \
protrusions (horns, wings, tails, extra shapes growing FROM the heart) → false. \
Floating decorative elements near but NOT attached to the heart are OK.

## Scored Dimensions (1-10 each)

- character_likeness: Compare the generated character to the provided anchor photo. \
Does the character resemble the actual person? Check: gender, hair style/color, \
facial features (beard, glasses, moles, skin tone), age impression, expression. \
If no anchor photo is provided, evaluate against the text description only.
- heart_quality: Is the heart material candy/jelly with glossy surface (NOT balloon \
or plastic)? Does it occupy ~80% of the canvas? Is the color from the brand palette?
- decoration_harmony: Are decorations restrained and visually unified? Prefer \
single-color or minimal-color doodle/outline style. Penalize: too many different \
materials, too many colors, overcrowded elements competing with the character.
- composition: Character framed chest-up occupying ~70% of heart? Text present \
at bottom (not oversized, not obscuring character)? Overall visual balance good?

Respond with ONLY valid JSON:
{
  "hard_gates": {
    "background_black": <bool>,
    "heart_shape_clean": <bool>
  },
  "dimensions": {
    "character_likeness": <float>,
    "heart_quality": <float>,
    "decoration_harmony": <float>,
    "composition": <float>
  },
  "issues": [<list of specific problems found, empty if none>],
  "suggestion": "<targeted fix suggestion, empty if passed>"
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
        anchor_photo: bytes | None = None,
    ) -> EvalResult:
        """Evaluate a generated image against criteria and references."""
        user_content = self._build_user_message(
            generated_image, input_data, good_refs, bad_refs, anchor_photo
        )

        logger.info("Calling eval: text_output=%s", input_data.get("text_output"))

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=1000,
        )

        return self._parse_response(response)

    def _build_user_message(
        self,
        generated_image: bytes,
        input_data: dict,
        good_refs: list[EvalReference],
        bad_refs: list[EvalReference],
        anchor_photo: bytes | None = None,
    ) -> list[dict]:
        """Build multimodal user message with images and criteria."""
        content = []

        criteria = {
            "text_output": input_data.get("text_output", ""),
            "brand_palette": input_data.get("brand_palette", {}),
            "anchor_characterization": input_data.get("anchor_characterization", ""),
        }
        content.append({
            "type": "text",
            "text": f"## Evaluation Criteria\n{json.dumps(criteria, indent=2)}",
        })

        if anchor_photo:
            content.append({
                "type": "text",
                "text": "## Anchor photo — the character should resemble this person",
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(anchor_photo).decode()}"
                },
            })

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
        """Parse GPT response: check hard gates, then compute score from dimensions."""
        text = response.choices[0].message.content
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.error("Failed to parse eval response: %s", text)
            return EvalResult(passed=False, total_score=0.0, suggestion="Eval parse error")

        hard_gates = data.get("hard_gates", {})
        dimensions = data.get("dimensions", {})
        issues = data.get("issues", [])
        suggestion = data.get("suggestion", "")

        # Hard gate failure → instant reject
        if not hard_gates.get("background_black", True) or not hard_gates.get("heart_shape_clean", True):
            return EvalResult(
                passed=False,
                total_score=0.0,
                dimensions=dimensions,
                issues=issues,
                suggestion=suggestion,
            )

        # Simple average of 4 dimensions
        scores = list(dimensions.values())
        total_score = sum(scores) / len(scores) if scores else 0.0

        return EvalResult(
            passed=total_score >= 8.0,
            total_score=round(total_score, 1),
            dimensions=dimensions,
            issues=issues,
            suggestion=suggestion,
        )
