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
            max_completion_tokens=1000,
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
