# prompt_builder.py
"""LLM-powered prompt assembly for badge generation.

Pipeline: analyze_photo → assemble_keywords → expand_prompt → validate
"""

import base64
import json
import logging
import os

import openai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class PromptValidationError(Exception):
    """Raised when assembled prompt fails pre-checks."""
    pass


# --- System prompts for each LLM step ---

PHOTO_ANALYSIS_PROMPT = """\
You are a visual analyst. Analyze the person in this photo and extract their \
physical appearance features. Focus ONLY on objectively visible traits.

Return ONLY valid JSON:
{
  "gender": "male/female",
  "hair": "description of hair style, length, color",
  "eyes": "eye color and shape",
  "expression": "facial expression",
  "skin_tone": "skin tone description",
  "notable_features": "any distinctive features like glasses, beard, accessories"
}"""

KEYWORD_ASSEMBLY_PROMPT = """\
You are a creative director assembling structured keywords for a 3D badge image.

Given the input data (anchor characterization, brand palette, photo analysis), \
produce a structured keyword JSON that maps each input field to badge elements.

Rules:
- Character clothing comes from anchor_characterization (their signature style), \
NOT from the photo. If no specific clothing mentioned, use a casual style that \
matches the anchor's personality.
- Decorations are derived from anchor's personality, interests, and stream themes. \
Include the anchor's emoji or mascot if present in their nickname.
- text_output is used EXACTLY as-is for the badge text — do not translate or modify.

COLOR ASSIGNMENT — do NOT mechanically map primary=heart. Instead:
- Look at ALL three palette colors and assign them for maximum visual impact.
- The heart carrier should use the most VIBRANT, eye-catching color from the palette.
- Background should provide strong CONTRAST with the heart.
- If primary is a dark/neutral color (black, grey, white), use secondary or tertiary \
for the heart instead.
- Text accents should complement the heart color.
- Overall palette should feel vibrant, high-saturation, and visually appealing.

Return ONLY valid JSON:
{
  "character": {
    "gender": "<from photo_analysis>",
    "hair": "<from photo_analysis>",
    "eyes": "<from photo_analysis>",
    "expression": "<from photo_analysis>",
    "clothing": "<derived from anchor_characterization>"
  },
  "heart_carrier": {
    "color_name": "<chosen vibrant color name>",
    "color_hex": "<hex>",
    "material": "soft candy or jelly, smooth glossy surface"
  },
  "decorations": {
    "elements": ["<3-5 small icons/doodles derived from anchor personality>"]
  },
  "text": {
    "content": "<text_output EXACTLY as provided>",
    "color_tint": "<complementary color hex>"
  },
  "background_color": "<contrasting color hex>"
}"""

PROMPT_EXPANSION_PROMPT = """\
You are a prompt engineer writing image generation prompts for Seedream 4.5.

Given structured keywords, expand them into a single detailed paragraph prompt.

FIXED structure you MUST include:
- Start with: "C4D Badge, 3D Pixar realistic cartoon style."
- Heart: "A large plump rounded solid 3D heart shape as the background carrier, \
the heart is thick and voluminous like soft candy or jelly, smooth glossy surface"
- Anti-balloon: "NOT a balloon NOT inflatable, no wrinkles no seams no strings"
- Character: "positioned in front of the upper area of the heart from chest up, \
occupying 70% of the heart"
- Text: "3D bold thick retro cursive gradient text" with "silver chrome sweep \
light effect, holographic iridescent metallic material" and "at the bottom of \
the heart, partially extending beyond the heart edge"
- Lighting: "Warm side light from left, cool side light from right, soft front key light"
- End with: "vibrant high-saturation color palette, commercial art illustration style"

VARIABLE elements from keywords:
- Heart gradient color (from heart_carrier)
- Character appearance + clothing (from character)
- Decorations (from decorations.elements)
- Text content in EXACT quotes (from text.content — do NOT change the text)
- Background color (from background_color)

IMPORTANT: The text.content value must appear EXACTLY in the prompt wrapped in \
double quotes. Do not translate, modify, or paraphrase it.

Return ONLY the prompt text. One single paragraph."""


def analyze_photo(client: openai.OpenAI, image_bytes: bytes) -> dict:
    """Step 1: GPT vision analyzes photo → returns photo_analysis dict."""
    logger.info("Step 1: Analyzing photo with GPT vision...")

    b64 = base64.b64encode(image_bytes).decode()
    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "system", "content": PHOTO_ANALYSIS_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Analyze this person's appearance:"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                }},
            ]},
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=500,
    )

    result = json.loads(response.choices[0].message.content)
    logger.info("Photo analysis: %s", result)
    return result


def assemble_keywords(client: openai.OpenAI, input_data: dict) -> dict:
    """Step 2: LLM reads full input → returns structured keyword JSON."""
    logger.info("Step 2: Assembling keywords...")

    # Build context for LLM — include all useful fields
    context = {
        "text_output": input_data.get("text_output", ""),
        "anchor_characterization": input_data.get("anchor_characterization", ""),
        "brand_palette": input_data.get("brand_palette", {}),
        "photo_analysis": input_data.get("photo_analysis", {}),
        "anchor_nickname": input_data.get("anchor_nickname", ""),
        "anchor_bio": input_data.get("anchor_bio", ""),
    }

    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "system", "content": KEYWORD_ASSEMBLY_PROMPT},
            {"role": "user", "content": (
                f"Assemble badge keywords from this input:\n"
                f"{json.dumps(context, indent=2)}"
            )},
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=800,
    )

    result = json.loads(response.choices[0].message.content)
    logger.info("Keywords assembled: %s", json.dumps(result, indent=2)[:500])
    return result


def expand_prompt(client: openai.OpenAI, keywords: dict) -> str:
    """Step 3: LLM expands keywords into final Seedream prompt."""
    logger.info("Step 3: Expanding keywords into prompt...")

    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "system", "content": PROMPT_EXPANSION_PROMPT},
            {"role": "user", "content": (
                f"Expand these keywords into a Seedream prompt:\n"
                f"{json.dumps(keywords, indent=2)}"
            )},
        ],
        max_completion_tokens=1500,
    )

    prompt = response.choices[0].message.content.strip()
    logger.info("Expanded prompt: %s", prompt[:300])
    return prompt


def validate_prompt(prompt: str, input_data: dict) -> None:
    """Step 4: Pre-check the assembled prompt.

    Raises PromptValidationError if critical elements are missing.
    """
    text_output = input_data.get("text_output", "").strip()
    if not text_output:
        raise PromptValidationError("text_output is empty or missing")

    if text_output not in prompt:
        raise PromptValidationError(
            f"text_output '{text_output}' not found in assembled prompt"
        )

    # Check fixed style elements are present
    if "C4D Badge" not in prompt and "3D" not in prompt:
        raise PromptValidationError("Missing render style (C4D Badge / 3D)")

    if "heart" not in prompt.lower():
        raise PromptValidationError("Missing heart carrier description")


def build_prompt(input_data: dict) -> str:
    """Full pipeline: analyze photo (if needed) → keywords → expand → validate.

    For backward compatibility, also works without an OpenAI client if
    photo_analysis is already provided — falls back to template mode.
    """
    # Validate minimum required fields
    text_output = input_data.get("text_output", "").strip()
    if not text_output:
        raise PromptValidationError("text_output is empty or missing")

    if not input_data.get("photo_analysis"):
        raise PromptValidationError("photo_analysis is missing")

    if not input_data.get("brand_palette"):
        raise PromptValidationError("brand_palette is missing")

    # Use LLM pipeline if client available, otherwise fallback to template
    client = _get_client()
    if client:
        keywords = assemble_keywords(client, input_data)
        prompt = expand_prompt(client, keywords)
        validate_prompt(prompt, input_data)
        return prompt

    # Fallback: simple template (for tests without OpenAI key)
    return _template_fallback(input_data)


def build_negative_prompt() -> str:
    """Return the fixed negative prompt for badge generation."""
    return (
        "nsfw, balloon, inflatable, wrinkles, seams, strings, rope, "
        "flat, 2D, thin, metal frame, hard border, badge pin, deflated"
    )


def _get_client() -> openai.OpenAI | None:
    """Get OpenAI client if key is available."""
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return openai.OpenAI(api_key=key)
    return None


def _template_fallback(input_data: dict) -> str:
    """Simple template-based prompt assembly (no LLM, for tests)."""
    photo = input_data["photo_analysis"]
    palette = input_data["brand_palette"]
    heart_color = f"{palette['primary']['name']} (#{palette['primary']['hex'].lstrip('#')})"
    bg_color = palette["tertiary"]["hex"]

    return (
        f'C4D Badge, 3D Pixar realistic cartoon style. '
        f'A large plump rounded solid 3D heart shape as the background carrier, '
        f'the heart is thick and voluminous like soft candy or jelly, '
        f'smooth glossy surface with gradient {heart_color} color, '
        f'NOT a balloon NOT inflatable, no wrinkles no seams no strings, '
        f'just a clean solid puffy 3D heart object. '
        f'A {photo.get("gender", "")} character with {photo.get("hair", "")}, '
        f'{photo.get("eyes", "")}, {photo.get("expression", "")}, '
        f'the character is positioned in front of the upper area of the heart '
        f'from chest up, occupying 70% of the heart. '
        f'3D bold thick retro cursive gradient text "{input_data["text_output"]}" '
        f'at the bottom of the heart, partially extending beyond the heart edge, '
        f'the text has silver chrome sweep light effect, '
        f'holographic iridescent metallic material. '
        f'Warm side light from left, cool side light from right, '
        f'soft front key light. '
        f'Pure {bg_color} background, vibrant high-saturation color palette, '
        f'commercial art illustration style.'
    )
