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
        for sentence in characterization.split("."):
            if "hoodie" in sentence.lower():
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
