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
