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
