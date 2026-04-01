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
            "Structure: heart carrier occupies ~85% of canvas, plump and rounded, "
            "glossy surface with visible light reflection. "
            "Subject: character from chest up, occupies ~65-70% of heart area, "
            "centered horizontally, face clearly visible and expressive. "
            "Text: 3-letter name 'Aya' at bottom-center, partially overlapping heart "
            "edge, bold cursive 3D metallic chrome with secondary color outline, "
            "sparkle/light sweep effects, fully legible. "
            "Decorations: scattered small icons and doodles fill heart background "
            "around character, adding visual richness without obscuring subject. "
            "Color: high saturation candy-like palette, colors look appetizing and "
            "vibrant, strong contrast between heart/text/character. "
            "Quality: no artifacts, no deformed features, clean edges throughout."
        ),
        is_good=True,
        score=9.0,
        issues=[],
    ),
    EvalReference(
        image_path=os.path.join(EVAL_REFERENCES_DIR, "good_02.png"),
        description=(
            "Structure: heart carrier occupies ~80% of canvas, plump and glossy, "
            "smooth surface with gradient sheen. "
            "Subject: chibi-proportioned character, occupies ~75% of heart area, "
            "centered, full upper body visible with clear facial features and "
            "dynamic pose (waving hand). "
            "Text: 2-character Japanese text 'あい' at bottom-center, overlapping "
            "heart edge, bold 3D metallic chrome with color-matched tint, sparkle "
            "effects on edges, fully legible. "
            "Decorations: hand-drawn heart outlines on heart surface, subtle and "
            "complementary, not cluttered. "
            "Color: high saturation purple-pink candy palette, vivid and bright, "
            "character clothing provides complementary blue contrast. "
            "Quality: no artifacts, crisp details, clean rendering throughout."
        ),
        is_good=True,
        score=9.0,
        issues=[],
    ),
    EvalReference(
        image_path=os.path.join(EVAL_REFERENCES_DIR, "good_03.png"),
        description=(
            "Structure: heart carrier occupies ~80% of canvas, plump shape with "
            "soft fluffy/plush texture, pillow-like feel. "
            "Subject: character from shoulder up in side profile, occupies ~60% of "
            "heart area, positioned slightly left of center, face and hair clearly "
            "visible. "
            "Text: 6-letter name 'Hayley' at bottom-center, partially overlapping "
            "heart edge, bold cursive 3D metallic chrome with holographic iridescent "
            "rainbow effect, fully legible. "
            "Decorations: small 3D clouds floating around heart, minimal and clean. "
            "Color: soft pastel blue palette, still saturated enough to feel candy-like "
            "and pleasant, gentle contrast between elements. "
            "Quality: no artifacts, smooth rendering, harmonious and balanced composition."
        ),
        is_good=True,
        score=8.5,
        issues=[],
    ),
]

_BAD_REFERENCES = [
    EvalReference(
        image_path=os.path.join(EVAL_REFERENCES_DIR, "bad_01.png"),
        description=(
            "Structure: heart carrier occupies ~90% of canvas but character is "
            "squeezed inside with a large animal, composition feels cramped. "
            "Subject: character and leopard together fill the heart, character "
            "face pushed to right edge, arms truncated by heart boundary. "
            "Text: NO text present at all — missing name/title completely. "
            "Decorations: only a few small hearts etched on heart surface, minimal. "
            "Color: purple heart is saturated but overall image feels heavy, "
            "dark shadows on character's skin. "
            "Quality: character arms clipped at heart edge, unnatural cropping."
        ),
        is_good=False,
        score=4.0,
        issues=[
            "No text/name rendered at bottom — critical missing element",
            "Character arms truncated at heart boundary, unnatural cropping",
            "Composition too cramped — two subjects competing for space",
            "Subject not centered, pushed to right side",
        ],
    ),
    EvalReference(
        image_path=os.path.join(EVAL_REFERENCES_DIR, "bad_02.png"),
        description=(
            "Structure: heart carrier occupies only ~30% of canvas, far too small, "
            "positioned at bottom-center like a prop rather than background carrier. "
            "Subject: character is ABOVE and OUTSIDE the heart, body from waist up "
            "extends far beyond heart boundary, arms spread wide outside canvas — "
            "heart is behind character not framing them. "
            "Text: NO text present at all. "
            "Decorations: none visible. "
            "Color: dark grey clothing with muted pink heart, low saturation, "
            "looks dull and muddy, excessive dark shadows on skin making image "
            "appear dirty and dim overall. "
            "Quality: character proportions distorted, arms stretched unnaturally."
        ),
        is_good=False,
        score=2.0,
        issues=[
            "Heart carrier far too small — should be background, not a small prop",
            "Character not inside heart — body extends way beyond heart boundary",
            "No text/name rendered — critical missing element",
            "Colors look dirty and dim — excessive dark shadows, low saturation",
            "No decorative elements at all",
            "Arms stretched and truncated at canvas edge",
        ],
    ),
    EvalReference(
        image_path=os.path.join(EVAL_REFERENCES_DIR, "bad_03.png"),
        description=(
            "Structure: heart carrier occupies ~60% of canvas but character is "
            "too large, head and shoulders overflow heart boundary significantly. "
            "Subject: character from chest up, but occupies ~90% of heart AND "
            "extends beyond it — arms truncated at both sides by canvas edge, "
            "head clips above heart top. "
            "Text: NO text present at all. "
            "Decorations: single lightning bolt icon, very sparse. "
            "Color: gold heart with teal shirt, dark heavy shadows on face and "
            "skin creating a muddy/dirty appearance, overall image feels dark "
            "and dim rather than bright candy-like palette. "
            "Quality: arms cut off at both sides, overall composition unbalanced."
        ),
        is_good=False,
        score=3.0,
        issues=[
            "No text/name rendered at bottom — critical missing element",
            "Character overflows heart boundary — head and arms clip outside",
            "Colors look dark and muddy — heavy shadows, not candy-like",
            "Arms truncated at canvas edges on both sides",
            "Heart too small relative to character, not framing properly",
            "Overall dim and dark, lacks vibrancy and brightness",
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
