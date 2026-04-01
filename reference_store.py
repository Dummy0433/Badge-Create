# reference_store.py
"""Reference image store for few-shot style guidance.

Currently hardcoded with 3 reference images. Will be replaced with a
database-backed store that supports dynamic dataset injection.
"""

import os
import random
from dataclasses import dataclass

REFERENCES_DIR = os.path.join(os.path.dirname(__file__), "references")


@dataclass
class ReferenceImage:
    """A reference image with its description for PE injection."""

    image_path: str
    description: str

    def load_bytes(self) -> bytes:
        with open(self.image_path, "rb") as f:
            return f.read()


# Hardcoded reference images — will be replaced with DB lookup
_REFERENCES = [
    ReferenceImage(
        image_path=os.path.join(REFERENCES_DIR, "ref_01_gelik.png"),
        description=(
            "C4D Badge, 3D Pixar realistic cartoon style. A large plump rounded solid 3D heart shape "
            "as the background carrier, smooth glossy surface with gradient rose-pink color. A female "
            "character with long dark brown hair, both hands raised framing her head, playful expression "
            "with tongue sticking out, wearing a pink zip-up hoodie, positioned from chest up occupying "
            "70% of the heart. Small pink 3D butterflies floating around the heart as decorative elements. "
            '3D bold thick retro cursive gradient text "Gelik" at the bottom of the heart, silver chrome '
            "sweep light effect with pink-tinted holographic iridescent metallic material. Pure black "
            "background, candy pink color palette, commercial art illustration style."
        ),
    ),
    ReferenceImage(
        image_path=os.path.join(REFERENCES_DIR, "ref_02_hikachuu.png"),
        description=(
            "C4D Badge, 3D Pixar realistic cartoon style. A large plump rounded solid 3D heart shape "
            "as the background carrier, smooth glossy surface with deep cherry-red color. A female "
            "character with brown bob-length hair wearing a brown bear ear hood onesie, white face mask, "
            "white wired earbuds, making a pointing-up gesture with right hand, positioned from chest up "
            "occupying 70% of the heart. Small 3D social media icons floating around the heart — music "
            "note, video play button, chat bubble, emoji as decorative elements. 3D bold thick gradient "
            'text "ひかちゅう" at the bottom of the heart, silver chrome sweep light effect holographic '
            "iridescent metallic material. Pure black background, warm red color palette, commercial art "
            "illustration style."
        ),
    ),
    ReferenceImage(
        image_path=os.path.join(REFERENCES_DIR, "ref_03_makimaki.png"),
        description=(
            "C4D Badge, 3D Pixar realistic cartoon style. A large plump rounded solid 3D heart shape "
            "as the background carrier, smooth glossy surface with gradient pink-to-magenta color. A "
            "female character with medium-length brown hair wearing a white cat ear hood with a hair clip, "
            "happy singing expression with mouth open, holding a beige comb in one hand, professional "
            "studio microphone on a stand to the right, positioned from chest up occupying 70% of the "
            "heart. Small hand-drawn graffiti-style outline hearts and love doodles on the heart surface "
            'as decorative elements. 3D bold thick gradient text "まきまき" at the bottom of the heart, '
            "silver chrome sweep light effect with pink-tinted holographic iridescent metallic material. "
            "Pure black background, candy pink color palette, commercial art illustration style."
        ),
    ),
]


def get_all_references() -> list[ReferenceImage]:
    return list(_REFERENCES)


def pick_references(count: int = 2) -> list[ReferenceImage]:
    """Randomly pick N reference images from the store."""
    count = min(count, len(_REFERENCES))
    return random.sample(_REFERENCES, count)
