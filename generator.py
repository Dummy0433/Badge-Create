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
