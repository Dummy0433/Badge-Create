# orchestrator.py
"""Orchestrator: generate → eval → retry/reroll pipeline."""

import json
import logging
import os
from dataclasses import dataclass, field
from random import randint

import openai
from dotenv import load_dotenv

from eval_client import EvalClient, EvalResult
from eval_store import pick_eval_references
from prompt_builder import (
    analyze_photo, assemble_keywords, expand_prompt,
    validate_prompt, build_negative_prompt, PromptValidationError,
)
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
        """Full pipeline: photo analysis → keywords → prompt → generate → eval → retry/reroll."""

        # Step 1: Analyze photo if photo_analysis not provided
        if not input_data.get("photo_analysis"):
            anchor_photo = input_data.get("anchor_photo", "")
            if not anchor_photo:
                raise PromptValidationError("Neither photo_analysis nor anchor_photo provided")
            # Load photo bytes
            if os.path.isfile(anchor_photo):
                with open(anchor_photo, "rb") as f:
                    photo_bytes = f.read()
            else:
                import base64 as b64mod
                photo_bytes = b64mod.b64decode(anchor_photo)
            input_data["photo_analysis"] = analyze_photo(self.adj_client, photo_bytes)
            logger.info("Photo analysis complete: %s", input_data["photo_analysis"])

        # Step 2-3: Assemble keywords → expand into prompt
        keywords = assemble_keywords(self.adj_client, input_data)
        original_prompt = expand_prompt(self.adj_client, keywords)

        # Step 4: Validate
        validate_prompt(original_prompt, input_data)
        logger.info("Prompt validated. Starting generation loop.")

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
            max_completion_tokens=2000,
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
                    f"{json.dumps(input_data, indent=2)}\n\n"
                    f"Return ONLY the prompt text:"
                )},
            ],
            max_completion_tokens=2000,
        )
        rerolled = response.choices[0].message.content.strip()
        logger.info("Prompt rerolled: %s", rerolled[:200])
        return rerolled
