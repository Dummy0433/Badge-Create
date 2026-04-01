# orchestrator.py
"""Orchestrator: unified pipeline with retry escalation, batch, and sweep."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from random import randint

from eval_client import EvalClient, EvalResult
from eval_store import pick_eval_references
from generator import Generator, GenerateResult
from input_processor import process
from prompt_builder import (
    analyze_photo, build_prompt, expand_prompt,
    build_negative_prompt, PromptValidationError,
)
from reference_store import pick_references

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output"


@dataclass
class UnitResult:
    """Result of one pipeline unit (possibly after retries)."""

    image: bytes
    score: float
    passed: bool
    seed: int
    prompt: str
    eval_result: EvalResult
    rounds: int
    request_id: str


@dataclass
class BatchResult:
    """Result of a batch run."""

    results: list[UnitResult] = field(default_factory=list)
    prompt: str = ""
    keywords: dict = field(default_factory=dict)
    total: int = 0
    success: int = 0
    failed: int = 0


class Orchestrator:
    """Unified pipeline: input -> prompt -> generate -> eval -> retry -> batch."""

    def __init__(
        self,
        llm_client,
        generator: Generator,
        eval_client: EvalClient,
        max_retries: int = 2,
        max_reexpands: int = 1,
        pass_threshold: float = 8.0,
        max_workers: int = 10,
    ):
        self.llm_client = llm_client
        self.generator = generator
        self.eval_client = eval_client
        self.max_retries = max_retries
        self.max_reexpands = max_reexpands
        self.pass_threshold = pass_threshold
        self.max_workers = max_workers

    def run_batch(
        self,
        input_data: dict,
        photo_bytes: bytes | None = None,
        count: int = 1,
        use_refs: bool = True,
        ref_count: int = 2,
    ) -> BatchResult:
        """Main entry point for both single and batch generation.

        1. Normalize input
        2. Analyze photo (if provided)
        3. Build prompt (keywords + expand)
        4. Parallel N x _run_single_unit with retry escalation
        5. Sort results by score
        """
        input_data = process(input_data)

        if photo_bytes and not input_data.get("photo_analysis"):
            input_data["photo_analysis"] = analyze_photo(self.llm_client, photo_bytes)

        keywords, prompt = build_prompt(input_data, client=self.llm_client)
        negative_prompt = build_negative_prompt()
        logger.info("Prompt built (%d chars)", len(prompt))

        refs = pick_references(ref_count) if use_refs else []
        good_refs, bad_refs = pick_eval_references()

        def _run_unit(_idx: int) -> UnitResult:
            return self._run_single_unit(
                prompt=prompt,
                negative_prompt=negative_prompt,
                keywords=keywords,
                input_data=input_data,
                photo_bytes=photo_bytes,
                refs=refs,
                good_refs=good_refs,
                bad_refs=bad_refs,
            )

        results = []
        workers = min(count, self.max_workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_unit, i): i for i in range(count)}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error("Unit %d failed: %s", futures[future], e)

        results.sort(key=lambda r: r.score, reverse=True)
        success = sum(1 for r in results if r.passed)

        return BatchResult(
            results=results,
            prompt=prompt,
            keywords=keywords,
            total=count,
            success=success,
            failed=count - len(results),
        )

    def run_sweep(
        self,
        input_data: dict,
        photo_bytes: bytes | None = None,
        param_combos: list[dict] | None = None,
        use_refs: bool = True,
        ref_count: int = 2,
    ) -> BatchResult:
        """Like run_batch but each unit uses different generation params."""
        if not param_combos:
            return self.run_batch(input_data, photo_bytes, count=1,
                                  use_refs=use_refs, ref_count=ref_count)

        input_data = process(input_data)

        if photo_bytes and not input_data.get("photo_analysis"):
            input_data["photo_analysis"] = analyze_photo(self.llm_client, photo_bytes)

        keywords, prompt = build_prompt(input_data, client=self.llm_client)
        negative_prompt = build_negative_prompt()

        refs = pick_references(ref_count) if use_refs else []
        good_refs, bad_refs = pick_eval_references()

        def _run_unit(params: dict) -> UnitResult:
            return self._run_single_unit(
                prompt=prompt,
                negative_prompt=negative_prompt,
                keywords=keywords,
                input_data=input_data,
                photo_bytes=photo_bytes,
                refs=refs,
                good_refs=good_refs,
                bad_refs=bad_refs,
                extra_gen_kwargs=params,
            )

        results = []
        workers = min(len(param_combos), self.max_workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_unit, p): p for p in param_combos}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error("Sweep unit failed: %s", e)

        results.sort(key=lambda r: r.score, reverse=True)
        success = sum(1 for r in results if r.passed)

        return BatchResult(
            results=results,
            prompt=prompt,
            keywords=keywords,
            total=len(param_combos),
            success=success,
            failed=len(param_combos) - len(results),
        )

    def _run_single_unit(
        self,
        prompt: str,
        negative_prompt: str,
        keywords: dict,
        input_data: dict,
        photo_bytes: bytes | None,
        refs: list,
        good_refs: list,
        bad_refs: list,
        extra_gen_kwargs: dict | None = None,
    ) -> UnitResult:
        """One atomic unit with retry escalation.

        Round 0:   generate(prompt) -> eval
        Level 1:   generate(prompt, new_seed) -> eval  (up to max_retries)
        Level 2:   expand(keywords) -> generate -> eval (up to max_reexpands)
        Exhaust:   return best scoring result
        """
        gen_kwargs = {
            "guidance_scale": 8.0,
            "cfg_rescale_factor": 0.0,
            "single_edit_guidance_weight": 2.0,
            "single_edit_guidance_weight_image": 1.0,
        }
        if extra_gen_kwargs:
            gen_kwargs.update(extra_gen_kwargs)

        best: UnitResult | None = None
        current_prompt = prompt
        round_num = 0

        def _attempt(p: str) -> UnitResult:
            nonlocal round_num
            round_num += 1
            seed = randint(0, 2**31)
            gen_result = self.generator.generate(
                prompt=p,
                negative_prompt=negative_prompt,
                seed=seed,
                photo_bytes=photo_bytes,
                refs=refs if refs else None,
                **gen_kwargs,
            )
            self._save_image(gen_result.image, seed, round_num)
            eval_result = self.eval_client.evaluate(
                generated_image=gen_result.image,
                input_data=input_data,
                good_refs=good_refs,
                bad_refs=bad_refs,
                anchor_photo=photo_bytes,
            )
            logger.info("Round %d: seed=%d score=%.1f", round_num, seed, eval_result.total_score)
            return UnitResult(
                image=gen_result.image,
                score=eval_result.total_score,
                passed=eval_result.passed,
                seed=seed,
                prompt=p,
                eval_result=eval_result,
                rounds=round_num,
                request_id=gen_result.request_id,
            )

        def _track_best(unit: UnitResult) -> UnitResult:
            nonlocal best
            if best is None or unit.score > best.score:
                best = unit
            return unit

        # Initial attempt
        result = _track_best(_attempt(current_prompt))
        if result.passed:
            return result

        # Level 1: cheap retries (new seed, same prompt)
        for _ in range(self.max_retries):
            result = _track_best(_attempt(current_prompt))
            if result.passed:
                return result

        # Level 2: re-expand prompt from same keywords
        for _ in range(self.max_reexpands):
            current_prompt = expand_prompt(self.llm_client, keywords)
            logger.info("Level 2: re-expanded prompt (%d chars)", len(current_prompt))
            result = _track_best(_attempt(current_prompt))
            if result.passed:
                return result

        # All exhausted — return best
        logger.warning("All %d rounds exhausted, returning best (score=%.1f)",
                        round_num, best.score)
        best.rounds = round_num
        return best

    def _save_image(self, image: bytes, seed: int, round_num: int) -> None:
        """Save intermediate image for debugging."""
        if not image:
            return
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(OUTPUT_DIR, f"{ts}_r{round_num}_s{seed}.jpg")
        with open(path, "wb") as f:
            f.write(image)
        logger.info("Saved to %s", path)
