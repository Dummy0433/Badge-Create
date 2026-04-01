# server.py
"""FastAPI server exposing Seedream 4.5 SDK as a REST API."""

import base64
import itertools
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from random import randint

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from eval_client import EvalClient
from eval_store import pick_eval_references
from orchestrator import preprocess_input
from prompt_builder import analyze_photo, assemble_keywords, expand_prompt, validate_prompt, build_negative_prompt

# Lazy-init OpenAI client for prompt building
_llm_client = None

def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        import openai as _openai
        from dotenv import load_dotenv
        load_dotenv()
        _llm_client = _openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _llm_client
from reference_store import pick_references, ReferenceImage
from seedream_sdk import SeedreamClient, SeedreamAPIError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Seedream 4.5 Test UI")

client = SeedreamClient()

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _inject_references(
    images_bytes: list[bytes] | None,
    refs: list[ReferenceImage],
) -> list[bytes]:
    """Prepend reference image bytes before user images.

    Order: [ref1, ref2, ..., user_photo]
    So that input1=ref1, input2=ref2, and user photo is last
    (associated via the output/prompt description).
    """
    result = [ref.load_bytes() for ref in refs]
    if images_bytes:
        result.extend(images_bytes)
    return result


def _build_ref_kwargs(prompt: str, refs: list[ReferenceImage]) -> dict:
    """Build use_pre_llm + pre_llm_result kwargs from references."""
    ref_descs = ", ".join(f"input{i+1}" for i in range(len(refs)))
    edit_instruction = (
        f"Generate a C4D Badge 3D Pixar realistic cartoon style image, "
        f"following the exact visual style of {ref_descs} — 3D heart carrier, "
        f"character from chest up, metallic text at bottom, pure black background."
    )
    pe = {}
    for i, ref in enumerate(refs, start=1):
        pe[f"input{i}"] = ref.description
    pe["edit"] = edit_instruction
    pe["output"] = prompt
    pe["ratio"] = "1:1"
    result = {
        "use_pre_llm": False,
        "pre_llm_result": json.dumps(pe),
    }
    logger.info("Injected %d refs, PE: %s", len(refs), result["pre_llm_result"])
    return result


@app.post("/api/build_prompt")
async def build_prompt_endpoint(
    input_json: str = Form(...),
    anchor_photo: UploadFile | None = File(None),
):
    """Build prompt from JSON + photo. Returns the generated prompt text."""
    try:
        input_data = json.loads(input_json)
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

    input_data = preprocess_input(input_data)
    llm = _get_llm_client()

    # Photo analysis
    if anchor_photo:
        photo_bytes = await anchor_photo.read()
        input_data["photo_analysis"] = analyze_photo(llm, photo_bytes)
    elif not input_data.get("photo_analysis"):
        return JSONResponse(status_code=400, content={"error": "No photo or photo_analysis"})

    keywords = assemble_keywords(llm, input_data)
    prompt = expand_prompt(llm, keywords)
    validate_prompt(prompt, input_data)

    return {
        "prompt": prompt,
        "negative_prompt": build_negative_prompt(),
        "keywords": keywords,
        "photo_analysis": input_data.get("photo_analysis", {}),
    }


@app.get("/api/references")
def list_references(count: int = 2):
    """Preview which reference images would be injected."""
    refs = pick_references(count)
    return [
        {"filename": os.path.basename(r.image_path), "description": r.description}
        for r in refs
    ]


@app.post("/api/generate")
async def generate(
    prompt: str = Form(...),
    width: int = Form(2048),
    height: int = Form(2048),
    seed: int = Form(-1),
    negative_prompt: str = Form("nsfw"),
    force_single: bool = Form(True),
    use_refs: bool = Form(False),
    ref_count: int = Form(2),
    files: list[UploadFile] | None = File(None),
):
    images_bytes = None
    if files:
        images_bytes = [await f.read() for f in files]

    extra_kwargs = {}
    if use_refs:
        refs = pick_references(ref_count)
        images_bytes = _inject_references(images_bytes, refs)
        extra_kwargs.update(_build_ref_kwargs(prompt, refs))

    try:
        result = client.generate(
            prompt=prompt,
            images=images_bytes,
            width=width,
            height=height,
            seed=seed,
            negative_prompt=negative_prompt,
            force_single=force_single,
            **extra_kwargs,
        )
    except SeedreamAPIError as e:
        logger.error("Seedream API error: %s", e)
        return JSONResponse(
            status_code=502,
            content={"error": str(e), "raw_response": e.raw_response},
        )
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )

    # Save images locally
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, img in enumerate(result.images):
        path = os.path.join(OUTPUT_DIR, f"{timestamp}_{i}.jpg")
        with open(path, "wb") as f:
            f.write(img)
        logger.info("Saved image to %s", path)

    # Encode images as base64 data URIs
    images_b64 = [
        f"data:image/jpeg;base64,{base64.b64encode(img).decode()}"
        for img in result.images
    ]

    return {
        "request_id": result.request_id,
        "llm_result": result.llm_result,
        "image_prompts": result.image_prompts,
        "images": images_b64,
        "raw_response": result.raw_response,
    }


@app.post("/api/generate_batch")
async def generate_batch(
    prompt: str = Form(...),
    width: int = Form(2048),
    height: int = Form(2048),
    negative_prompt: str = Form("nsfw"),
    force_single: bool = Form(True),
    use_refs: bool = Form(False),
    ref_count: int = Form(2),
    count: int = Form(10),
    files: list[UploadFile] | None = File(None),
):
    """Generate multiple images concurrently with different random seeds."""
    images_bytes = None
    if files:
        images_bytes = [await f.read() for f in files]

    extra_kwargs = {}
    if use_refs:
        refs = pick_references(ref_count)
        images_bytes = _inject_references(images_bytes, refs)
        extra_kwargs.update(_build_ref_kwargs(prompt, refs))

    # Generate random seeds
    seeds = [randint(0, 2**31) for _ in range(count)]

    def _call(seed: int):
        return client.generate(
            prompt=prompt,
            images=images_bytes,
            width=width,
            height=height,
            seed=seed,
            negative_prompt=negative_prompt,
            force_single=force_single,
            **extra_kwargs,
        )

    results = []
    errors = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with ThreadPoolExecutor(max_workers=min(count, 10)) as pool:
        future_to_seed = {pool.submit(_call, s): s for s in seeds}
        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                result = future.result()
                images_b64 = [
                    f"data:image/jpeg;base64,{base64.b64encode(img).decode()}"
                    for img in result.images
                ]
                # Save images locally
                for i, img in enumerate(result.images):
                    path = os.path.join(OUTPUT_DIR, f"{timestamp}_s{seed}_{i}.jpg")
                    with open(path, "wb") as f:
                        f.write(img)
                results.append({
                    "seed": seed,
                    "request_id": result.request_id,
                    "llm_result": result.llm_result,
                    "images": images_b64,
                })
            except SeedreamAPIError as e:
                logger.error("Batch seed=%d error: %s", seed, e)
                errors.append({"seed": seed, "error": str(e)})
            except Exception as e:
                logger.error("Batch seed=%d unexpected: %s", seed, e)
                errors.append({"seed": seed, "error": str(e)})

    return {
        "total": count,
        "success": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }


@app.post("/api/generate_sweep")
async def generate_sweep(
    prompt: str = Form(...),
    width: int = Form(2048),
    height: int = Form(2048),
    negative_prompt: str = Form("nsfw"),
    force_single: bool = Form(True),
    use_refs: bool = Form(False),
    ref_count: int = Form(2),
    guidance_scales: str = Form("2.5"),
    cfg_rescale_factors: str = Form("0.0"),
    edit_text_weights: str = Form("3.0"),
    edit_image_weights: str = Form("1.6"),
    files: list[UploadFile] | None = File(None),
):
    """Parameter sweep: cartesian product of all parameter arrays, each with a random seed."""
    images_bytes = None
    if files:
        images_bytes = [await f.read() for f in files]

    extra_kwargs = {}
    if use_refs:
        refs = pick_references(ref_count)
        images_bytes = _inject_references(images_bytes, refs)
        extra_kwargs.update(_build_ref_kwargs(prompt, refs))

    def _parse_floats(s: str) -> list[float]:
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    gs_list = _parse_floats(guidance_scales)
    cfg_list = _parse_floats(cfg_rescale_factors)
    tw_list = _parse_floats(edit_text_weights)
    iw_list = _parse_floats(edit_image_weights)

    has_images = bool(images_bytes)
    combos = []
    for gs, cfg, tw, iw in itertools.product(gs_list, cfg_list, tw_list, iw_list):
        params = {"guidance_scale": gs, "cfg_rescale_factor": cfg}
        if has_images:
            params["single_edit_guidance_weight"] = tw
            params["single_edit_guidance_weight_image"] = iw
        combos.append(params)

    def _call(params: dict):
        seed = randint(0, 2**31)
        result = client.generate(
            prompt=prompt,
            images=images_bytes,
            width=width,
            height=height,
            seed=seed,
            negative_prompt=negative_prompt,
            force_single=force_single,
            **extra_kwargs,
            **params,
        )
        return seed, params, result

    results = []
    errors = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("Sweep: %d combinations", len(combos))

    with ThreadPoolExecutor(max_workers=min(len(combos), 10)) as pool:
        future_map = {pool.submit(_call, p): p for p in combos}
        for future in as_completed(future_map):
            params = future_map[future]
            try:
                seed, params, result = future.result()
                images_b64 = [
                    f"data:image/jpeg;base64,{base64.b64encode(img).decode()}"
                    for img in result.images
                ]
                for i, img in enumerate(result.images):
                    path = os.path.join(OUTPUT_DIR, f"{timestamp}_sweep_s{seed}_{i}.jpg")
                    with open(path, "wb") as f:
                        f.write(img)
                results.append({
                    "seed": seed,
                    "params": params,
                    "request_id": result.request_id,
                    "llm_result": result.llm_result,
                    "images": images_b64,
                })
            except SeedreamAPIError as e:
                logger.error("Sweep params=%s error: %s", params, e)
                errors.append({"params": params, "error": str(e)})
            except Exception as e:
                logger.error("Sweep params=%s unexpected: %s", params, e)
                errors.append({"params": params, "error": str(e)})

    return {
        "total": len(combos),
        "success": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }


@app.post("/api/pipeline")
async def pipeline(
    input_json: str = Form(...),
    count: int = Form(10),
    use_refs: bool = Form(True),
    ref_count: int = Form(2),
    anchor_photo: UploadFile | None = File(None),
):
    """One-click pipeline: JSON+photo → prompt → batch generate (with photo as ref) → eval → ranked.

    This is the main endpoint. It:
    1. Builds prompt from JSON + photo (LLM photo analysis → keywords → expand)
    2. Batch generates N images using the same logic as generate_batch
       (with anchor_photo + fewshot refs passed to Seedream)
    3. Evals each image and returns results sorted by score
    """
    try:
        input_data = json.loads(input_json)
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

    input_data = preprocess_input(input_data)
    llm = _get_llm_client()

    # --- Step 1: Build prompt ---
    photo_bytes = None
    if anchor_photo:
        photo_bytes = await anchor_photo.read()
        input_data["photo_analysis"] = analyze_photo(llm, photo_bytes)
    elif not input_data.get("photo_analysis"):
        return JSONResponse(status_code=400, content={"error": "No photo provided"})

    keywords = assemble_keywords(llm, input_data)
    prompt = expand_prompt(llm, keywords)
    validate_prompt(prompt, input_data)
    negative_prompt = build_negative_prompt()
    logger.info("Pipeline: prompt built (%d chars)", len(prompt))

    # --- Step 2: Batch generate (same as generate_batch) ---
    images_bytes = [photo_bytes] if photo_bytes else None

    extra_kwargs = {
        "guidance_scale": 8.0,
        "cfg_rescale_factor": 0.0,
        "single_edit_guidance_weight": 2.0,
        "single_edit_guidance_weight_image": 1.0,
    }

    if use_refs:
        refs = pick_references(ref_count)
        images_bytes = _inject_references(images_bytes, refs)
        extra_kwargs.update(_build_ref_kwargs(prompt, refs))

    seeds = [randint(0, 2**31) for _ in range(count)]

    def _call(seed: int):
        return seed, client.generate(
            prompt=prompt,
            images=images_bytes,
            width=2048,
            height=2048,
            seed=seed,
            negative_prompt=negative_prompt,
            force_single=True,
            **extra_kwargs,
        )

    results = []
    errors = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with ThreadPoolExecutor(max_workers=min(count, 10)) as pool:
        future_to_seed = {pool.submit(_call, s): s for s in seeds}
        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                seed, result = future.result()
                for i, img in enumerate(result.images):
                    path = os.path.join(OUTPUT_DIR, f"{timestamp}_s{seed}_{i}.jpg")
                    with open(path, "wb") as f:
                        f.write(img)
                results.append({
                    "seed": seed,
                    "request_id": result.request_id,
                    "llm_result": result.llm_result,
                    "_raw_images": result.images,
                    "images": [
                        f"data:image/jpeg;base64,{base64.b64encode(img).decode()}"
                        for img in result.images
                    ],
                })
            except SeedreamAPIError as e:
                logger.error("Pipeline seed=%d error: %s", seed, e)
                errors.append({"seed": seed, "error": str(e)})
            except Exception as e:
                logger.error("Pipeline seed=%d unexpected: %s", seed, e)
                errors.append({"seed": seed, "error": str(e)})

    # --- Step 3: Eval each result ---
    eval_c = EvalClient()
    good_refs, bad_refs = pick_eval_references()

    for r in results:
        raw = r.pop("_raw_images", [])
        if raw:
            try:
                ev = eval_c.evaluate(
                    generated_image=raw[0],
                    input_data=input_data,
                    good_refs=good_refs,
                    bad_refs=bad_refs,
                )
                r["eval"] = {
                    "score": ev.total_score,
                    "passed": ev.passed,
                    "dimensions": ev.dimensions,
                    "issues": ev.issues,
                }
                logger.info("Pipeline eval seed=%d score=%.1f", r["seed"], ev.total_score)
            except Exception as e:
                logger.error("Pipeline eval seed=%d error: %s", r["seed"], e)
                r["eval"] = {"score": 0, "passed": False, "dimensions": {}, "issues": [str(e)]}

    results.sort(key=lambda r: r.get("eval", {}).get("score", 0), reverse=True)

    return {
        "total": count,
        "success": len(results),
        "failed": len(errors),
        "prompt": prompt,
        "results": results,
        "errors": errors,
    }


@app.post("/api/pipeline_sweep")
async def pipeline_sweep(
    input_json: str = Form(...),
    use_refs: bool = Form(True),
    ref_count: int = Form(2),
    guidance_scales: str = Form("8.0"),
    cfg_rescale_factors: str = Form("0.0"),
    edit_text_weights: str = Form("2.0"),
    edit_image_weights: str = Form("1.0"),
    anchor_photo: UploadFile | None = File(None),
):
    """Sweep pipeline: JSON+photo → prompt → param sweep generate → eval → ranked."""
    try:
        input_data = json.loads(input_json)
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

    input_data = preprocess_input(input_data)
    llm = _get_llm_client()

    # Step 1: Build prompt
    photo_bytes = None
    if anchor_photo:
        photo_bytes = await anchor_photo.read()
        input_data["photo_analysis"] = analyze_photo(llm, photo_bytes)
    elif not input_data.get("photo_analysis"):
        return JSONResponse(status_code=400, content={"error": "No photo provided"})

    keywords = assemble_keywords(llm, input_data)
    prompt = expand_prompt(llm, keywords)
    validate_prompt(prompt, input_data)
    negative_prompt = build_negative_prompt()

    # Step 2: Build param combos
    def _parse_floats(s: str) -> list[float]:
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    gs_list = _parse_floats(guidance_scales)
    cfg_list = _parse_floats(cfg_rescale_factors)
    tw_list = _parse_floats(edit_text_weights)
    iw_list = _parse_floats(edit_image_weights)

    images_bytes = [photo_bytes] if photo_bytes else None
    base_kwargs = {}

    if use_refs:
        refs = pick_references(ref_count)
        images_bytes = _inject_references(images_bytes, refs)
        base_kwargs.update(_build_ref_kwargs(prompt, refs))

    has_images = bool(images_bytes)
    combos = []
    for gs, cfg, tw, iw in itertools.product(gs_list, cfg_list, tw_list, iw_list):
        params = {"guidance_scale": gs, "cfg_rescale_factor": cfg}
        if has_images:
            params["single_edit_guidance_weight"] = tw
            params["single_edit_guidance_weight_image"] = iw
        combos.append(params)

    def _call(params: dict):
        seed = randint(0, 2**31)
        result = client.generate(
            prompt=prompt,
            images=images_bytes,
            width=2048,
            height=2048,
            seed=seed,
            negative_prompt=negative_prompt,
            force_single=True,
            **base_kwargs,
            **params,
        )
        return seed, params, result

    results = []
    errors = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with ThreadPoolExecutor(max_workers=min(len(combos), 10)) as pool:
        future_map = {pool.submit(_call, p): p for p in combos}
        for future in as_completed(future_map):
            params = future_map[future]
            try:
                seed, params, result = future.result()
                for i, img in enumerate(result.images):
                    path = os.path.join(OUTPUT_DIR, f"{timestamp}_sweep_s{seed}_{i}.jpg")
                    with open(path, "wb") as f:
                        f.write(img)
                results.append({
                    "seed": seed,
                    "params": params,
                    "_raw_images": result.images,
                    "images": [
                        f"data:image/jpeg;base64,{base64.b64encode(img).decode()}"
                        for img in result.images
                    ],
                })
            except SeedreamAPIError as e:
                logger.error("Sweep seed=%d error: %s", seed, e)
                errors.append({"params": params, "error": str(e)})
            except Exception as e:
                logger.error("Sweep seed=%d error: %s", seed, e)
                errors.append({"params": params, "error": str(e)})

    # Step 3: Eval
    eval_c = EvalClient()
    good_refs, bad_refs = pick_eval_references()

    for r in results:
        raw = r.pop("_raw_images", [])
        if raw:
            try:
                ev = eval_c.evaluate(
                    generated_image=raw[0],
                    input_data=input_data,
                    good_refs=good_refs,
                    bad_refs=bad_refs,
                )
                r["eval"] = {
                    "score": ev.total_score,
                    "passed": ev.passed,
                    "dimensions": ev.dimensions,
                    "issues": ev.issues,
                }
            except Exception as e:
                r["eval"] = {"score": 0, "passed": False, "dimensions": {}, "issues": [str(e)]}

    results.sort(key=lambda r: r.get("eval", {}).get("score", 0), reverse=True)

    return {
        "total": len(combos),
        "success": len(results),
        "failed": len(errors),
        "prompt": prompt,
        "results": results,
        "errors": errors,
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")
