# server.py
"""FastAPI server — thin HTTP adapter for the badge generation pipeline."""

import base64
import itertools
import json
import logging
import os

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from eval_client import EvalClient
from generator import Generator
from orchestrator import Orchestrator, BatchResult
from seedream_sdk import SeedreamClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Seedream 4.5 Test UI")


# --- Initialize shared orchestrator ---
def _init_orchestrator() -> Orchestrator:
    import openai as _openai
    from dotenv import load_dotenv
    load_dotenv()
    llm_client = _openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return Orchestrator(
        llm_client=llm_client,
        generator=Generator(SeedreamClient()),
        eval_client=EvalClient(),
    )


orchestrator = _init_orchestrator()


def _format_batch_response(result: BatchResult) -> dict:
    """Convert BatchResult to JSON-serializable dict for frontend."""
    formatted_results = []
    for r in result.results:
        images_b64 = []
        if r.image:
            images_b64.append(
                f"data:image/jpeg;base64,{base64.b64encode(r.image).decode()}"
            )
        formatted_results.append({
            "seed": r.seed,
            "request_id": r.request_id,
            "images": images_b64,
            "eval": {
                "score": r.eval_result.total_score,
                "passed": r.eval_result.passed,
                "dimensions": r.eval_result.dimensions,
                "issues": r.eval_result.issues,
            },
        })

    return {
        "total": result.total,
        "success": result.success,
        "failed": result.failed,
        "prompt": result.prompt,
        "results": formatted_results,
        "errors": [],
    }


@app.post("/api/pipeline")
async def pipeline(
    input_json: str = Form(...),
    count: int = Form(10),
    use_refs: bool = Form(True),
    ref_count: int = Form(2),
    anchor_photo: UploadFile | None = File(None),
):
    """One-click pipeline: JSON+photo -> prompt -> batch generate -> eval -> ranked."""
    try:
        input_data = json.loads(input_json)
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

    photo_bytes = None
    if anchor_photo:
        photo_bytes = await anchor_photo.read()

    try:
        result = orchestrator.run_batch(
            input_data=input_data,
            photo_bytes=photo_bytes,
            count=count,
            use_refs=use_refs,
            ref_count=ref_count,
        )
    except Exception as e:
        logger.error("Pipeline error: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

    return _format_batch_response(result)


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
    """Sweep pipeline: param combinations -> each with full retry."""
    try:
        input_data = json.loads(input_json)
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

    photo_bytes = None
    if anchor_photo:
        photo_bytes = await anchor_photo.read()

    def _parse_floats(s: str) -> list[float]:
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    combos = []
    for gs, cfg, tw, iw in itertools.product(
        _parse_floats(guidance_scales),
        _parse_floats(cfg_rescale_factors),
        _parse_floats(edit_text_weights),
        _parse_floats(edit_image_weights),
    ):
        combos.append({
            "guidance_scale": gs,
            "cfg_rescale_factor": cfg,
            "single_edit_guidance_weight": tw,
            "single_edit_guidance_weight_image": iw,
        })

    try:
        result = orchestrator.run_sweep(
            input_data=input_data,
            photo_bytes=photo_bytes,
            param_combos=combos,
            use_refs=use_refs,
            ref_count=ref_count,
        )
    except Exception as e:
        logger.error("Sweep error: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

    return _format_batch_response(result)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
