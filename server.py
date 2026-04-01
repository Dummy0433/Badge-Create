# server.py
"""FastAPI server exposing Seedream 4.5 SDK as a REST API."""

import base64
import logging
import os
from datetime import datetime

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from seedream_sdk import SeedreamClient, SeedreamAPIError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Seedream 4.5 Test UI")

client = SeedreamClient()

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.post("/api/generate")
async def generate(
    prompt: str = Form(...),
    width: int = Form(2048),
    height: int = Form(2048),
    seed: int = Form(-1),
    negative_prompt: str = Form("nsfw"),
    force_single: bool = Form(True),
    files: list[UploadFile] | None = File(None),
):
    images_bytes = None
    if files:
        images_bytes = [await f.read() for f in files]

    try:
        result = client.generate(
            prompt=prompt,
            images=images_bytes,
            width=width,
            height=height,
            seed=seed,
            negative_prompt=negative_prompt,
            force_single=force_single,
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


app.mount("/", StaticFiles(directory="static", html=True), name="static")
