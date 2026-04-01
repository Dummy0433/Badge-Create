# Seedream 4.5 Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python SDK + FastAPI Web UI to call Seedream 4.5 image generation API for manual testing.

**Architecture:** Single-layer project — `seedream_sdk.py` (SDK class), `server.py` (FastAPI), `static/index.html` (frontend). SDK wraps the HTTP multipart API at `api2.musical.ly/media/api/pic/afr`, server exposes it as a REST endpoint, frontend provides a simple test UI.

**Tech Stack:** Python 3.11+, requests, FastAPI, uvicorn, pytest

**Spec:** `docs/superpowers/specs/2026-04-01-seedream-client-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `seedream_sdk.py` | `SeedreamResponse` dataclass + `SeedreamClient` class wrapping HTTP API |
| `server.py` | FastAPI app with `/api/generate` endpoint, serves static files |
| `static/index.html` | Single-page test UI with prompt input, image upload, result display |
| `requirements.txt` | Project dependencies |
| `tests/test_seedream_sdk.py` | SDK unit tests with mocked HTTP |
| `tests/test_server.py` | Server endpoint tests with mocked SDK |

---

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `tests/__init__.py`

- [ ] **Step 1: Initialize git repo and create requirements.txt**

```bash
cd /Users/bytedance/Documents/Projects/Badge_Create
git init
```

```
# requirements.txt
requests>=2.31.0
fastapi>=0.110.0
uvicorn>=0.29.0
python-multipart>=0.0.9
pytest>=8.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
```

- [ ] **Step 2: Create directory structure**

```bash
mkdir -p static output tests
touch tests/__init__.py
```

- [ ] **Step 3: Create .gitignore**

```
# .gitignore
__pycache__/
*.pyc
output/
.pytest_cache/
*.egg-info/
venv/
.env
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt tests/__init__.py .gitignore
git commit -m "chore: project setup with dependencies and directory structure"
```

---

### Task 2: SeedreamClient SDK — Tests

**Files:**
- Create: `tests/test_seedream_sdk.py`

- [ ] **Step 1: Write test for T2I (text-to-image, no reference images)**

```python
# tests/test_seedream_sdk.py
import json
from unittest.mock import patch, MagicMock
import pytest
from seedream_sdk import SeedreamClient, SeedreamResponse, SeedreamAPIError


def _mock_success_response(image_bytes=b"fake-jpeg-data", llm_result="", request_id="test-123"):
    """Create a mock requests.Response for a successful Seedream API call."""
    resp = MagicMock()
    resp.status_code = 200
    resp_body = {
        "status_code": 0,
        "data": {
            "afr_data": [],
            "resp_json": json.dumps({
                "llm_result": llm_result,
                "image_prompt": [],
                "request_id": request_id,
            }),
            "binary_data": [],
        },
        "extra": {"log_id": "log-abc"},
    }
    resp.json.return_value = resp_body
    resp.content = image_bytes
    resp.headers = {"Content-Type": "application/json"}
    return resp


def _mock_error_response(algo_code=100199, algo_msg="algoError"):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "status_code": 1,
        "algo_status_code": algo_code,
        "algo_status_message": algo_msg,
        "data": {"afr_data": []},
        "message": "Couldn't process image.",
    }
    return resp


class TestSeedreamClientBuildRequest:
    """Test that the client builds correct HTTP requests."""

    @patch("seedream_sdk.requests.post")
    def test_t2i_request_format(self, mock_post):
        mock_post.return_value = _mock_success_response()
        client = SeedreamClient()
        client.generate(prompt="a cute cat", width=1024, height=1024)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args

        # Check endpoint
        assert call_kwargs.args[0] == "https://api2.musical.ly/media/api/pic/afr"

        # Check form data contains algorithms and conf
        data = call_kwargs.kwargs.get("data", {})
        assert data["algorithms"] == "tt_vlm_high_aes_scheduler"

        conf = json.loads(data["conf"])
        assert conf["prompt"] == "a cute cat"
        assert conf["width"] == 1024
        assert conf["height"] == 1024
        assert conf["model_version"] == "general_v4.5"
        assert conf["pre_vlm_version"] == "tt_seed_x2i_40l_pe_20b_T2_18"
        assert conf["force_single"] is True

    @patch("seedream_sdk.requests.post")
    def test_i2i_request_includes_files(self, mock_post):
        mock_post.return_value = _mock_success_response()
        client = SeedreamClient()
        fake_image = b"\xff\xd8\xff\xe0fake-jpeg"
        client.generate(prompt="edit this", images=[fake_image])

        call_kwargs = mock_post.call_args
        data = call_kwargs.kwargs.get("data", {})
        assert data["input_img_type"] == "multiple_files"

        files = call_kwargs.kwargs.get("files", [])
        assert len(files) == 1
        assert files[0][0] == "files[]"

    @patch("seedream_sdk.requests.post")
    def test_kwargs_passthrough(self, mock_post):
        mock_post.return_value = _mock_success_response()
        client = SeedreamClient()
        client.generate(prompt="test", guidance_scale=5.0, num_inference_timesteps=20)

        call_kwargs = mock_post.call_args
        data = call_kwargs.kwargs.get("data", {})
        conf = json.loads(data["conf"])
        assert conf["guidance_scale"] == 5.0
        assert conf["num_inference_timesteps"] == 20


class TestSeedreamClientErrors:

    @patch("seedream_sdk.requests.post")
    def test_algo_error_raises(self, mock_post):
        mock_post.return_value = _mock_error_response(100199, "JSON parse error")
        client = SeedreamClient()

        with pytest.raises(SeedreamAPIError) as exc_info:
            client.generate(prompt="bad request")
        assert "100199" in str(exc_info.value)

    @patch("seedream_sdk.requests.post")
    def test_http_error_raises(self, mock_post):
        resp = MagicMock()
        resp.status_code = 500
        resp.raise_for_status.side_effect = Exception("Internal Server Error")
        mock_post.return_value = resp
        client = SeedreamClient()

        with pytest.raises(Exception):
            client.generate(prompt="test")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_seedream_sdk.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'seedream_sdk'`

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_seedream_sdk.py
git commit -m "test: add SDK unit tests for request building and error handling"
```

---

### Task 3: SeedreamClient SDK — Implementation

**Files:**
- Create: `seedream_sdk.py`

- [ ] **Step 1: Implement seedream_sdk.py**

```python
# seedream_sdk.py
"""Seedream 4.5 image generation SDK wrapping the HTTP multipart API."""

import json
import logging
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)


class SeedreamAPIError(Exception):
    """Raised when Seedream API returns a non-zero status code."""

    def __init__(self, status_code: int, message: str, raw_response: dict):
        self.status_code = status_code
        self.message = message
        self.raw_response = raw_response
        super().__init__(f"Seedream API error {status_code}: {message}")


@dataclass
class SeedreamResponse:
    """Parsed response from Seedream API."""

    images: list[bytes] = field(default_factory=list)
    llm_result: str = ""
    image_prompts: list[str] = field(default_factory=list)
    request_id: str = ""
    raw_response: dict = field(default_factory=dict)


class SeedreamClient:
    """Client for calling Seedream 4.5 image generation via HTTP API."""

    def __init__(
        self,
        endpoint: str = "https://api2.musical.ly/media/api/pic/afr",
        req_key: str = "tt_vlm_high_aes_scheduler",
        timeout: int = 120,
    ):
        self.endpoint = endpoint
        self.req_key = req_key
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        images: list[bytes] | None = None,
        width: int = 2048,
        height: int = 2048,
        model_version: str = "general_v4.5",
        pre_vlm_version: str = "tt_seed_x2i_40l_pe_20b_T2_18",
        negative_prompt: str = "nsfw",
        seed: int = -1,
        force_single: bool = True,
        **kwargs,
    ) -> SeedreamResponse:
        """Generate image(s) using Seedream 4.5.

        Args:
            prompt: Text prompt for image generation.
            images: Optional list of reference image bytes for I2I/edit.
            width: Output image width (default 2048).
            height: Output image height (default 2048).
            model_version: Model version string.
            pre_vlm_version: PE version string (required for compliance).
            negative_prompt: Negative prompt.
            seed: Random seed (-1 for random).
            force_single: Force single image output.
            **kwargs: Additional parameters passed directly to the API conf.

        Returns:
            SeedreamResponse with generated images and metadata.

        Raises:
            SeedreamAPIError: If the API returns a non-zero status code.
            requests.RequestException: If the HTTP request fails.
        """
        conf = {
            "prompt": prompt,
            "model_version": model_version,
            "pre_vlm_version": pre_vlm_version,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "seed": seed,
            "force_single": force_single,
            **kwargs,
        }

        data = {
            "algorithms": self.req_key,
            "conf": json.dumps(conf),
        }

        files = None
        if images:
            data["input_img_type"] = "multiple_files"
            files = [("files[]", (f"image_{i}.jpg", img, "image/jpeg")) for i, img in enumerate(images)]

        logger.info("Calling Seedream API: prompt=%r, images=%d", prompt, len(images) if images else 0)

        resp = requests.post(
            self.endpoint,
            data=data,
            files=files,
            timeout=self.timeout,
        )
        resp.raise_for_status()

        return self._parse_response(resp)

    def _parse_response(self, resp: requests.Response) -> SeedreamResponse:
        """Parse the HTTP response from Seedream API.

        The HTTP API wraps the response in a JSON envelope. Image binary data
        may be in data.afr_data or returned as the response body directly.
        This method handles both cases and extracts available metadata.
        """
        body = resp.json()

        status_code = body.get("status_code", -1)
        if status_code != 0:
            raise SeedreamAPIError(
                status_code=body.get("algo_status_code", status_code),
                message=body.get("algo_status_message", body.get("message", "Unknown error")),
                raw_response=body,
            )

        # Extract metadata from resp_json if available
        resp_json_str = body.get("data", {}).get("resp_json", "{}")
        try:
            resp_json = json.loads(resp_json_str) if isinstance(resp_json_str, str) else resp_json_str
        except json.JSONDecodeError:
            resp_json = {}

        # Extract images from afr_data
        import base64
        raw_images = []
        afr_data = body.get("data", {}).get("afr_data", [])
        for item in afr_data:
            if isinstance(item, str):
                raw_images.append(base64.b64decode(item))
            elif isinstance(item, dict) and "binary" in item:
                raw_images.append(base64.b64decode(item["binary"]))

        # Also check binary_data field
        binary_data = body.get("data", {}).get("binary_data", [])
        for item in binary_data:
            if isinstance(item, str):
                raw_images.append(base64.b64decode(item))

        return SeedreamResponse(
            images=raw_images,
            llm_result=resp_json.get("llm_result", ""),
            image_prompts=resp_json.get("image_prompt", []),
            request_id=resp_json.get("request_id", body.get("extra", {}).get("log_id", "")),
            raw_response=body,
        )
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
pytest tests/test_seedream_sdk.py -v
```

Expected: All tests PASS. Note: `_parse_response` tests rely on mock return values. The actual response parsing may need adjustment after a real successful API call — this is expected and noted in the spec.

- [ ] **Step 3: Commit**

```bash
git add seedream_sdk.py
git commit -m "feat: implement SeedreamClient SDK wrapping HTTP API"
```

---

### Task 4: FastAPI Server — Tests

**Files:**
- Create: `tests/test_server.py`

- [ ] **Step 1: Write server endpoint tests**

```python
# tests/test_server.py
import json
from unittest.mock import patch, MagicMock
import pytest
from httpx import AsyncClient, ASGITransport
from server import app
from seedream_sdk import SeedreamResponse


@pytest.fixture
def mock_sdk_response():
    return SeedreamResponse(
        images=[b"\xff\xd8\xff\xe0fake-jpeg-bytes"],
        llm_result="A cute orange cat sitting on a windowsill",
        image_prompts=[],
        request_id="test-req-001",
        raw_response={"status_code": 0},
    )


@pytest.mark.asyncio
async def test_generate_t2i_success(mock_sdk_response):
    with patch("server.client") as mock_client:
        mock_client.generate.return_value = mock_sdk_response

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/generate",
                data={"prompt": "a cute cat", "width": "1024", "height": "1024"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["request_id"] == "test-req-001"
        assert body["llm_result"] == "A cute orange cat sitting on a windowsill"
        assert len(body["images"]) == 1
        assert body["images"][0].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_generate_with_files(mock_sdk_response):
    with patch("server.client") as mock_client:
        mock_client.generate.return_value = mock_sdk_response

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/generate",
                data={"prompt": "edit this image"},
                files=[("files", ("test.jpg", b"\xff\xd8fake", "image/jpeg"))],
            )

        assert resp.status_code == 200
        mock_client.generate.assert_called_once()
        call_kwargs = mock_client.generate.call_args.kwargs
        assert call_kwargs["images"] is not None
        assert len(call_kwargs["images"]) == 1


@pytest.mark.asyncio
async def test_generate_missing_prompt():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/api/generate", data={})

    assert resp.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_server.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'server'`

- [ ] **Step 3: Commit**

```bash
git add tests/test_server.py
git commit -m "test: add FastAPI server endpoint tests"
```

---

### Task 5: FastAPI Server — Implementation

**Files:**
- Create: `server.py`

- [ ] **Step 1: Implement server.py**

```python
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
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
pytest tests/test_server.py -v
```

Expected: All tests PASS.

- [ ] **Step 3: Also run SDK tests to make sure nothing broke**

```bash
pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add server.py
git commit -m "feat: implement FastAPI server with /api/generate endpoint"
```

---

### Task 6: Frontend UI

**Files:**
- Create: `static/index.html`

- [ ] **Step 1: Implement the frontend page**

```html
<!-- static/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Seedream 4.5 Test UI</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; color: #333; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { margin-bottom: 20px; font-size: 24px; }

        .form-section { background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .form-section h2 { font-size: 16px; margin-bottom: 12px; color: #666; }

        label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 4px; color: #555; }
        textarea { width: 100%; height: 100px; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; resize: vertical; font-family: inherit; }
        input[type="number"], input[type="text"] { width: 100%; padding: 8px 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; }
        input[type="file"] { font-size: 14px; }

        .params-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 8px; }
        .param-group { display: flex; flex-direction: column; }

        .checkbox-group { display: flex; align-items: center; gap: 6px; margin-top: 20px; }
        .checkbox-group input { width: auto; }

        .btn { display: inline-block; padding: 12px 32px; background: #2563eb; color: #fff; border: none; border-radius: 6px; font-size: 15px; font-weight: 600; cursor: pointer; margin-top: 16px; }
        .btn:hover { background: #1d4ed8; }
        .btn:disabled { background: #94a3b8; cursor: not-allowed; }

        .result-section { background: #fff; border-radius: 8px; padding: 20px; margin-top: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); display: none; }
        .result-section.visible { display: block; }
        .result-meta { font-size: 13px; color: #666; margin-bottom: 12px; word-break: break-all; }
        .result-meta strong { color: #333; }
        .result-images { display: flex; flex-wrap: wrap; gap: 12px; }
        .result-images img { max-width: 100%; border-radius: 6px; border: 1px solid #eee; }

        .llm-result { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; margin-top: 12px; font-size: 13px; white-space: pre-wrap; max-height: 200px; overflow-y: auto; }

        .loading { text-align: center; padding: 40px; font-size: 16px; color: #666; }
        .loading::after { content: ''; display: inline-block; width: 20px; height: 20px; border: 3px solid #ddd; border-top-color: #2563eb; border-radius: 50%; animation: spin 0.8s linear infinite; margin-left: 10px; vertical-align: middle; }
        @keyframes spin { to { transform: rotate(360deg); } }

        .error { background: #fef2f2; border: 1px solid #fecaca; color: #dc2626; border-radius: 6px; padding: 12px; margin-top: 12px; font-size: 13px; white-space: pre-wrap; }

        .raw-toggle { font-size: 12px; color: #2563eb; cursor: pointer; margin-top: 8px; display: inline-block; }
        .raw-content { display: none; background: #f1f5f9; border-radius: 6px; padding: 12px; margin-top: 8px; font-size: 12px; font-family: monospace; white-space: pre-wrap; max-height: 300px; overflow-y: auto; }
        .raw-content.visible { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Seedream 4.5 Test UI</h1>

        <div class="form-section">
            <h2>Prompt</h2>
            <textarea id="prompt" placeholder="Enter your prompt here..."></textarea>
        </div>

        <div class="form-section">
            <h2>Parameters</h2>
            <div class="params-grid">
                <div class="param-group">
                    <label for="width">Width</label>
                    <input type="number" id="width" value="2048" min="256" max="4096">
                </div>
                <div class="param-group">
                    <label for="height">Height</label>
                    <input type="number" id="height" value="2048" min="256" max="4096">
                </div>
                <div class="param-group">
                    <label for="seed">Seed (-1 = random)</label>
                    <input type="number" id="seed" value="-1">
                </div>
            </div>
            <div class="params-grid" style="margin-top: 12px;">
                <div class="param-group" style="grid-column: span 2;">
                    <label for="negative_prompt">Negative Prompt</label>
                    <input type="text" id="negative_prompt" value="nsfw">
                </div>
                <div class="param-group">
                    <div class="checkbox-group">
                        <input type="checkbox" id="force_single" checked>
                        <label for="force_single" style="margin: 0;">Force Single</label>
                    </div>
                </div>
            </div>
        </div>

        <div class="form-section">
            <h2>Reference Images (optional)</h2>
            <input type="file" id="ref_images" multiple accept="image/*">
        </div>

        <button class="btn" id="generateBtn" onclick="generate()">Generate</button>

        <div class="result-section" id="resultSection">
            <h2>Result</h2>
            <div id="resultContent"></div>
        </div>
    </div>

    <script>
        async function generate() {
            const btn = document.getElementById('generateBtn');
            const section = document.getElementById('resultSection');
            const content = document.getElementById('resultContent');
            const prompt = document.getElementById('prompt').value.trim();

            if (!prompt) {
                alert('Please enter a prompt');
                return;
            }

            btn.disabled = true;
            section.classList.add('visible');
            content.innerHTML = '<div class="loading">Generating...</div>';

            const formData = new FormData();
            formData.append('prompt', prompt);
            formData.append('width', document.getElementById('width').value);
            formData.append('height', document.getElementById('height').value);
            formData.append('seed', document.getElementById('seed').value);
            formData.append('negative_prompt', document.getElementById('negative_prompt').value);
            formData.append('force_single', document.getElementById('force_single').checked);

            const fileInput = document.getElementById('ref_images');
            for (const file of fileInput.files) {
                formData.append('files', file);
            }

            try {
                const resp = await fetch('/api/generate', { method: 'POST', body: formData });
                const data = await resp.json();

                if (!resp.ok) {
                    content.innerHTML = `<div class="error">${data.error || 'Unknown error'}\n\n${JSON.stringify(data.raw_response || {}, null, 2)}</div>`;
                    return;
                }

                let html = '';
                html += `<div class="result-meta"><strong>Request ID:</strong> ${data.request_id}</div>`;

                if (data.images && data.images.length > 0) {
                    html += '<div class="result-images">';
                    for (const img of data.images) {
                        html += `<img src="${img}" alt="Generated image">`;
                    }
                    html += '</div>';
                } else {
                    html += '<div class="error">No images returned</div>';
                }

                if (data.llm_result) {
                    html += `<div class="result-meta" style="margin-top:12px;"><strong>PE Rewrite:</strong></div>`;
                    html += `<div class="llm-result">${data.llm_result}</div>`;
                }

                if (data.image_prompts && data.image_prompts.length > 0) {
                    html += `<div class="result-meta" style="margin-top:12px;"><strong>Image Prompts:</strong></div>`;
                    html += `<div class="llm-result">${data.image_prompts.join('\n\n')}</div>`;
                }

                html += `<span class="raw-toggle" onclick="toggleRaw()">Show raw response</span>`;
                html += `<div class="raw-content" id="rawContent">${JSON.stringify(data.raw_response || {}, null, 2)}</div>`;

                content.innerHTML = html;
            } catch (e) {
                content.innerHTML = `<div class="error">Request failed: ${e.message}</div>`;
            } finally {
                btn.disabled = false;
            }
        }

        function toggleRaw() {
            document.getElementById('rawContent').classList.toggle('visible');
        }
    </script>
</body>
</html>
```

- [ ] **Step 2: Start server and verify page loads**

```bash
uvicorn server:app --reload --port 8000
```

Open `http://localhost:8000` — verify the page renders correctly with all form elements.

- [ ] **Step 3: Commit**

```bash
git add static/index.html
git commit -m "feat: add frontend test UI for Seedream generation"
```

---

### Task 7: End-to-End Smoke Test

- [ ] **Step 1: Run all unit tests**

```bash
pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 2: Start server and test a real T2I call**

```bash
uvicorn server:app --reload --port 8000
```

Open `http://localhost:8000`, enter a simple prompt (e.g. "a cute cat"), set width/height to 1024, click Generate. Verify:
- Request goes through without HTTP error
- If API returns an error, the error message is displayed clearly with the raw response
- If successful, the generated image is displayed and saved in `output/`

This step validates the actual API response format. If the image extraction path (`afr_data` / `binary_data`) needs adjustment, update `seedream_sdk.py:_parse_response` accordingly.

- [ ] **Step 3: Commit any response parsing fixes**

```bash
git add seedream_sdk.py
git commit -m "fix: adjust response parsing based on live API response"
```
