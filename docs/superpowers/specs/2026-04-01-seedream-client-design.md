# Seedream 4.5 Client - Design Spec

## Overview

A Python SDK + FastAPI Web UI for calling ByteDance internal Seedream 4.5 image generation API. This is the first step of a larger orchestration pipeline (LLM analysis → keyword extraction → Seedream generation → quality evaluation → retry loop). The current scope is limited to manual prompt + image upload testing to explore Seedream's capabilities and boundaries.

## Architecture

```
project/
├── seedream_sdk.py      # SDK class wrapping Seedream HTTP API
├── server.py            # FastAPI server exposing SDK as REST API
├── static/
│   └── index.html       # Single-page test UI
├── output/              # Generated images saved locally
└── requirements.txt
```

Single-layer structure. SDK is an importable class; FastAPI wraps it as HTTP API; frontend is a single HTML file served statically.

## Component 1: SeedreamClient (seedream_sdk.py)

### Class Interface

```python
@dataclass
class SeedreamResponse:
    images: list[bytes]           # Generated image bytes (JPEG)
    llm_result: str               # PE rewrite result
    image_prompts: list[str]      # Per-image prompts (multi-image output)
    request_id: str
    raw_response: dict            # Full response for debugging

class SeedreamClient:
    def __init__(self, req_key="tt_vlm_high_aes_scheduler"):
        self.endpoint = "https://api2.musical.ly/media/api/pic/afr"
        self.req_key = req_key

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
        ...
```

### HTTP Request Format

Based on the confirmed curl API:

```
POST https://api2.musical.ly/media/api/pic/afr
Content-Type: multipart/form-data

Fields:
  - algorithms: req_key string
  - conf: JSON string of req_json parameters
  - input_img_type: "multiple_files" (when images provided)
  - files[]: image file(s)
```

### HTTP Response Format

```json
{
  "status_code": 0,
  "data": {
    "afr_data": [...]
  },
  "extra": {
    "log_id": "..."
  }
}
```

Exact binary extraction path TBD — needs a successful call to confirm. The SDK will parse both the documented resp_json fields and the HTTP-specific wrapper.

### Authentication

None required. Internal network (VPN) access is sufficient. Confirmed via live curl test returning algorithm-level error (not 401/403).

### Key Defaults

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| width/height | 2048 | Doc recommends 2k for quality |
| model_version | general_v4.5 | Current version |
| pre_vlm_version | tt_seed_x2i_40l_pe_20b_T2_18 | Required, US-TTP compliant |
| negative_prompt | nsfw | Doc default |
| force_single | True | Testing single image output first |
| seed | -1 | Random |

### Error Handling

- HTTP errors: raise with status code and message
- Algorithm errors (status_code != 0): raise with algo_status_code and algo_status_message
- Timeout: configurable, default 120s (image generation can be slow)

## Component 2: FastAPI Server (server.py)

### Endpoint

```
POST /api/generate

Form fields:
  - prompt: str (required)
  - width: int = 2048
  - height: int = 2048
  - seed: int = -1
  - negative_prompt: str = "nsfw"
  - force_single: bool = true
  - files: list[UploadFile] (optional, reference images)

Response (JSON):
  {
    "request_id": "...",
    "llm_result": "...",
    "image_prompts": [...],
    "images": ["data:image/jpeg;base64,..."]
  }
```

- Images returned as base64 data URIs for direct frontend rendering
- Generated images also saved to `output/` directory with timestamp filenames
- Static files served from `static/` directory

## Component 3: Frontend (static/index.html)

Single HTML file with inline CSS/JS. Features:

- **Prompt input**: textarea for main prompt
- **Parameter controls**: width, height, seed, negative_prompt, force_single
- **Image upload**: file input supporting multiple reference images
- **Generate button**: calls POST /api/generate
- **Results area**: displays generated image(s) + PE rewrite result + request_id
- **Loading state**: spinner/indicator during generation (can take 10-30s)

No framework dependencies. Vanilla HTML/CSS/JS.

## Future Considerations (Out of Scope)

These are noted for the broader orchestration pipeline but NOT implemented now:

- LLM-based image analysis and keyword extraction
- Keyword injection into prompts
- Quality evaluation loop with automatic retry
- RPC transport (gateway_vproxy) for production
- URL-based image input
- Batch generation
