# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Seedream 4.5 image generation testing tool — Python SDK + FastAPI server + vanilla HTML UI for calling ByteDance internal Seedream 4.5 API. This is a stepping stone toward a full orchestration pipeline (LLM analysis -> keyword extraction -> Seedream generation -> quality evaluation -> retry loop).

## Commands

```bash
pip install -r requirements.txt                # Install dependencies
uvicorn server:app --reload --port 8000        # Start dev server (UI at http://localhost:8000)
pytest tests/ -v                               # Run all tests (32 tests)
pytest tests/test_seedream_sdk.py -v           # SDK tests only
pytest tests/test_server.py -v                 # Server tests only
pytest tests/path::TestClass::test_name -v     # Single test
python3 run_orchestrator.py                    # Run orchestration pipeline with sample input
python3 run_orchestrator.py input.json         # Run with custom input JSON
```

## Product Features

Three modes share ONE pipeline: `JSON + Photo → LLM prompt build → Seedream generate → GPT eval`

| Mode | What it does | Use case |
|------|-------------|----------|
| **Generate** | 1 image, fixed params (gs=8, cfg=0, tw=2, iw=1) + eval | Quick single test |
| **Batch** | N images (default 10), fixed params, different seeds + eval all, sort by score | Production: pick best from N |
| **Sweep** | Cartesian product of param arrays (guidance_scale, cfg, text/image weight), each with random seed + eval | Find optimal "magic number" parameters |

All three:
1. Accept the same input: datamining JSON + anchor photo
2. Auto build prompt: GPT-5.4 photo analysis → keyword assembly → prompt expansion
3. Pass anchor photo + few-shot reference images to Seedream
4. Eval every generated image with GPT-5.4 (7 dimensions, weighted scoring)
5. Return results sorted by eval score

## Architecture

```
Input (JSON + Photo)
    ↓
prompt_builder.py  →  GPT-5.4: analyze_photo → assemble_keywords → expand_prompt → validate
    ↓
server.py pipeline  →  Seedream SDK: batch generate N images (photo + fewshot refs as input)
    ↓
eval_client.py  →  GPT-5.4: score each image on 7 dimensions → sort by score
    ↓
Frontend: display ranked results with eval scores
```

### Modules

- **`seedream_sdk.py`** — `SeedreamClient` wraps Seedream HTTP multipart API. Optimal params: gs=8, cfg=0, tw=2, iw=1, cot_mode="enable".
- **`prompt_builder.py`** — 3-step LLM prompt pipeline: `analyze_photo()` (GPT vision → extract appearance), `assemble_keywords()` (map JSON fields → badge elements, smart color assignment), `expand_prompt()` (keywords → final Seedream prompt with fixed structure constraints). Template fallback for tests.
- **`eval_client.py`** — GPT-5.4 vision eval. 7 dimensions (heart_carrier, character, decorations, text_render, color_match, composition, quality). Weighted scoring: text_render=0.5 weight (diffusion models struggle with text), others=1.0. Pass threshold: weighted avg >= 8.0.
- **`eval_store.py`** — Good/bad reference images for eval (3 good, 3 bad, hardcoded → future DB). Objective descriptions: structure proportions, subject occupancy %, text position, color saturation, quality.
- **`reference_store.py`** — Few-shot style reference images injected into Seedream for style guidance (3 refs, hardcoded → future DB).
- **`orchestrator.py`** — `preprocess_input()` maps datamining nested format to internal flat format. Also contains CLI retry/reroll logic (legacy, not used by server pipeline).
- **`server.py`** — FastAPI endpoints: `/api/pipeline` (main: JSON+photo → prompt → batch → eval → ranked), `/api/generate` (single), `/api/generate_batch` (batch), `/api/generate_sweep` (param sweep), `/api/build_prompt` (prompt only).
- **`static/index.html`** — Single-page vanilla HTML/CSS/JS. JSON input + photo upload + Generate button → results grid sorted by eval score.

### Input Format (datamining)

```json
{
  "slogan": "Name",
  "anchor_photo": "path or uploaded",
  "anchor_info": {
    "anchor": { "nick_name": "...", "bio_description": "..." },
    "anchor_characterization": "...",
    "brand_palette": {
      "primary": {"name": "...", "hex": "#..."},
      "secondary": {"name": "...", "hex": "#..."},
      "tertiary": {"name": "...", "hex": "#..."}
    }
  }
}
```

`preprocess_input()` flattens this to: `text_output`, `anchor_characterization`, `brand_palette`, `anchor_nickname`, `anchor_bio`.

## Seedream 4.5 API Reference

### Endpoint & Auth

- URL: `https://api2.musical.ly/media/api/pic/afr`
- req_key: `tt_vlm_high_aes_scheduler` (prod), `tt_vlm_high_aes_scheduler_mirror` (test)
- No API key needed — internal network (VPN) access only
- Timeout: 120s recommended (generation can take 10-30s)

### Request Format

HTTP multipart/form-data with fields:
- `algorithms`: req_key string
- `conf`: JSON string of parameters below
- `input_img_type`: `"multiple_files"` (when images provided)
- `files[]`: image file(s), max 14, each <=15MB

### Core Parameters (conf JSON)

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | yes | - | Generation prompt |
| `model_version` | string | yes | `"general_v4.5"` | Model version |
| `pre_vlm_version` | string | yes | `"tt_seed_x2i_40l_pe_20b_T2_18"` | VLM PE model (US-TTP compliant, cannot be empty) |
| `negative_prompt` | string | no | `"nsfw"` | Negative prompt |
| `width` | int | no* | - | Image width. Product w*h in [1M, 16M]. 2k recommended |
| `height` | int | no* | - | Image height |
| `size` | int | no* | - | Alternative to w/h — model auto-selects aspect ratio from prompt. Range [1M, 16M] |
| `seed` | int64 | no | -1 | -1=random, otherwise mapped to int32 positive |
| `force_single` | bool | no | false | Force single image output |
| `cot_mode` | string | no | `"auto"` | `"auto"` / `"enable"` / `"disable"` — chain-of-thought mode |
| `min_ratio` | float | no | 1/16 | Min width/height ratio [1/16, 16) |
| `max_ratio` | float | no | 16 | Max width/height ratio (1/16, 16] |

*size and width/height: pick one. Both missing = error. Both present = width/height wins.

### VLM Bypass Parameters

The built-in VLM (PE) is mandatory by default but can be bypassed with your own PE result:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_pre_llm` | bool | true | Set false to use own PE. Only valid if: (1) passing own PE via pre_llm_result, or (2) T2I with explicit width/height |
| `pre_llm_result` | string | `""` | JSON-dumped string of your PE result. Set this = skip built-in PE |

**pre_llm_result formats:**

T2I:
```json
{"output": "optimized prompt text", "ratio": "16:9"}
```

Edit (I2I):
```json
{"input1": "description of input image", "edit": "editing instruction", "output": "final image description", "ratio": "4:3"}
```

### Image Generation Parameters

**Tested optimal defaults** (badge generation use case):
| Parameter | Optimal | API Default | Range | Description |
|-----------|---------|-------------|-------|-------------|
| `guidance_scale` | **8.0** | 2.5 | [1.0, 10.0] | T2I text guidance strength — 8.0 produces best prompt adherence for badge style |
| `cfg_rescale_factor` | **0.0** | 0.0 | [0.0, 1.0] | T2I CFG rescale — 0.0 works best |
| `single_edit_guidance_weight` | **2.0** | 3.0 | [1.0, 10.0] | Single-image edit: text guidance — lower than default gives better results |
| `single_edit_guidance_weight_image` | **1.0** | 1.6 | [1.0, 10.0] | Single-image edit: image guidance — minimum for more creative freedom |

Other parameters (using API defaults):
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `num_inference_timesteps` | int | 25 | [4, 25] | DDIM denoising steps |
| `final_linear_steps` | int | 5 | < timesteps | Final linear steps count |
| `max_shift` | float | 1.15 | [0, 4] | Shift param (overridden by shift_list if set) |
| `multi_edit_guidance_weight` | float | 3.5 | [1.0, 10.0] | Multi-image edit: text guidance |
| `multi_edit_guidance_weight_image` | float | 1.0 | [1.0, 10.0] | Multi-image edit: image guidance |
| `single_min_cfg_time` | float | 0.9 | [0.0, 1.0] | Single edit cfg time min |
| `single_max_cfg_time` | float | 1.0 | [0.0, 1.0] | Single edit cfg time max |
| `multi_min_cfg_time` | float | 0.9 | [0.0, 1.0] | Multi edit cfg time min |
| `multi_max_cfg_time` | float | 1.0 | [0.0, 1.0] | Multi edit cfg time max |

### Safety & Rewrite Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_safety_agent` | bool | false | Safety agent — may pass, rewrite, or reject |
| `enable_qwen_rewrite` | bool | false | CN bias PE for compliance |
| `ori_prompt` | string | `""` | Original user prompt (for analytics, separate from optimized prompt) |

### Multi-image / Series Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `series_mode` | int | 0 | 0=auto (from PE tags), 1=force parallel (non-story), 2=force sequential (story) |
| `image_offset_start` | int | - | Skip first N images in series (for partial retry) |
| `image_offset_end` | int | - | Stop series generation at image N |
| `system_prompt` | string | - | Custom VLM system prompt (use with caution) |

Image count constraint: input images + output images <= 15 (DIT limit of 14 inputs).

### Response Format

```json
{
  "status_code": 0,
  "data": {
    "afr_data": [
      {
        "pic": "<base64 JPEG>",
        "pic_conf": "{\"llm_result\": \"...\", \"request_id\": \"...\", \"seed\": ...}"
      }
    ]
  },
  "extra": {"log_id": "..."}
}
```

- `status_code != 0` = error, check `algo_status_code` and `algo_status_message`
- `pic_conf.llm_result` = VLM PE rewrite result (the "thinking" + final prompt)
- `extra.log_id` = fallback request_id

## Testing Conventions

- SDK tests mock `requests.post`, server tests mock `server.client`
- Response parsing tests use realistic `afr_data[].pic` + `afr_data[].pic_conf` structure
- Server tests use httpx AsyncClient with ASGI transport
