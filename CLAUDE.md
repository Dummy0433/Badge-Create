# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Seedream 4.5 badge generation pipeline — 5 clean modules with unified retry logic, FastAPI server + vanilla HTML UI. Takes anchor/streamer data + photo, generates 3D heart-shaped badge images via Seedream 4.5, evaluates with GPT-5.4, and automatically retries on low scores.

## Commands

```bash
pip install -r requirements.txt                # Install dependencies
uvicorn server:app --reload --port 8000        # Start dev server (UI at http://localhost:8000)
pytest tests/ -v                               # Run all tests (49 tests)
pytest tests/test_orchestrator.py -v           # Orchestrator + retry tests
pytest tests/test_generator.py -v              # Generator + ref injection tests
pytest tests/test_input_processor.py -v        # Input normalization tests
pytest tests/path::TestClass::test_name -v     # Single test
python3 run_orchestrator.py                    # Run pipeline with sample input (count=1)
python3 run_orchestrator.py input.json         # Run with custom input JSON
python3 run_orchestrator.py input.json 10      # Run batch of 10
```

## Product Features

Three modes share ONE pipeline, each with automatic retry escalation:

| Mode | What it does | Use case |
|------|-------------|----------|
| **Generate** | 1 image + eval + retry if score < 8.0 | Quick single test |
| **Batch** | N images (default 10), each independently retries + eval, sort by score | Production: pick best from N |
| **Sweep** | Cartesian product of param arrays, each combination with full retry + eval | Find optimal parameters |

All three:
1. Accept the same input: datamining JSON + anchor photo
2. Auto build prompt: GPT-5.4 photo analysis → keyword assembly → prompt expansion
3. Pass anchor photo + few-shot reference images to Seedream
4. Eval every generated image with GPT-5.4 (7 dimensions, weighted scoring)
5. **Retry escalation** if eval score < 8.0:
   - Level 1: regenerate with new seed, same prompt (up to 2 retries)
   - Level 2: re-expand keywords into new prompt, then generate (1 time)
   - All fail: return best scoring result
6. Return results sorted by eval score

## Architecture

5 modules with single responsibility, unified pipeline:

```
① input_processor.py  →  ② prompt_builder.py  →  ③ generator.py  →  ④ eval_client.py
                                  ↑                                        |
                                  └──── ⑤ orchestrator.py (retry) ────────┘
```

```
server.py (thin HTTP layer)  →  orchestrator.run_batch() / run_sweep()
run_orchestrator.py (CLI)    →  orchestrator.run_batch()
```

### Modules

- **`input_processor.py`** — `process()` normalizes raw input from any source (frontend form, external API, CLI). Accepts both nested datamining format and flat internal format. Validates required fields (text_output, brand_palette).
- **`prompt_builder.py`** — LLM prompt pipeline: `analyze_photo()` (GPT vision → extract appearance), `assemble_keywords()` (map JSON fields → badge elements, smart color assignment), `validate_keywords()` (check assembled JSON), `expand_prompt()` (keywords → final Seedream prompt). `build_prompt()` returns `(keywords, prompt)` so orchestrator can re-expand from keywords on retry. Template fallback for tests.
- **`generator.py`** — `Generator` class wraps `SeedreamClient` with reference injection and PE (prompt engineering) construction. Prepends few-shot reference images before user photo, builds `pre_llm_result` JSON for style guidance.
- **`eval_client.py`** — GPT-5.4 vision eval. 7 dimensions (heart_carrier, character, decorations, text_render, color_match, composition, quality). Weighted scoring: text_render=0.5 weight (diffusion models struggle with text), others=1.0. Pass threshold: weighted avg >= 8.0.
- **`orchestrator.py`** — `Orchestrator` class: `run_batch(count=N)` runs N parallel pipeline units, each with retry escalation. `run_sweep(param_combos)` runs param combinations with full retry. `_run_single_unit()` implements the retry escalation: initial → Level 1 (new seed ×2) → Level 2 (re-expand ×1) → return best.
- **`seedream_sdk.py`** — `SeedreamClient` wraps Seedream HTTP multipart API (low-level). Optimal params: gs=8, cfg=0, tw=2, iw=1, cot_mode="enable".
- **`eval_store.py`** — Good/bad reference images for eval (3 good, 3 bad, hardcoded → future DB).
- **`reference_store.py`** — Few-shot style reference images for Seedream style guidance (3 refs, hardcoded → future DB).
- **`server.py`** — Thin FastAPI HTTP adapter. Two endpoints: `/api/pipeline` (main), `/api/pipeline_sweep` (param sweep). Delegates all logic to `orchestrator`.
- **`static/index.html`** — Single-page vanilla HTML/CSS/JS. JSON input + photo upload + Generate/Batch/Sweep buttons → results grid sorted by eval score.

### Retry Escalation (per pipeline unit)

```
generate(prompt, seed=random) → eval
  score >= 8.0 → PASS, return

  Level 1: generate(same prompt, new seed) → eval  ← up to 2 times
  Level 2: expand_prompt(keywords) → generate → eval  ← 1 time
  All fail → return best scoring result
```

Max cost per unit: 4 Seedream calls + 4 eval calls + 1 LLM expand call.

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

`input_processor.process()` flattens this to: `text_output`, `anchor_characterization`, `brand_palette`, `anchor_nickname`, `anchor_bio`, `community_type`, `slogan_lang`.

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

- 49 tests total across 8 test files
- SDK tests mock `requests.post`, generator tests mock `SeedreamClient`
- Orchestrator tests mock `orchestrator.process`, `orchestrator.build_prompt`, `orchestrator.expand_prompt`, `orchestrator.pick_eval_references`
- Server tests mock `server.orchestrator` (thin layer, test delegation not business logic)
- prompt_builder tests use `@patch("prompt_builder._get_client", return_value=None)` to force template fallback
- Response parsing tests use realistic `afr_data[].pic` + `afr_data[].pic_conf` structure
- Server tests use httpx AsyncClient with ASGI transport
