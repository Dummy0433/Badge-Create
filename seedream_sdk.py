# seedream_sdk.py
"""Seedream 4.5 image generation SDK wrapping the HTTP multipart API."""

import base64
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
        cot_mode: str = "enable",
        **kwargs,
    ) -> SeedreamResponse:
        conf = {
            "prompt": prompt,
            "model_version": model_version,
            "pre_vlm_version": pre_vlm_version,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "seed": seed,
            "force_single": force_single,
            "cot_mode": cot_mode,
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
        body = resp.json()

        status_code = body.get("status_code", -1)
        if status_code != 0:
            raise SeedreamAPIError(
                status_code=body.get("algo_status_code", status_code),
                message=body.get("algo_status_message", body.get("message", "Unknown error")),
                raw_response=body,
            )

        # Extract images and metadata from afr_data
        # Each afr_data item is a dict with:
        #   - "pic": base64-encoded JPEG image
        #   - "pic_conf": JSON string with llm_result, request_id, seed, etc.
        raw_images = []
        llm_result = ""
        image_prompts = []
        request_id = ""

        afr_data = body.get("data", {}).get("afr_data", [])
        for item in afr_data:
            if isinstance(item, dict):
                # Extract image from "pic" field
                pic_b64 = item.get("pic", "")
                if pic_b64:
                    raw_images.append(base64.b64decode(pic_b64))

                # Extract metadata from "pic_conf" field
                pic_conf_str = item.get("pic_conf", "")
                if pic_conf_str:
                    try:
                        pic_conf = json.loads(pic_conf_str)
                        if not llm_result:
                            llm_result = pic_conf.get("llm_result", "")
                        if not request_id:
                            request_id = pic_conf.get("request_id", "")
                    except json.JSONDecodeError:
                        pass
            elif isinstance(item, str):
                # Fallback: treat as raw base64 image
                raw_images.append(base64.b64decode(item))

        # Fallback request_id from extra.log_id
        if not request_id:
            request_id = body.get("extra", {}).get("log_id", "")

        return SeedreamResponse(
            images=raw_images,
            llm_result=llm_result,
            image_prompts=image_prompts,
            request_id=request_id,
            raw_response=body,
        )
