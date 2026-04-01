# run_orchestrator.py
"""CLI script to run the badge generation pipeline."""

import json
import logging
import os
import sys

import openai
from dotenv import load_dotenv

from eval_client import EvalClient
from generator import Generator
from orchestrator import Orchestrator
from seedream_sdk import SeedreamClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

SAMPLE_INPUT = {
    "slogan": "Wells",
    "community_type": "host",
    "slogan_lang": "English",
    "anchor_info": {
        "anchor": {
            "nick_name": "Wells",
            "bio_description": "Just chatting streamer",
        },
        "anchor_characterization": (
            "This Jordanian anchor hosts an energetic and highly interactive "
            "'Just Chatting' stream. He engages directly with his audience. "
            "Visually, he presents a casual style, frequently seen in a "
            "signature blue hoodie."
        ),
        "brand_palette": {
            "primary": {"name": "Gold", "hex": "#D4AF37"},
            "secondary": {"name": "Vibrant Blue", "hex": "#007BFF"},
            "tertiary": {"name": "Black", "hex": "#000000"},
        },
    },
}


def main():
    input_data = SAMPLE_INPUT
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            input_data = json.load(f)

    llm_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    orch = Orchestrator(
        llm_client=llm_client,
        generator=Generator(SeedreamClient()),
        eval_client=EvalClient(),
    )

    # Load photo if referenced in input
    photo_bytes = None
    photo_path = input_data.get("anchor_photo", "")
    if photo_path and os.path.isfile(photo_path):
        with open(photo_path, "rb") as f:
            photo_bytes = f.read()

    count = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    result = orch.run_batch(input_data, photo_bytes=photo_bytes, count=count)

    print(f"\n{'='*60}")
    print(f"Total: {result.total} | Success: {result.success} | Failed: {result.failed}")
    print(f"Prompt: {result.prompt[:150]}...")

    for i, r in enumerate(result.results):
        print(f"\n--- Result {i+1} ---")
        print(f"Score: {r.score:.1f} | Passed: {r.passed} | Rounds: {r.rounds} | Seed: {r.seed}")
        if r.eval_result.dimensions:
            print(f"Dimensions: {r.eval_result.dimensions}")
        if r.eval_result.issues:
            print(f"Issues: {r.eval_result.issues}")

        if r.image:
            out_path = f"output/cli_result_{i}_{r.seed}.jpg"
            os.makedirs("output", exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(r.image)
            print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
