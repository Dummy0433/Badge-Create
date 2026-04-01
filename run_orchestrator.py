# run_orchestrator.py
"""CLI script to run the badge generation orchestration pipeline."""

import json
import logging
import sys

from eval_client import EvalClient
from orchestrator import Orchestrator
from seedream_sdk import SeedreamClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

SAMPLE_INPUT = {
    "text_output": "Wells",
    "anchor_characterization": (
        "This Jordanian anchor hosts an energetic and highly interactive "
        "'Just Chatting' stream. He engages directly with his audience, "
        "responding to song requests, comments, and playful banter with "
        "quick-witted humor. Visually, he presents a casual style, "
        "frequently seen in a signature blue hoodie."
    ),
    "brand_palette": {
        "primary": {"name": "Gold", "hex": "#D4AF37"},
        "secondary": {"name": "Vibrant Blue", "hex": "#007BFF"},
        "tertiary": {"name": "Black", "hex": "#000000"},
    },
    "photo_analysis": {
        "gender": "male",
        "hair": "dark black side-parted short hair",
        "eyes": "deep brown eyes",
        "expression": "gentle smile",
        "skin_tone": "medium olive",
    },
}


def main():
    input_data = SAMPLE_INPUT
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            input_data = json.load(f)

    orch = Orchestrator(
        seedream_client=SeedreamClient(),
        eval_client=EvalClient(),
    )

    result = orch.run(input_data)

    print(f"\n{'='*60}")
    print(f"Result: {'PASSED' if result.passed else 'FAILED'}")
    print(f"Score: {result.score:.1f}")
    print(f"Rounds: {result.rounds}")
    print(f"Image size: {len(result.image)} bytes")

    if result.image:
        out_path = "output/orchestrator_result.jpg"
        with open(out_path, "wb") as f:
            f.write(result.image)
        print(f"Saved to: {out_path}")

    for i, (prompt, ev) in enumerate(zip(result.prompt_history, result.eval_history)):
        print(f"\n--- Round {i} ---")
        print(f"Score: {ev.total_score:.1f} | Passed: {ev.passed}")
        print(f"Dimensions: {ev.dimensions}")
        if ev.issues:
            print(f"Issues: {ev.issues}")
        print(f"Prompt: {prompt[:150]}...")


if __name__ == "__main__":
    main()
