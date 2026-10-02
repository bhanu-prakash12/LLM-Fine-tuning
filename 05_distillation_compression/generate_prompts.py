"""Generate the prompt set for the distillation demo.

Real distillation setups use ~10k+ prompts; we use 200 to fit comfortably on
Colab's free tier while still being enough to see a trend. We reuse the same
running example (Amazon-style customer support) by reading the Topic 1
dataset's "problem" field, rather than re-writing a new generator.
"""

import json
from pathlib import Path

N_PROMPTS = 200
TOPIC1_DATA = Path(__file__).parent.parent / "01_sft_fundamentals" / "data" / "support_tickets.json"
OUTPUT_PATH = Path(__file__).parent / "data" / "distillation_prompts.json"


def main():
    with open(TOPIC1_DATA) as f:
        support_examples = json.load(f)

    prompts = [ex["problem"] for ex in support_examples[:N_PROMPTS]]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(prompts, f, indent=2)

    print(f"Wrote {len(prompts)} prompts to {OUTPUT_PATH} (reused from Topic 1's dataset)")
    print("Sample:", prompts[0])


if __name__ == "__main__":
    main()
