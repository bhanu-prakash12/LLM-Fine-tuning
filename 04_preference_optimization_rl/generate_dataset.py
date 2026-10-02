"""Generate a small synthetic math-reasoning dataset for the SFT -> DPO ->
GRPO demo: multi-step arithmetic word problems with a correct step-by-step
solution, a "preferred vs rejected" pair for DPO, and a ground-truth final
answer a verifier function can check for GRPO.
"""

import json
import random
from pathlib import Path

SEED = 42
N_PROBLEMS = 200
OUTPUT_PATH = Path(__file__).parent / "data" / "math_problems.json"

NAMES = ["Sarah", "Alex", "Jordan", "Priya", "Sam", "Taylor", "Morgan", "Chris"]
ITEMS = ["apples", "boxes of pens", "notebooks", "stickers", "marbles", "coins"]


def make_problem(rng):
    name = rng.choice(NAMES)
    item = rng.choice(ITEMS)
    groups = rng.randint(2, 6)
    per_group = rng.randint(3, 12)
    given_away = rng.randint(1, groups * per_group - 1)

    total = groups * per_group
    final_answer = total - given_away

    problem = (
        f"{name} has {groups} groups of {item}, with {per_group} {item} in each group. "
        f"{name} gives away {given_away} {item}. How many {item} does {name} have left?"
    )

    correct_solution = (
        f"Step 1: {name} starts with {groups} groups of {per_group} {item}, "
        f"so the total is {groups} x {per_group} = {total} {item}.\n"
        f"Step 2: {name} gives away {given_away} {item}, so {total} - {given_away} = {final_answer} {item}.\n"
        f"#### {final_answer}"
    )

    # A plausible WRONG solution: forgets to subtract (common error pattern),
    # still "shows reasoning" so it's not trivially distinguishable by length.
    wrong_answer = total
    wrong_solution = (
        f"Step 1: {name} has {groups} groups of {per_group} {item}, "
        f"so the total is {groups} x {per_group} = {total} {item}.\n"
        f"Step 2: {name} has {total} {item} left.\n"
        f"#### {wrong_answer}"
    )

    return {
        "problem": problem,
        "correct_solution": correct_solution,
        "wrong_solution": wrong_solution,
        "final_answer": final_answer,
    }


def main():
    rng = random.Random(SEED)
    problems = [make_problem(rng) for _ in range(N_PROBLEMS)]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(problems, f, indent=2)

    print(f"Wrote {len(problems)} math problems to {OUTPUT_PATH}")
    print("Sample:")
    print(json.dumps(problems[0], indent=2))


if __name__ == "__main__":
    main()
