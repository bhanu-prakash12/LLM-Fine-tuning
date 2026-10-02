# Topic 4 — Preference Optimization + RL (SFT → DPO → GRPO)

Runs the modern post-training progression — **SFT → DPO → GRPO** — on a
single simple math word-problem task, where correctness is mechanically
checkable (a real verifier function), making the reward signal concrete.

## What you'll learn

- DPO: preferred vs. rejected response pairs, no separate reward model needed
- GRPO: multiple generated candidates per prompt, scored by a verifier
- How to write and test a verifier/reward function
- Why SFT → DPO → GRPO is a progression, not independent alternatives
- When a verifiable reward is (and isn't) available for a task

## Contents

- `generate_dataset.py` — deterministic generator for 200 math word problems with correct + plausible-wrong solutions
- `data/math_problems.json` — the generated dataset
- `requirements.txt` — Python packages needed
- `sft_dpo_grpo.ipynb` — the notebook
- `README.md` — this file

## How to run

**Recommended: Google Colab (free tier, T4 GPU)** — upload this folder,
open the notebook, set Runtime → GPU (T4), run all cells. Runs SFT, then
DPO on top of the SFT model, then GRPO on top of the DPO model.

**Locally without a CUDA GPU** — safe to run. Dataset generation,
preference-pair construction, and the verifier/reward function (with real
test cases) all execute for real; the three training stages are skipped
with a clear message.

## Execution runbook (sample output)

Actual captured output from running `sft_dpo_grpo.ipynb` top to bottom
locally (no CUDA GPU, Python 3.12). ✅ = real output verified here; ⏭️ = GPU-only, skipped here.

### Step 1 — Welcome banner ✅

```
╔═════════════════════════════════════════════╗
║ Topic 4: Preference Optimization + RL       ║
║ SFT -> DPO -> GRPO on a math reasoning task ║
╚═════════════════════════════════════════════╝
```

### Step 2 — Environment check ✅

```
  Key              Value
 ━━━━━━━━━━━━━━━━━━━━━━━━━
  CUDA available   False
  PyTorch          2.14.1
GPU NOT AVAILABLE HERE: dataset/preference-pair construction and the
verifier function are real CPU-safe work; SFT/DPO/GRPO training need Colab GPU.
```

### Step 3 — Installs ✅

```
Not running in Colab -- assuming the project virtualenv already has these.
```

### Step 4 — Load the math problem dataset ✅

```
── Loading the math problem dataset ────────────────────────────
Loaded 200 math problems from .../data/math_problems.json

Sample problem:
  Problem: Alex has 4 groups of apples, with 6 apples in each group. Alex
  gives away 8 apples. How many apples does Alex have left?

  Correct solution:
  Step 1: Alex starts with 4 groups of 6 apples, so the total is 4 x 6 = 24 apples.
  Step 2: Alex gives away 8 apples, so 24 - 8 = 16 apples.
  #### 16

  Wrong solution (DPO "rejected" example):
  Step 1: Alex has 4 groups of 6 apples, so the total is 4 x 6 = 24 apples.
  Step 2: Alex has 24 apples left.
  #### 24
```

### Step 5 — SFT dataset + training

```
SFT training set ready: Dataset({ features: ['text'], num_rows: 190 })
```

SFT dataset construction is ✅ *verified — real output*; the training run
itself is ⏭️ *GPU-only, skipped here*:

```
SKIPPED -- SFT training requires a CUDA GPU.
Expected on Colab: training loss dropping over ~3 epochs as the model learns
the Step 1/Step 2/#### <answer> format consistently.
```

### Step 6 — DPO preference pairs + training

```
DPO training set ready: Dataset({ features: ['prompt', 'chosen', 'rejected'], num_rows: 190 })

Sample DPO training example:
  Prompt (truncated): ...Chris gives away 26 apples. How many apples does Chris have left?
  Chosen: Step 1: Chris starts with 5 groups of 8 apples... Step 2: ...
  Rejected: Step 1: Chris has 5 groups of 8 apples... Step 2: Chris has 40 apples left...
```

Preference-pair construction is ✅ *verified — real output*; the DPO
training run is ⏭️ *GPU-only, skipped here*:

```
SKIPPED -- DPO training requires a CUDA GPU (and the SFT model from Stage 1).
Expected on Colab: DPO loss decreasing as the model's relative preference for
the chosen (correct) solution over the rejected one grows.
```

### Step 7 — Verifier / reward function ✅ *(verified — real output)*

```
── Testing the verifier function on real sample completions ────
                          Verifier reward -- real test cases
  Case                                    Extracted answer   Correct answer   Reward
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Correct completion                      16                 16               1.0
  Wrong completion                        24                 16               0.0
  Malformed completion (no #### marker)   None               16               0.0
```

This is a genuinely real, working reward function — no GPU needed to prove it's correct.

### Step 8 — GRPO training ⏭️ *(GPU-only, skipped here)*

```
SKIPPED -- GRPO training requires a CUDA GPU (and the DPO model from Stage 2).
Expected on Colab: for each prompt, 4 candidate completions are generated; the
fraction scoring reward=1 (correct final answer) should trend upward over
training as the model learns to favor whatever reasoning patterns reliably
produce correct answers.
```

### Step 9 — Scale discussion (~100 vs ~10k users) ✅

```
                                                 DPO/GRPO at scale
  Concern                       ~100 users                                ~10k users
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Preference data               A few hundred pairs show real signal      Needs thousands of real, reviewed pairs
  GRPO compute cost             Cheap for a small model                   Scales with generations x batch x problems
  Reward function reliability   Hand-checked verifier easy to audit       Needs reward-hacking monitoring
  Deployment risk               Manual review is enough                   Needs regression suites + canary rollout
```

### Step 10 — Summary ✅

```
╔═══════════════════════════════════════════════ Topic 4 complete ════════════════════════════════════════════════╗
║ What we did:                                                                                                     ║
║   1. Generated a math word-problem dataset with correct and plausible-wrong solutions                           ║
║   2. (On GPU) ran SFT to teach the task format and correct reasoning style                                      ║
║   3. Built DPO preference pairs and (on GPU) trained the model to prefer correct over wrong                     ║
║   4. Built and TESTED a real verifier/reward function (no GPU needed)                                           ║
║   5. (On GPU) ran GRPO, rewarding the model directly for correct final answers                                  ║
║                                                                                                                  ║
║ Next: Topic 5 -- Distillation + Model Compression                                                                ║
╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

### Verification summary

| Step | What it does | Verified here (no GPU) |
|---|---|---|
| 1–4 | Banner, env check, installs, load dataset | ✅ |
| 5 | SFT dataset build / SFT training | ✅ build / ⏭️ training needs Colab GPU |
| 6 | DPO preference pairs / DPO training | ✅ pairs / ⏭️ training needs Colab GPU |
| 7 | Verifier/reward function + real test cases | ✅ |
| 8 | GRPO training | ⏭️ needs Colab GPU |
| 9 | Scale discussion table | ✅ |
| 10 | Summary banner | ✅ |

## Next

Topic 5 (`05_distillation_compression/`) distills a larger teacher model into
a smaller student.
