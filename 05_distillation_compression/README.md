# Topic 5 — Distillation + Model Compression

Distills a larger teacher (`Qwen2.5-1.5B`) into a smaller student
(`Qwen2.5-0.5B`) using TRL's on-policy `GKDTrainer`, on the same customer-
support running example, and makes "soft targets" / "temperature" / "KL
divergence" concrete with a real worked numeric example.

## What you'll learn

- Teacher/student, logits, soft targets, KL divergence, temperature
- Response-level vs. on-policy distillation
- Capacity vs. quality trade-offs
- Real parameter-count-driven latency/cost comparisons

## Contents

- `generate_prompts.py` — reuses 200 prompts from Topic 1's dataset
- `data/distillation_prompts.json` — the generated prompt set
- `requirements.txt` — Python packages needed
- `distillation.ipynb` — the notebook
- `README.md` — this file

## How to run

**Recommended: Google Colab (free tier, T4 GPU)** — upload this folder,
open the notebook, set Runtime → GPU (T4), run all cells. Loads both the
1.5B teacher and 0.5B student — check you have enough Colab free-tier GPU
memory (T4 has 15GB, which comfortably fits both at this scale).

**Locally without a CUDA GPU** — safe to run. Prompt loading, the
temperature/KL-divergence worked example, and real parameter-count/cost
estimates all execute for real; the actual distillation training run is
skipped with a clear message.

## Execution runbook (sample output)

Actual captured output from running `distillation.ipynb` top to bottom
locally (no CUDA GPU, Python 3.12). ✅ = real output verified here; ⏭️ = GPU-only, skipped here.

### Step 1 — Welcome banner ✅

```
╔══════════════════════════════════════════════╗
║ Topic 5: Distillation + Model Compression    ║
║ Teacher Qwen2.5-1.5B -> Student Qwen2.5-0.5B ║
╚══════════════════════════════════════════════╝
```

### Step 2 — Environment check ✅

```
  Key              Value
 ━━━━━━━━━━━━━━━━━━━━━━━━━
  CUDA available   False
  PyTorch          2.14.1
GPU NOT AVAILABLE HERE: prompt loading, the KL/temperature worked example,
and real parameter-count/cost comparisons run for real; teacher generation
and distillation training need Colab GPU.
```

### Step 3 — Installs ✅

```
Not running in Colab -- assuming the project virtualenv already has these.
```

### Step 4 — Load the distillation prompt set ✅

```
── Loading the distillation prompt set ─────────────────────────
Loaded 200 prompts from .../data/distillation_prompts.json (reused from
Topic 1's running example)

Sample prompt: "I signed up for a monthly subscription but I was charged
again this month even though I thought I canceled it last week. Can you check?"
```

### Step 5 — Temperature + KL divergence worked example ✅ *(real computed numbers)*

```
    Teacher's distribution at different temperatures
  Token   T=1.0   T=2.0   T=5.0
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  16      0.902   0.614   0.347
  61      0.045   0.137   0.190
  26      0.027   0.107   0.172
  14      0.020   0.092   0.162
  40      0.006   0.050   0.128

At T=1, the teacher looks almost certain '16' is correct. At T=5, the
distribution flattens -- the student can now "see" that the teacher also
considered '61' and '26' somewhat plausible ("dark knowledge").

Untrained student: KL(teacher || student) = 1.1227
Trained student:   KL(teacher || student) = 0.0206
```

Lower KL = the student's distribution is closer to the teacher's — exactly
what distillation training minimizes.

### Step 6 — Real teacher/student parameter counts + cost estimate ✅

```
           Real parameter counts
  Key                       Value
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Teacher (Qwen2.5-1.5B)    1,543,714,304
  Student (Qwen2.5-0.5B)    494,032,768
  Teacher / student ratio   3.12x

Estimated per-token latency & relative cost (from real param counts)
  Metric                           Teacher (1.5B)    Student (0.5B)
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Parameters                       1,543,714,304     494,032,768
  Est. latency / token             ~12.3 ms          ~4.0 ms
  Est. relative cost per request   1.0x (baseline)   ~0.32x
```

Latency uses a simple illustrative linear-in-parameters constant, not a
measured benchmark — real serving latency also depends on batching,
hardware, and KV-cache behavior.

### Step 7 — On-policy distillation training ⏭️ *(GPU-only, skipped here)*

```
SKIPPED -- loading the teacher+student models and running GKDTrainer requires a CUDA GPU.
Expected on Colab (T4): the student's loss (KL divergence to the teacher's distribution)
decreasing over training; qualitatively, the student's responses should start to pick up
the teacher's tone/structure on customer-support prompts, even though it's 3x smaller.
```

### Step 8 — Scale discussion (~100 vs ~10k users) ✅

```
                                          Distillation economics at scale
  Concern                     ~100 users                                 ~10k users
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Is distillation worth it?   Often not yet -- teacher may be cheap      Frequently yes -- student serving most
                               enough already                            traffic (teacher fallback) cuts cost a lot
  Teacher generation cost     Cheap and quick for a few hundred prompts  Real pipeline cost at 10k+ prompts
  Serving cost per request    Dominated by fixed overhead                Scales with parameter count -- real $ savings
  Quality monitoring          Manual spot-check                         Needs ongoing eval pipeline vs. teacher
```

### Step 9 — Summary ✅

```
╔═══════════════════════════════════════════════ Topic 5 complete ════════════════════════════════════════════════╗
║ What we did:                                                                                                     ║
║   1. Reused Topic 1's customer-support prompts as the distillation prompt set                                   ║
║   2. Worked a real numeric example of temperature softening and KL divergence                                   ║
║   3. Built teacher (1.5B) and student (0.5B) architectures and computed REAL parameter counts                   ║
║   4. Estimated latency/cost from those real counts                                                               ║
║   5. (On GPU) ran on-policy distillation with TRL's GKDTrainer                                                   ║
║                                                                                                                  ║
║ Next: Topic 6 -- Scaling, Evaluation & Production Fine-Tuning                                                    ║
╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

### Verification summary

| Step | What it does | Verified here (no GPU) |
|---|---|---|
| 1–4 | Banner, env check, installs, load prompts | ✅ |
| 5 | Temperature softening + KL divergence worked example | ✅ (real numbers) |
| 6 | Real teacher/student parameter counts + cost estimate | ✅ |
| 7 | On-policy distillation training (GKDTrainer) | ⏭️ needs Colab GPU |
| 8 | Scale discussion table | ✅ |
| 9 | Summary banner | ✅ |

## Next

Topic 6 (`06_scaling_eval_production/`) builds the full production path:
training → evaluation → model registry → serving → monitoring.
