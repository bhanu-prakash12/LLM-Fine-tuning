# Topic 2 — LoRA / QLoRA / PEFT

Compares **Full fine-tuning vs. LoRA vs. QLoRA** on the exact same dataset
and model from Topic 1 (`Qwen3-0.6B`, 500 customer-support examples),
measuring trainable parameters, estimated/measured GPU memory, training
time, validation score, and adapter size.

## What you'll learn

- Low-rank matrices, trainable vs. frozen parameters, adapter weights
- LoRA rank `r`, LoRA alpha, target modules
- 4-bit quantization and why QLoRA reduces memory
- Adapter merging and running multiple adapters
- How to actually run and compare three fine-tuning methods side by side

## Contents

- `data/support_tickets.json` — the same 500-example dataset from Topic 1 (copied, not regenerated)
- `requirements.txt` — Python packages needed
- `lora_qlora_peft.ipynb` — the notebook
- `README.md` — this file

## How to run

**Recommended: Google Colab (free tier, T4 GPU)** — upload this folder,
open the notebook, set Runtime → GPU (T4), run all cells. This runs three
real short training experiments back-to-back (~1-2 min each).

**Locally without a CUDA GPU** — safe to run. Parameter counts, estimated
memory, and adapter size are computed for real from the model's actual
architecture (no GPU or weight download needed); the three real training
runs are skipped with a clear message.

## Execution runbook (sample output)

Actual captured output from running `lora_qlora_peft.ipynb` top to bottom
locally (no CUDA GPU, Python 3.12). ✅ = real output verified here; ⏭️ = GPU-only, skipped here (run on Colab for the real measured numbers).

### Step 1 — Welcome banner ✅

```
╔═══════════════════════════════════════════════════════════════╗
║ Topic 2: LoRA / QLoRA / PEFT                                  ║
║ Full FT vs LoRA vs QLoRA on the same customer-support dataset ║
╚═══════════════════════════════════════════════════════════════╝
```

### Step 2 — Environment check ✅

```
── Environment check ───────────────────────────────────────────
  Key              Value
 ━━━━━━━━━━━━━━━━━━━━━━━━━
  CUDA available   False
  PyTorch          2.14.1

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ GPU NOT AVAILABLE HERE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ No CUDA GPU detected. Structural comparisons below (parameter counts, adapter size, estimated memory) are      ┃
┃ computed for real without needing a GPU. Actual training time and validation score require a GPU -- run on    ┃
┃ Colab (T4) to get those.                                                                                       ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

### Step 3 — Installs ✅

```
Not running in Colab -- assuming the project virtualenv already has these.
```

### Step 4 — Load the shared dataset ✅

```
── Loading the shared dataset ──────────────────────────────────
Loaded 500 examples from .../02_lora_qlora_peft/data/support_tickets.json (same dataset as Topic 1)
```

### Step 5 — Build Full FT / LoRA / QLoRA model variants + count parameters ✅

```
── Building model variants (architecture only -- no weight download needed)
    Parameter counts (real, computed from the actual Qwen3-0.6B architecture)
  Key                    Value
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Full FT -- trainable   596,049,920 / 596,049,920 (100%)
  LoRA -- trainable      4,587,520 / 600,637,440 (0.76%)
  QLoRA -- trainable     4,587,520 / 600,637,440 (0.76%) (same adapters as LoRA)
```

These are **real parameter counts** from Qwen3-0.6B's actual architecture —
built via `AutoModelForCausalLM.from_config()`, which constructs the model
with correct parameter shapes instantly without downloading any pretrained
weights (works identically whether or not a GPU is present).

### Step 6 — Estimated GPU memory & adapter size ✅

```
                              Full FT vs LoRA vs QLoRA -- Topic 2 experiment results
  Metric                       Full FT (A)                 LoRA (B)                     QLoRA (C)
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  GPU memory (estimated)       ~4.77 GB                    ~1.23 GB                     ~0.33 GB
  Trainable parameters (real)  596,049,920                 4,587,520                    4,587,520
  Training time                measured on GPU -- see below (next step)
  Validation score             measured on GPU -- see below (next step)
  Adapter size on disk         N/A (full model)             ~9.2 MB                      ~9.2 MB (same adapter)
```

Memory estimates use a simple AdamW accounting model (trainable params ≈
weight+grad+2 optimizer moments; frozen params ≈ storage only at their
dtype size) — a back-of-envelope estimate to make the *relative* difference
concrete without needing a GPU.

### Step 7 — Real training experiments ⏭️ *(GPU-only, skipped here)*

```
SKIPPED -- running the three real training experiments requires a CUDA GPU.
Expected on Colab (T4): three short training runs (~1-2 min each for this small
dataset/model), after which Full FT should show the highest GPU memory and QLoRA the
lowest, while validation keyword-overlap scores should land in a similar range for
all three (LoRA/QLoRA shouldn't meaningfully lag Full FT in quality on a task this small).
```

### Step 8 — Scale discussion (~100 vs ~10k users) ✅

```
                                           LoRA/QLoRA adapters at scale
  Concern                     ~100 users                                 ~10k users
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Adapter storage             A few MB per adapter -- trivial to store   Still trivial -- storage was never the
                               many versions                              bottleneck; compounds with many adapters
  Serving multiple variants   One adapter at a time on a single GPU      vLLM-style multi-LoRA serving -- many
                                                                          adapters hot-swapped on one base model
  Retraining cost             Cheap enough to do ad hoc                  Still cheap per adapter -- bottleneck
                                                                          shifts to eval/review throughput
  Base model updates          Fast to re-run LoRA against new base       Needs a rollout plan: re-train/validate
                                                                          all adapters before switching traffic
```

### Step 9 — Summary ✅

```
╔═══════════════════════════════════════════════ Topic 2 complete ════════════════════════════════════════════════╗
║ What we did:                                                                                                     ║
║   1. Reused the Topic 1 dataset                                                                                  ║
║   2. Built Full FT / LoRA / QLoRA model variants and counted REAL trainable parameters                          ║
║   3. Computed estimated GPU memory and adapter size from those real counts                                      ║
║   4. (On GPU) ran all three as real training experiments and measured memory/time/quality                       ║
║   5. Discussed how adapters change the economics of serving many fine-tuned variants at scale                   ║
║                                                                                                                  ║
║ Next: Topic 3 -- Continued Pretraining / Domain Adaptation                                                       ║
╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

### Verification summary

| Step | What it does | Verified here (no GPU) |
|---|---|---|
| 1–4 | Banner, env check, installs, load dataset | ✅ |
| 5 | Build 3 model variants + count real trainable params | ✅ |
| 6 | Estimate GPU memory / adapter size from real counts | ✅ |
| 7 | Run 3 real training experiments, measure time/memory/score | ⏭️ needs Colab GPU |
| 8 | Scale discussion table | ✅ |
| 9 | Summary banner | ✅ |

## Next

Topic 3 (`03_continued_pretraining/`) moves from Q&A-style SFT to raw-text
next-token prediction (continued pretraining) on a technical corpus.
