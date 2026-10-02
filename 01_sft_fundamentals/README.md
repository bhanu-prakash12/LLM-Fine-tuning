# Topic 1 — SFT + Fine-Tuning Fundamentals

Teaches the core supervised fine-tuning (SFT) loop by fine-tuning a small
model (`Qwen3-0.6B`, via Unsloth) on 500 synthetic customer-support examples
for an Amazon-style AI shopping assistant — the running example used across
this entire fine-tuning series.

## What you'll learn

- The full training loop: `Dataset → Tokenizer → Forward pass → Loss →
  Backprop → Optimizer → Updated weights`
- Instruction tuning, prompt/completion vs. conversational datasets, chat
  templates
- Cross-entropy loss, epochs, batch size, learning rate, gradient
  accumulation
- Overfitting and catastrophic forgetting
- How full fine-tuning differs from pretraining
- How to measure "did fine-tuning actually help" with a before/after
  comparison

## Contents

- `generate_dataset.py` — deterministic script that generates the 500
  synthetic (problem, response) pairs in `data/support_tickets.json`
- `data/support_tickets.json` — the generated dataset (10 categories, 50
  examples each: order status, returns/refunds, damaged items, payment
  issues, account access, shipping delays, cancellations, product
  questions, subscription issues, promo/discount issues)
- `requirements.txt` — Python packages needed
- `sft_fundamentals.ipynb` — the notebook
- `README.md` — this file

## How to run

**Recommended: Google Colab (free tier, T4 GPU)**

1. Upload this whole `01_sft_fundamentals/` folder to Colab (or mount Drive).
2. Open `sft_fundamentals.ipynb` and run all cells top to bottom.
3. Runtime → Change runtime type → GPU (T4) before running, so the
   Unsloth/training cells execute.

**Locally without a CUDA GPU**

The notebook is also safe to run locally without CUDA (e.g. on a Mac). Every
GPU-dependent cell (model loading, training, GPU inference) detects the
absence of a CUDA GPU and prints a clear "skipped — needs CUDA, run on
Colab" message instead of erroring, so the rest of the notebook (dataset
prep, tokenizer/chat-template demo, hyperparameter config, scale-economics
discussion) still runs and teaches the concepts. Regenerate the dataset
first if needed:

```bash
python3 generate_dataset.py
```

## Execution runbook (sample output)

This is the **actual captured output** from running `sft_fundamentals.ipynb`
top to bottom, so you can see what every cell produces without re-running
it. It was executed locally on a machine with no CUDA GPU (Apple M4 Pro,
Python 3.12), so CPU-safe cells show real output and GPU-only cells show
their graceful "SKIPPED" message with the output you should expect once you
run the same notebook on Colab with a T4 GPU.

### Step 1 — Welcome banner

```
╔══════════════════════════════════════════════════════════════════════╗
║ Topic 1: SFT + Fine-Tuning Fundamentals                              ║
║ Amazon-style AI shopping assistant — running example for this series ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Step 2 — Environment check

```
── Environment check ───────────────────────────────────────────
            Runtime
  Key                   Value
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Python                3.12.5
  PyTorch               2.14.1
  CUDA available        False
  Apple MPS available   True

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ GPU NOT AVAILABLE HERE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ No CUDA GPU detected on this machine.                                                                          ┃
┃                                                                                                                ┃
┃ Unsloth and bitsandbytes require a CUDA GPU. The cells below that need                                        ┃
┃ real model loading / training / GPU inference will detect this and                                           ┃
┃ print a clear SKIPPED message instead of failing -- run this                                                 ┃
┃ notebook on Google Colab with a T4 GPU (Runtime -> Change runtime type -> GPU)                                ┃
┃ to execute those cells for real.                                                                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

**On Colab with a T4 GPU**, `CUDA available` will read `True` and the GPU
name/memory (e.g. `Tesla T4`, `~15 GB`) will print instead of the warning
panel.

### Step 3 — Installs

```
Not running in Colab -- assuming the project virtualenv already has these installed
(see requirements.txt). Skipping pip install to save time.
```

**On Colab**, this cell instead runs `pip install -q unsloth transformers trl
peft bitsandbytes torch datasets rich` and prints `Installed packages for
Colab.`

### Step 4 — Load the dataset ✅ *(verified — real output)*

```
── Loading the dataset ─────────────────────────────────────────
Loaded 500 examples from data/support_tickets.json

    Category distribution
  Category             Count
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  account_access          50
  cancellation            50
  damaged_item            50
  order_status            50
  payment_issue           50
  product_question        50
  promo_discount          50
  returns_refunds         50
  shipping_delay          50
  subscription_issue      50

Sample example:
╭─────────────────────────────────────────────────────────────────────────────────────╮
│ Category: subscription_issue                                                        │
│                                                                                       │
│ Problem (user):                                                                      │
│ I signed up for a monthly subscription but I was charged again this month even      │
│ though I thought I canceled it last week. Can you check?                            │
│                                                                                       │
│ Response (assistant):                                                               │
│ I looked into your subscription and I can confirm the cancellation request from     │
│ last week did not fully process due to a billing sync issue on our end. I've now    │
│ canceled the subscription effective immediately and refunded this month's charge    │
│ in full — it will reflect within 3-5 business days.                                 │
╰─────────────────────────────────────────────────────────────────────────────────────╯
```

### Step 5 — Train/held-out split + chat template demo ✅ *(verified — real output)*

```
Training examples: 490  |  Held-out (never trained on) examples: 10

── Chat template demo ──────────────────────────────────────────
Before templating (raw conversation list):
[
    {
        'role': 'system',
        'content': "You are a helpful customer support assistant for an online shopping
platform. Respond to the customer's problem clearly and helpfully."
    },
    {
        'role': 'user',
        'content': "My running shoes from order #245095 was supposed to arrive 7 days
ago and it still hasn't shown up. This is really frustrating."
    },
    {
        'role': 'assistant',
        'content': "I completely understand the frustration, and I'm sorry for the
delay on your running shoes in order #245095. I checked with the carrier and the
package is delayed in transit but still moving — updated delivery is within 2 days.
As an apology for the delay, I've credited your account with a $10 voucher for your
next order."
    }
]

After templating (what the model actually sees as text):
╭─────────────────────────────────────────────────────────────────────────────────────╮
│ <|im_start|>system                                                                   │
│ You are a helpful customer support assistant for an online shopping platform.       │
│ Respond to the customer's problem clearly and helpfully.<|im_end|>                  │
│ <|im_start|>user                                                                     │
│ My running shoes from order #245095 was supposed to arrive 7 days ago and it        │
│ still hasn't shown up. This is really frustrating.<|im_end|>                        │
│ <|im_start|>assistant                                                               │
│ <think>                                                                              │
│                                                                                       │
│ </think>                                                                             │
│                                                                                       │
│ I completely understand the frustration, and I'm sorry for the delay on your        │
│ running shoes in order #245095. I checked with the carrier and the package is       │
│ delayed in transit but still moving — updated delivery is within 2 days. As an      │
│ apology for the delay, I've credited your account with a $10 voucher for your       │
│ next order.<|im_end|>                                                               │
╰─────────────────────────────────────────────────────────────────────────────────────╯

This example is 150 tokens long once templated.
```

*(Note the empty `<think></think>` block — Qwen3's chat template always
reserves a reasoning slot, even when the assistant turn itself has no
chain-of-thought.)*

### Step 6 — Load the model via Unsloth ⏭️ *(GPU-only, skipped here)*

```
╭───────────────────────────────────────────── Model load (GPU-only) ─────────────────────────────────────────────╮
│ SKIPPED -- loading Qwen3-0.6B via Unsloth requires a CUDA GPU.                                                   │
│ Expected on Colab: a few seconds to download + load the model, then a                                           │
│ confirmation that the model and tokenizer are ready for training.                                               │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### Step 7 — Baseline inference ⏭️ *(GPU-only, skipped here)*

```
╭───────────────────────────────────────── Baseline inference (GPU-only) ─────────────────────────────────────────╮
│ SKIPPED -- generating from the base model requires the model loaded above.                                      │
│ Expected on Colab: the un-tuned model's replies to 3 held-out customer problems,                                 │
│ typically generic, sometimes off-topic or missing the expected tone/structure.                                  │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### Step 8 — SFTTrainer configuration ✅ *(verified — real output)*

```
          SFTConfig (this run)
  Key                           Value
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  epochs                        3
  per_device_train_batch_size   4
  gradient_accumulation_steps   2
  effective_batch_size          8
  learning_rate                 0.0002
  max_length                    512
```

### Step 9 — Build training dataset + train

```
Training dataset ready: Dataset({
    features: ['text'],
    num_rows: 490
})
╭──────────────────────────────────────────── SFT training (GPU-only) ────────────────────────────────────────────╮
│ SKIPPED -- actual training requires a CUDA GPU.                                                                 │
│ Expected on Colab (T4): a progress bar over ~3 epochs x ~123 steps,                                             │
│ training loss printed every 5 steps, generally dropping from roughly                                           │
│ 2.5-3.0 down toward 0.3-0.8 by the end of training for a dataset this size.                                     │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Building the `Dataset` object (490 rows) is ✅ *verified — real output*; the
training run itself is ⏭️ *GPU-only, skipped here*.

### Step 10 — Post-training comparison ⏭️ *(GPU-only, skipped here)*

```
╭────────────────────────────────────── Post-training comparison (GPU-only) ──────────────────────────────────────╮
│ SKIPPED -- post-training inference requires the trained model above.                                            │
│ Expected on Colab: the fine-tuned model's replies should closely match the                                      │
│ style/structure/tone of the dataset's target responses (empathetic opener,                                      │
│ concrete resolution, specific timeframe) far more consistently than the base                                    │
│ model's generic replies captured earlier.                                                                       │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### Step 11 — Scale discussion (~100 vs ~10k users) ✅ *(verified — real output)*

```
── Scale discussion: ~100 users vs ~10k users ──────────────────
                          SFT fine-tuning at scale (Qwen3-0.6B-class model)

  Concern                ~100 users                                 ~10k users
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Training cost          Single T4/A10 GPU, < $1-5 per retrain,      Still a small model -- a few A100-hours per
                         minutes to run                              retrain; the cost center shifts to data
                                                                     curation and eval, not raw compute

  Retrain cadence        Ad hoc, whenever support playbooks change   Scheduled retrains (e.g. weekly) as new
                                                                     support categories/edge cases accumulate

  Data volume needed     500 examples is plausible and sufficient    Needs thousands-to-tens-of-thousands of
                         to start                                    real (sampled + reviewed) support
                                                                     interactions to cover the long tail of
                                                                     categories

  Serving latency        A single instance easily absorbs bursty     Needs autoscaled inference replicas,
                         traffic; no autoscaling needed               request batching, and p95/p99 latency SLOs
                                                                     (Topic 6)

  Memory / infra         One GPU, maybe even CPU-only for a 0.6B     Multiple GPU replicas behind a load
                         model at low QPS                             balancer; GPU utilization becomes a cost
                                                                     lever

  Operational concerns   Manual before/after spot-checks (like this  Needs automated regression eval suites,
                         notebook) are enough                        canary rollout, and rollback plans before
                                                                     every retrain reaches 100% of traffic
                                                                     (Topic 6)
```

### Step 12 — Summary ✅ *(verified — real output)*

```
── Summary ─────────────────────────────────────────────────────
╔═══════════════════════════════════════════════ Topic 1 complete ════════════════════════════════════════════════╗
║ What we did:                                                                                                     ║
║   1. Generated/loaded 500 synthetic customer-support examples                                                    ║
║   2. Converted them to chat-formatted conversations using Qwen3's real chat template                             ║
║   3. (On GPU) loaded Qwen3-0.6B via Unsloth and captured BASELINE responses                                      ║
║   4. Configured and ran SFTTrainer with explained hyperparameters                                                ║
║   5. (On GPU) compared BEFORE vs AFTER responses on held-out problems                                            ║
║   6. Discussed how this scales from ~100 to ~10k users                                                           ║
║                                                                                                                   ║
║ Next: Topic 2 -- LoRA / QLoRA / PEFT                                                                             ║
║ We reuse this exact dataset and run three experiments on it --                                                  ║
║ Full fine-tuning vs. LoRA vs. QLoRA -- and measure GPU memory, trainable                                        ║
║ parameters, training time, validation score and adapter size side by side.                                      ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

### Verification summary

| Step | What it does | Verified here (no GPU) | Needs Colab GPU |
|---|---|---|---|
| 1–3 | Banner, env check, installs | ✅ | — |
| 4 | Load dataset + inspect | ✅ | — |
| 5 | Chat template rendering | ✅ | — |
| 6 | Load Qwen3-0.6B via Unsloth | — | ✅ |
| 7 | Baseline (pre-SFT) inference | — | ✅ |
| 8 | SFTConfig construction | ✅ | — |
| 9 | Build `Dataset` / run `trainer.train()` | ✅ (dataset build only) | ✅ (training) |
| 10 | Post-training before/after comparison | — | ✅ |
| 11 | Scale discussion table | ✅ | — |
| 12 | Summary banner | ✅ | — |

## Next

Topic 2 (`02_lora_qlora_peft/`) reuses this same dataset and compares full
fine-tuning vs. LoRA vs. QLoRA on it.
