# Topic 3 — Continued Pretraining / Domain Adaptation

Teaches the key distinction between SFT (`Question → Answer`) and continued
pretraining (`Raw domain text → next-token prediction`) using a tiny
synthetic multi-domain corpus (PostgreSQL internals, Java, distributed
systems, ServiceNow/Glide platform docs).

## What you'll learn

- Domain adaptation, corpus construction, token distributions, data mixtures
- Why continued pretraining uses a smaller learning rate than SFT
- Forgetting risk in an unlabeled-text setting
- Domain-specific vocabulary and tokenization
- Continued pretraining vs. SFT — when to use which (and in what order)

## Contents

- `generate_corpus.py` — deterministic script generating the synthetic domain corpus + a small domain eval set
- `data/domain_corpus.txt` — ~2,400-word synthetic corpus across 4 domains (25 paragraphs each)
- `data/domain_eval.json` — 8 domain eval prompts (terminology, Q&A, summarization, reasoning)
- `requirements.txt` — Python packages needed
- `continued_pretraining.ipynb` — the notebook
- `README.md` — this file

## How to run

**Recommended: Google Colab (free tier, T4 GPU)** — upload this folder,
open the notebook, set Runtime → GPU (T4), run all cells.

**Locally without a CUDA GPU** — safe to run. Corpus generation,
tokenization, chunking into training blocks, and the domain-vocabulary
tokenizer analysis are all real CPU-safe work; the actual continued
pretraining run and before/after domain-eval generation are skipped with a
clear message.

## Execution runbook (sample output)

Actual captured output from running `continued_pretraining.ipynb` top to
bottom locally (no CUDA GPU, Python 3.12). ✅ = real output verified here; ⏭️ = GPU-only, skipped here.

### Step 1 — Welcome banner ✅

```
╔════════════════════════════════════════════════════╗
║ Topic 3: Continued Pretraining / Domain Adaptation ║
║ Raw technical corpus -> next-token prediction      ║
╚════════════════════════════════════════════════════╝
```

### Step 2 — Environment check ✅

```
── Environment check ───────────────────────────────────────────
  Key              Value
 ━━━━━━━━━━━━━━━━━━━━━━━━━
  CUDA available   False
  PyTorch          2.14.1
GPU NOT AVAILABLE HERE: corpus prep/tokenization/chunking are real CPU-safe
work; continued-pretraining + domain-eval generation need Colab GPU.
```

### Step 3 — Installs ✅

```
Not running in Colab -- assuming the project virtualenv already has these.
```

### Step 4 — Load the domain corpus ✅

```
── Loading the domain corpus ───────────────────────────────────
Loaded 100 paragraphs, 2431 words from .../data/domain_corpus.txt

Sample paragraph:
"The JIT compiler in the JVM is responsible for deciding which runnable
thread executes next. This scales much better under contention than a
synchronized HashMap."
```

### Step 5 — Tokenize + chunk the corpus ✅

```
── Tokenizing the corpus ───────────────────────────────────────
Full corpus tokenizes to 2972 tokens (1.22 tokens/word)
Chunked into 23 fixed-length blocks of 128 tokens each -- this is the actual
training unit for continued pretraining (no prompt/response split, every
token in the block is both input and target).
```

### Step 6 — Domain vocabulary tokenization check ✅

```
      Domain term -> token breakdown (real tokenizer output)
  Term                 # tokens   Tokens
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MVCC                        2   ['MV', 'CC']
  autovacuum                  4   ['aut', 'ov', 'ac', 'uum']
  WAL                         2   ['W', 'AL']
  tablespace                  2   ['table', 'space']
  ConcurrentHashMap            3   ['Con', 'current', 'HashMap']
  CompletableFuture            2   ['Comple', 'tableFuture']
  Raft                        2   ['Ra', 'ft']
  Paxos                       3   ['P', 'ax', 'os']
  sys_attachment_doc          3   ['sys', '_attachment', '_doc']
  ServiceNow                  2   ['Service', 'Now']
  PgBouncer                   3   ['Pg', 'B', 'ouncer']
  idempotent                  3   ['id', 'emp', 'otent']
```

Terms that split into many sub-word pieces are ones the tokenizer wasn't
built around — the model has to learn to "reassemble" their meaning from
fragments, which is part of why continued pretraining on domain text helps.

### Step 7 — Build LM dataset + trainer config ✅

```
Continued pretraining config (this run)
  Key                Value
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  block_size         128
  num_blocks         23
  learning_rate      2e-5  (vs. 2e-4 used for SFT in Topic 1 -- 10x smaller)
  num_train_epochs   3
  objective          causal language modeling (predict every next token)
```

### Step 8 — Continued pretraining + domain eval ⏭️ *(GPU-only, skipped here)*

```
SKIPPED -- continued pretraining and domain-eval generation require a CUDA GPU.
Expected on Colab (T4): baseline answers that are generic/non-specific about these domain
terms, a training loss curve over the small corpus, then after-training answers that more
often reference the specific terminology/relationships seen in the synthetic corpus
(though with a corpus this tiny, don't expect deep new knowledge -- the point is observing
the mechanism, not producing a genuinely domain-expert model).
```

### Step 9 — Scale discussion (~100 vs ~10k users) ✅

```
                                            Domain adaptation at scale
  Concern               ~100 users                                    ~10k users
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Corpus size needed    A few hundred MB can meaningfully help        GBs-to-TBs of curated text; corpus curation
                                                                       becomes a dedicated pipeline
  Training cost         Hours on a single GPU for a small model      Many GPU-days/weeks, often cluster-scale
  Data mixture tuning   Can eyeball a reasonable domain/general split Needs systematic ablations to avoid
                                                                       regressing general capability
  Refresh cadence       Ad hoc, when domain material changes          Periodic (e.g. quarterly), paired with
                                                                       automated regression evals
```

### Step 10 — Summary ✅

```
╔═══════════════════════════════════════════════ Topic 3 complete ════════════════════════════════════════════════╗
║ What we did:                                                                                                     ║
║   1. Generated a tiny synthetic multi-domain corpus                                                              ║
║   2. Tokenized and chunked it into fixed-length blocks for next-token-prediction training                       ║
║   3. Inspected how the tokenizer handles real domain vocabulary                                                 ║
║   4. Configured a continued-pretraining run (CLM objective, small LR)                                           ║
║   5. (On GPU) ran it and compared before/after answers on a domain eval set                                     ║
║                                                                                                                  ║
║ Next: Topic 4 -- Preference Optimization + RL (SFT -> DPO -> GRPO on a math task)                                ║
╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

### Verification summary

| Step | What it does | Verified here (no GPU) |
|---|---|---|
| 1–3 | Banner, env check, installs | ✅ |
| 4 | Load domain corpus | ✅ |
| 5 | Tokenize + chunk into training blocks | ✅ |
| 6 | Domain vocabulary tokenization analysis | ✅ |
| 7 | Build LM dataset + trainer config | ✅ |
| 8 | Run continued pretraining + before/after domain eval | ⏭️ needs Colab GPU |
| 9 | Scale discussion table | ✅ |
| 10 | Summary banner | ✅ |

## Next

Topic 4 (`04_preference_optimization_rl/`) runs SFT → DPO → GRPO on a simple
math reasoning task.
