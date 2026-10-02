Fine-Tuning Roadmap: 6 Topic Families for Big-Tech Training

For Big-Tech training, I would boil the whole fine-tuning landscape down to 6 topic families. There is no need to memorize dozens of algorithms. What matters is knowing why each family exists, when to apply it, what it changes inside the model, how the training runs, and how to show it made a difference.

The Hugging Face stack follows this same structure. SFT, DPO, GRPO, reward modeling, PEFT and distillation are all core training workflows in TRL/PEFT.

The 6 topics
#    Topic    Core question    Demo
1    SFT + Fine-Tuning Fundamentals    How do I teach a model a task?    Fine-tune a small model on 500 examples
2    LoRA / QLoRA / PEFT    How do I fine-tune a large model cheaply?    Compare full FT vs LoRA vs QLoRA
3    Continued Pretraining / Domain Adaptation    How do I teach a model domain language and knowledge patterns?    Adapt a base model to a technical corpus
4    Preference Optimization + RL    How do I make outputs more preferred, or improve reasoning?    SFT → DPO → GRPO on one task
5    Distillation + Model Compression    How do I shrink a strong model and cut its cost?    Teacher 3B → Student 0.5B
6    Scaling, Evaluation & Production Fine-Tuning    How do I make training reliable and ready for production?    Multi-GPU training + eval + deployment

1. SFT + Fine-Tuning Fundamentals

What to learn: Begin here, before LoRA. Get the full training loop clear in your head:

Dataset → Tokenizer → Forward pass → Loss → Backpropagation → Optimizer → Updated weights

Topics to cover: instruction tuning, prompt/completion datasets, conversational datasets, chat templates, cross-entropy loss, epochs, batch size, learning rate, gradient accumulation, overfitting, catastrophic forgetting, and how full fine-tuning differs from pretraining.

TRL's SFTTrainer handles both standard and conversational datasets, and its current quickstart fine-tunes a small Qwen model.

Demo: Build a customer-support classifier/explainer. The data maps a user problem to the expected support response. Use Qwen3-0.6B with 500 examples and SFT. Measure base-model accuracy/quality before, and fine-tuned accuracy/quality after.

Training questions to be ready for:

Why does SFT work?
What is the loss actually training on?
What happens when the dataset is too small?
Why can aggressive fine-tuning hurt general capabilities?

2. LoRA / QLoRA / PEFT

This is the most important practical fine-tuning topic. PEFT aims to train only a small set of added parameters rather than rewriting the whole model, and LoRA is the usual starting point.

What to learn: The progression is Full fine-tuning → LoRA → QLoRA. Understand low-rank matrices, trainable vs frozen parameters, adapter weights, rank r, LoRA alpha, target modules, quantization, 4-bit training, adapter merging, and running multiple adapters. PEFT supports QLoRA-style training explicitly, including targeting all linear layers.

Demo: Reuse the dataset from Demo 1 and run three experiments: A = full fine-tuning, B = LoRA, C = QLoRA. Record the following:

Metric    Full FT    LoRA    QLoRA
GPU memory
Trainable parameters
Training time
Validation score
Adapter size

Running this one experiment will teach you more than reading ten LoRA articles. Unsloth supports full fine-tuning, LoRA, QLoRA and related post-training methods, and keeps runnable notebooks for current models.

3. Continued Pretraining / Domain Adaptation

People often overlook this topic.

The key distinction:

SFT: Question → Answer
Continued pretraining: Raw domain text → Next-token prediction

For example: a base LLM is trained on 5B tokens of Java, PostgreSQL, ServiceNow docs and distributed-systems papers, producing a domain-adapted model.

What to learn: domain adaptation, corpus construction, token distributions, data mixtures, learning-rate selection, forgetting, domain-specific vocabulary, and continued pretraining vs SFT.

Demo: Take a small base model and a corpus of PostgreSQL docs, Java docs and distributed-systems articles. Run continued pretraining to get a domain model. Then build a test set covering domain terminology, domain Q&A, domain summarization and domain reasoning, and compare the base model against the continued-pretrained one. Unsloth lists continued pretraining next to LoRA, QLoRA, full fine-tuning, DPO and GRPO, so it works well for this experiment.

Training question: When would you pick continued pretraining over SFT? Being able to explain this distinction clearly is very important.

4. Preference Optimization + RL

This is the modern post-training layer: Base model → SFT → DPO → GRPO / RL. TRL offers SFT, DPO, GRPO, reward modeling and other post-training trainers.

What to learn:

DPO first. Each example has a prompt, a preferred answer and a rejected answer. The model learns to raise the probability of the preferred response and lower that of the rejected one.

Then GRPO / RL-style training. For each prompt the model produces several candidate answers, a reward or verifier scores them, and the model is updated accordingly.

Demo: Pick a simple math task.

SFT on question → correct reasoning.
Build preference pairs (good solution vs bad solution) and train with DPO.
Write a verifier (reward = 1 if the answer is correct, else 0) and run GRPO.

You end up with SFT → DPO → GRPO on a single task, which makes a strong training showcase because you can explain each step of the progression.

5. Distillation + Model Compression

This topic is about production economics. A larger teacher (7B / 14B) passes its knowledge to a small student (1B). The aim is to keep much of the teacher's behavior at a lower inference cost.

TRL now includes a DistillationTrainer, with an on-policy mode where the student generates its own completions and learns from the teacher's distribution.

What to learn: teacher/student, logits, soft targets, KL divergence, temperature, response-level distillation, on-policy distillation, capacity vs quality, and latency/cost trade-offs.

Demo: Teacher = Qwen 2.5 1.5B, Student = Qwen 2.5 0.5B. Prepare 10k prompts, have the teacher generate high-quality outputs, train the student on them, then evaluate. Compare quality, latency, GPU memory and cost per request. TRL's distillation example uses this same small-student / larger-teacher setup.

6. Scaling + Evaluation + Production

This is where you move from ML learner to Big-Tech AI engineer. Evaluation belongs in every fine-tuning experiment, but give scaling and operations their own dedicated project.

Distributed training: 1 GPU → Data Parallel → FSDP → Tensor Parallel → Pipeline Parallel. Megatron-LM offers advanced parallelism (TP, PP, DP, EP, CP) and includes production-scale examples such as LLaMA-3 training.

Evaluation: training loss, validation loss, task accuracy, instruction following, human preference, LLM judge, regression tests, safety tests.

Production path: checkpoint → model registry → staging → shadow → canary → production.

Demo: Take the LoRA model from Demo 2 and build Training → Evaluation → Model registry → vLLM → API. Then measure TTFT, TPOT, p50, p95, p99, tokens/sec, GPU utilization and cost per request.

This gives you a complete training story: "I didn't just fine-tune the model. I set a baseline, trained an adapter, checked for regressions, packaged the artifact, served it, and measured how it behaves in production."

The 6 demos at a glance
SFT from scratch. Goal: understand the mechanics (dataset → tokenizer → loss → gradients → weights). Qwen 0.6B + 500 examples.
Full FT vs LoRA vs QLoRA. Goal: understand efficiency. Same model, same data, three approaches; measure memory, speed, trainable parameters and quality. This is probably the highest-value demo.
Domain adaptation. Goal: understand continued pretraining. General LLM + technical corpus → domain model; evaluate domain performance and forgetting.
SFT → DPO → GRPO. Goal: understand modern post-training on a simple math/reasoning task.
Teacher → Student distillation. Goal: understand compression. 1.5B teacher → 0.5B student; measure quality vs latency vs cost.
Fine-tuning → production. Goal: think like a Staff/Principal engineer. Dataset → Training → Evaluation → Checkpoint → Registry → vLLM → Canary → Monitoring.

One important change to your learning strategy

Don't build six unrelated toy projects. Carry a single enterprise problem through all six stages, for example an Enterprise Incident Resolution AI:

Demo 1: SFT to generate resolution notes
Demo 2: LoRA / QLoRA
Demo 3: Domain adaptation on a ServiceNow / PostgreSQL / Java corpus
Demo 4: DPO / GRPO to improve reasoning and preferred answers
Demo 5: Distill into a smaller model
Demo 6: Deploy with vLLM + RAG + evaluation + canary

What you should be able to say after Big-Tech training

Once the six demos are done, you should be comfortable answering:

Why SFT?
Why LoRA instead of full fine-tuning?
When would you use QLoRA?
When is continued pretraining better than SFT?
What does DPO solve that SFT doesn't?
Why GRPO?
Why distill a model?
How do you evaluate whether fine-tuning actually helped?
How do you scale fine-tuning beyond one GPU?
How do you safely take the trained model to production?

That is the fine-tuning landscape I would target for Big-Tech training. For hands-on practice, Unsloth + Hugging Face PEFT + TRL is a practical combination: Unsloth covers LoRA/QLoRA, full fine-tuning, continued pretraining, DPO and GRPO, while TRL supplies the SFT/DPO/GRPO/reward/distillation stack.
