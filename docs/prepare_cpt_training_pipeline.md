# Preparing the CPT Training Pipeline — `Qwen/Qwen3.5-4B-Base` on one 48 GB GPU

This document records how the continued pre-training (CPT) run is set up and why: the training method, the
settings, epochs, checkpoints, live monitoring, how to check *during* the run that the model is improving,
and the AWS machine it runs on. It is a plan, not code; the training script is built from it.

- The dataset it trains on is built by `scripts/cpt/` — see [prepare_cpt_dataset.md](prepare_cpt_dataset.md).
- The big picture (CPT → SFT → merging, compute for later stages) is in [cpt_overview.md](cpt_overview.md).
  Where the two disagree, **this document wins** (see [section 13](#13-changes-compared-with-cpt_overviewmd)).
- API names were checked against `transformers` 5.19.0, the version the baseline notebooks pin.
- Dates: written 2026-10-17; the test phase ends 2026-10-27.

---

## Table of contents

1. [Decisions at a glance](#1-decisions-at-a-glance)
2. [What we are training](#2-what-we-are-training)
3. [Training method: full-parameter CPT](#3-training-method-full-parameter-cpt)
4. [Training settings](#4-training-settings)
5. [How many epochs](#5-how-many-epochs)
6. [Checkpoints and resuming after a failure](#6-checkpoints-and-resuming-after-a-failure)
7. [Watching progress live](#7-watching-progress-live)
8. [Checking that the model improves during training](#8-checking-that-the-model-improves-during-training)
9. [Traps and tips](#9-traps-and-tips)
10. [The machine: AWS g6e.xlarge](#10-the-machine-aws-g6exlarge)
11. [Budget](#11-budget)
12. [Order of work](#12-order-of-work)
13. [Changes compared with cpt_overview.md](#13-changes-compared-with-cpt_overviewmd)
14. [References](#14-references)

---

## 1. Decisions at a glance

| Question | Decision | Section |
|---|---|---|
| Pre-train from scratch, LoRA, or something else? | **Full-parameter continued pre-training** of the Base model. LoRA only as a fallback. | 3 |
| Does an unmerged LoRA break the 8B limit? | No (~4.2B + ~0.24B). We still merge, for simplicity and reproduction. | 3.2 |
| How does it fit in 48 GB? | bf16 weights + 8-bit AdamW with bf16 stochastic rounding + gradient checkpointing (~25 GB + activations). | 3.3 |
| How many epochs? | **One pass** over the packed data = 3 textbook epochs + 1 replay pass. Checkpoints at the epoch boundaries tell us if 2 or 4–5 would be better. | 5 |
| Checkpoints? | Full checkpoint every 25 steps (keep 2) + weights-only copies at the epoch boundaries, pushed to a private HF repo. Tested resume. | 6 |
| Watching progress? | Weights & Biases, live from browser or phone. | 7 |
| Is it improving vs the base model? | Step-0 baseline, then per-source validation loss + two MCQ probes during training, with stop/continue gates. | 8 |
| Where? | AWS g6e.xlarge (1× L40S 48 GB), On-Demand, stopped whenever not training. | 10 |

---

## 2. What we are training

| | |
|---|---|
| Model | `Qwen/Qwen3.5-4B-Base`, ~4.2B parameters, text-only (`Qwen3_5ForCausalLM`; the vision tower and multi-token-prediction head are not loaded) |
| Layers | 32: **24 Gated DeltaNet** (linear attention) + **8 full attention** (head dim 256, output gate), hidden size 2,560, MLP 9,216 |
| Vocabulary | 248,320 tokens; input and output embeddings are **tied** (one shared matrix, ~0.64B parameters) |
| Special tokens | `pad_token` = `eos_token` = `<|endoftext|>` (id 248044), no BOS. The same token separates documents in the packed blocks. |
| Train set | `data/processed/cpt/easy/packed/train/` — 11,181 blocks × 2,048 = **22.9M tokens** |
| Train mixture | Textbooks 3.5M unique tokens × 3 copies = 10.9M (48%); replay 12M (52%): Sinhala Wikipedia 4.0M, FineWeb-2 Sinhala 5.0M, English FineWeb-Edu 3.0M |
| Train order | Three consecutive **epoch segments** (column `epoch`), each = one textbook copy + ⅓ of the replay, shuffled inside the segment: blocks 0–3,743 / 3,744–7,473 / 7,474–11,180 |
| Validation set | `data/processed/cpt/easy/mix/val.jsonl` (also packed: 104 blocks, 213k tokens). Textbook 134k tokens from 19 held-out chapters, Wikipedia 40k, FineWeb-2 23k, English 16k. |

---

## 3. Training method: full-parameter CPT

**Objective:** plain next-token prediction on every token (`labels = input_ids`), no chat template, no prompt
masking. It is the same objective Qwen was pre-trained with, applied to new text: *continued* pre-training.
Pre-training from scratch is not an option (it needs trillions of tokens).

### 3.1 Options compared

| Option | Verdict | Why |
|---|---|---|
| **Full-parameter CPT** | **Use this** | The goal is to store textbook facts in the weights (inference allows no retrieval). Full fine-tuning absorbs new knowledge clearly better than LoRA ("LoRA Learns Less and Forgets Less", Biderman et al., 2024). The 52% replay data limits forgetting. It fits in 48 GB with the setup in 3.3. |
| LoRA r = 128 on all linear layers (incl. the DeltaNet projections) + trainable embeddings → merge | Fallback | Only if full training runs out of memory or is unstable. Learns less, forgets less. LR ~1e-4 for the adapters, ~1e-5 for the embeddings. |
| QLoRA (4-bit base) | No | Slower, not needed on 48 GB, and Unsloth advises against 4-bit for Qwen3.5 (larger-than-usual quantisation error). |
| Pre-training from scratch | No | Needs orders of magnitude more data and compute. |

### 3.2 The 8B parameter limit and LoRA

LoRA does not put the system over 8B. LoRA r = 128 on every linear layer of this model is ≈ 0.24B parameters,
so base + adapter ≈ 4.2 + 0.24 ≈ 4.4B. We still **merge** every adapter (`merge_and_unload()`): one model is
simpler to run, to publish and for the organisers to re-run, and the SFT stage adds its own adapter on top.

### 3.3 Memory on a 48 GB GPU

| Setup | Bytes per parameter | ~Total for 4.2B | Fits 48 GB? |
|---|---:|---:|---|
| Standard mixed precision: fp32 master weights + fp32 grads + fp32 AdamW states | 16 | ~67 GB + activations | No |
| bf16 weights + bf16 grads + **8-bit AdamW** | 6 | **~25 GB** + activations | **Yes** |

Activations are kept small with **gradient checkpointing** (`supports_gradient_checkpointing = True` for
Qwen3.5). The biggest activation is the output layer: with a 248k vocabulary, one micro-batch of 4 × 2,048
tokens produces ~2 billion logits (~4 GB in bf16, ~8 GB once upcast to fp32 for the loss). Start with a
micro-batch of 2 and try 4 in the smoke test.

### 3.4 The bf16 rounding trap and the fix

With bf16 weights and a learning rate around 2e-5, most weight updates are smaller than bf16 can represent
(~3 significant digits) and **round to zero**: the model barely trains while the loss looks plausible. The fix
is **stochastic rounding**, which is built into the Trainer through torchao's 8-bit AdamW:

```python
optim="adamw_torch_8bit",                      # torchao AdamW8bit (needs torchao >= 0.11; pip install torchao)
optim_args="bf16_stochastic_round=True",
```

Load the model with `dtype=torch.bfloat16` and set `bf16=True`. Do **not** use the bitsandbytes 8-bit optimizers
(`adamw_8bit`, `paged_adamw_8bit`) on bf16 weights: they have no stochastic rounding.

---

## 4. Training settings

| Setting | Value | `TrainingArguments` / note |
|---|---|---|
| Passes over the data | **1** | `num_train_epochs=1` (the 3 textbook copies are already in the data) |
| Sequence length | 2,048 | = the packed block size |
| Global batch | 32 sequences = 65,536 tokens per step → **~350 steps** | `per_device_train_batch_size=2` × `gradient_accumulation_steps=16` (or 4 × 8 if memory allows) |
| Epoch boundaries | steps **117, ~234, ~350** | 3,744 / 7,474 / 11,181 blocks ÷ 32 |
| Peak learning rate | **2e-5** (try 1e-5 – 3e-5) | `learning_rate=2e-5` |
| Warm-up | 5% (~18 steps) | `warmup_steps=0.05` (a float < 1 is a ratio in v5; there is no `warmup_ratio`) |
| Schedule | Warm-up → constant → linear decay over the last 20% (~70 steps) to 10% of the peak | `lr_scheduler_type="warmup_stable_decay"`, `lr_scheduler_kwargs={"num_decay_steps": 70, "decay_type": "linear", "min_lr_ratio": 0.1}` |
| Optimizer | 8-bit AdamW with bf16 stochastic rounding | see 3.4; `adam_beta1=0.9`, `adam_beta2=0.95`, `weight_decay=0.1`, `max_grad_norm=1.0` |
| Precision | bf16 | `bf16=True`, model loaded in bf16 |
| Memory | gradient checkpointing | `gradient_checkpointing=True` |
| Data order | **in order, not shuffled** | `train_sampling_strategy="sequential"` (built in, default is `"random"`) |
| Seed | 42 (= the dataset seed) | `seed=42`, `data_seed=42` |
| Logging | every step | `logging_steps=1`, `report_to="wandb"` |

### 4.1 Data loading and labels

- Read the Parquet shards with `datasets.load_dataset("parquet", ...)`. Each row is already exactly 2,048 tokens:
  no tokenising, no padding, no attention mask needed.
- **Set `labels = input_ids` yourself** (e.g. with `default_data_collator` and a `labels` column, or a 3-line
  collator). Do **not** use `DataCollatorForLanguageModeling`: it sets the label of every `pad_token_id` to
  -100, and here the pad token *is* `<|endoftext|>`, the document separator. The model would never learn where a
  document ends.
- The packed blocks contain several documents separated by `<|endoftext|>`, and attention (and the DeltaNet
  state) flows across the separator. This is standard for CPT and fine at 2,048 tokens. Resetting per document
  ("varlen"/padding-free training with `cu_seqlens`) is a possible later optimisation, not needed now.

### 4.2 Why sequential order

`s10_pack.py` already shuffles the documents inside each epoch segment, and every segment contains replay data.
Reading the blocks **in order** keeps the segments in order, so the checkpoints at steps 117 / ~234 / ~350 are
"after 1 / 2 / 3 textbook epochs". That turns one run into an epoch experiment for free (section 5). It also
spaces the repeats: each textbook section is seen once per segment, never twice in a row.

### 4.3 Why warm-up–stable–decay (WSD) instead of cosine

1. The checkpoints at the first two epoch boundaries are taken at the **same** learning rate, so they can be
   compared fairly (with cosine, each would sit at a different LR).
2. The checkpoint **before** the decay can be continued later with more data (Medium/Hard textbooks) without a new
   warm-up, instead of re-training from the base model. With little time left, this can save a run.

### 4.4 Fast kernels for the DeltaNet layers

The 24 Gated DeltaNet layers need two optional packages: **`flash-linear-attention`** and **`causal-conv1d`**.
Without them, `transformers` silently uses a readable reference PyTorch path that its own source code describes as
"more than an order of magnitude" slower (on an H100) and prints, once:

> `` `chunk_gated_delta_rule` is falling back to its reference PyTorch implementation because `flash-linear-attention` is not installed ``

Install both on the training machine and make sure this warning does **not** appear in the smoke test.
The 8 full-attention layers work with the default `sdpa` attention.

### 4.5 Compute and time

- Work ≈ 6 × parameters × tokens = 6 × 4.2B × 22.9M ≈ **5.8 × 10¹⁷ FLOPs**, plus about a third more for
  gradient checkpointing.
- L40S: ~362 dense bf16 TFLOPS. At 30–40% utilisation: **~1.5–2 hours**; plan 2–3 hours with evaluations
  (≈ $4–6 on g6e.xlarge).
- The real number comes from the smoke test: tokens per second → ETA.

---

## 5. How many epochs

- **One pass over the packed data**, which is already 3 textbook epochs and 1 replay pass.
- Why 3: in "Scaling Data-Constrained Language Models" (Muennighoff et al., 2023), up to ~4 repeats of the same
  data were almost as useful as new data; beyond that the gains drop quickly, and memorising and forgetting grow.
- **Measure rather than guess.** Run the probes of section 8 on the checkpoints at the three epoch boundaries:

| What the probes show | Next run |
|---|---|
| Still rising fast after epoch 3, replay losses flat | `textbook_epochs` 4–5 in [mixture.json](../scripts/cpt/config/mixture.json) |
| Flat after epoch 2 | `textbook_epochs` 2 |
| Replay losses rising (forgetting) | fewer epochs, lower LR, or more replay |

- The epoch-1 and epoch-2 checkpoints have not gone through the LR decay yet, so they score a little lower than a
  fully decayed run of the same length would.
- Beyond ~4 epochs, **reworded copies** of the textbooks (paraphrased by an LLM) teach facts better than more
  identical repeats: a small model that has seen a fact in only one wording often cannot use it to answer a
  question ("Physics of Language Models, Part 3.1", Allen-Zhu & Li). This is a later improvement; any closed
  API used for it must be disclosed.

---

## 6. Checkpoints and resuming after a failure

| What | How | Size |
|---|---|---|
| **Full checkpoint** (weights, optimizer, LR schedule, RNG, step, data position) | `save_strategy="steps"`, `save_steps=25` (~8–15 min), `save_total_limit=2` | ~17 GB each |
| **Weights-only copy at each epoch boundary** (steps 117, ~234, end) | a small callback calls `save_pretrained` (bf16 safetensors + tokenizer) into `epochs/e1`, `e2`, `e3`, and pushes it to a **private** Hugging Face repo | ~8.4 GB each |

- **Keep 2 full checkpoints:** the latest to resume from, the one before it to go back to if the loss suddenly
  spikes.
- **Resume** with `trainer.train(resume_from_checkpoint=True)`. It loads the last checkpoint in `output_dir`,
  restores everything and skips the batches already trained on. With the sequential sampler the skip is exact.
  Keep the same batch size, sequence length and data.
- **Disk:** save to the EBS volume. Never put checkpoints on the instance-store NVMe (`/opt/dlami/nvme` on the
  Deep Learning AMI): it is wiped when the machine stops.
- **Off the machine:** only the weights-only copies go to Hugging Face, never the optimizer checkpoints (size
  and download costs). The published weights are needed for the reproduction phase anyway.
- **Test the resume once on purpose**: stop the run right after the first checkpoint, resume, and check that the
  step count and the loss carry on. Five minutes now protects hours later.
- **Run inside `tmux`.** If the SSH connection drops, VS Code's terminal closes and takes the training with it.
  Never train from a notebook cell.

| Failure | What protects you |
|---|---|
| SSH / VS Code disconnects | `tmux` |
| Crash, out of memory | resume from the latest checkpoint (≤ 25 steps lost) |
| Machine stopped (manually, auto-stop, capacity) | checkpoints on EBS survive; start again and resume |
| Loss spike / NaN | resume from the older of the two checkpoints, lower the LR |

---

## 7. Watching progress live

- **Weights & Biases** (free personal account): `report_to="wandb"`, `run_name` per run. Charts update live in
  the browser or on a phone, survive the machine stopping, and runs overlay for comparison.
- Log: training loss, the four validation losses, both probe accuracies, learning rate, gradient norm,
  tokens per second.
- Put the git commit, the dataset `manifest.json` sha1, the full config, the seed and `pip freeze` in the W&B run
  config: it is also the reproducibility record for the report.
- A second tmux pane with `nvidia-smi -l 5` (or `nvtop`) shows GPU memory and utilisation.

---

## 8. Checking that the model improves during training

A run costs hours, so the improvement is checked **while it trains**, against the base model, with stop points
that end a bad run early.

### 8.1 Step 0 is the base model

`eval_on_start=True` runs the evaluation before the first update, and the probe callback also runs in
`on_train_begin`. Every chart therefore starts with the base model's value as the reference line.

### 8.2 Validation loss per source

- Pass four eval sets as a dict: `eval_dataset={"textbook": ..., "wiki": ..., "fineweb2": ..., "english": ...}`.
  The Trainer logs `eval_textbook_loss`, `eval_wiki_loss`, ... separately.
- Build them in the training script from `mix/val.jsonl` (it has a `source` column): tokenise and pack per source
  into 2,048-token blocks. It is only 213k tokens, so this takes seconds and needs no change to the dataset.
- `eval_strategy="steps"`, `eval_steps=25` (~30 seconds each).

| Curve | Expected | Warning sign |
|---|---|---|
| `textbook` (held-out chapters) | clearly down | flat → bug, or LR too low |
| `wiki`, `fineweb2` (Sinhala) | slightly down | up > 5% |
| `english` | about flat | up > 5–10% → forgetting: lower LR, fewer textbook epochs, more replay |

The *training* loss on textbooks drops sharply in epochs 2–3 because the model has seen that exact text: that is
not evidence of learning. The held-out chapters and the probes are.

### 8.3 Multiple-choice probes

Both use the scoring of the competition: the official Sinhala prompt and the option digit with the highest
next-token logit, as in [scripts/eval_baseline.py](../scripts/eval_baseline.py) — but **as plain text, without
the chat template** (the Base model has none). The prompt ends with `පිළිතුර:` followed by a space: Qwen
tokenizes `: 3` as `:` + ` ` + `3`, so the next token is the bare digit (ids 16–19). Use `build_prompt` from
[scripts/probes/probe_common.py](../scripts/probes/probe_common.py). The scoring code must be identical for every
checkpoint, including step 0. Files, format and checks: [scripts/probes/README.md](../scripts/probes/README.md).

**Probe A — close to the competition.** The public SinhalaMMLU Easy questions with gold answers,
[data/SinhalaMMLU/TEST/easy](../data/SinhalaMMLU/TEST/easy/) (1,851 questions, 14 subjects).

- A fixed subset of 504 (36 per subject) for the frequent checks; all 1,826 usable questions at the epoch
  boundaries.
- 25 questions are left out: 18 are near-copies of shared-task test questions (same question, at least 2
  identical options; an exact-text match finds none, because of spacing and punctuation differences), and 7 have
  two identical options. One gold answer stored as the option text ("24") is mapped to its number.
- Zero-shot by default. If the official few-shot examples are available (not in the repo now), 3-shot makes a
  base model's answers more stable.
- **Monitoring only: never train on these questions.**

**Probe B — did the textbook facts go in.** 305 MCQs written once by Claude (`claude-opus-5-5`) from 154
textbook sections that are **in training**, plus 100 from the **19 held-out chapters** as a control (4 options,
the same prompt format). Each question carries the quote it was written from; the check script verifies the quote
is in its section and that no held-out quote also appears in a training section.

| Result | Meaning |
|---|---|
| "in training" rises, "held-out" doesn't | knowledge injection works |
| both rise by the same amount | only general Sinhala improved, not the facts |
| neither moves | CPT is not adding knowledge → stop and rethink (LR, epochs, rewording) |

- It is the only signal for Medium/Hard, which have no labelled dev set: make Probe B for those levels when their
  textbooks are added.
- A Sinhala reader spot-checks a sample of the generated questions.
- Disclose the closed API if one is used. **Never put probe questions into SFT data.**

**Noise:** with 500 questions the standard error is ~2 points, so differences under ~4–5 points mean nothing.
With 1,851 it is ~1 point.

### 8.4 Fixed sample outputs

At every probe, the model continues 5 fixed Sinhala prompts (greedy, ~100 tokens), logged as a W&B table.
Broken script, wrong language or endless repetition show up at once.

### 8.5 Evaluation schedule

| What | When | Cost |
|---|---|---|
| 4 validation losses | step 0, then every 25 steps | ~30 s |
| Probe A (500) + Probe B + samples | step 0, then every 50 steps | ~2–3 min |
| Probe A (all 1,851) | epoch boundaries (117, ~234, end) | ~5 min |

### 8.6 Stop points

| Check | When | Pass if | If not |
|---|---|---|---|
| **Smoke test** | first ~10 min | first training loss close to the step-0 validation loss (≈ 12.4 = ln 248,320 means random guessing → data/label bug); no kernel-fallback warning; a few GB of GPU memory still free; tokens/s gives an ETA | fix the setup |
| **Check 1** | ~step 70 (20%) | textbook loss down; replay losses within ~+5% of step 0 | stop, change the LR |
| **Check 2** | step 117 (end of epoch 1) | Probe B "in training" up by more than the noise; Probe A not down by more than the noise | stop, rethink |

A failed check costs ~$1 instead of a full run.

### 8.7 The early dip is normal

When a trained model starts training again, general performance often gets slightly *worse* for the first
~10–20% of steps and then recovers (the "stability gap"). Replay losses and Probe A may dip there. That is why
Check 1 is at 20% and not earlier.

### 8.8 The deciding test, after the run

A raw base model's MCQ score is noisy, and CPT can change its answer habits even when its knowledge improves.
The real question is whether **CPT + SFT beats Base + SFT**:

1. Make a small SFT set (synthetic MCQs from the *training* textbook sections, separate from Probe B).
2. Apply the **same** short LoRA SFT (~20 minutes) to the base model and to the CPT model.
3. Compare both on Probe A and Probe B.

Only if the CPT model wins is the large final run worth paying for. The score to beat after that is the zero-shot
`Qwen/Qwen3.5-4B` (instruct) baseline from the baseline notebooks.

---

## 9. Traps and tips

- [ ] **Labels:** `labels = input_ids`; no `DataCollatorForLanguageModeling` (pad = EOS = separator, section 4.1).
- [ ] **Kernels:** `flash-linear-attention` + `causal-conv1d` installed; no fallback warning (section 4.4).
- [ ] **Rounding:** `adamw_torch_8bit` with `bf16_stochastic_round=True`, not a bitsandbytes optimizer (3.4).
- [ ] **Order:** `train_sampling_strategy="sequential"` (default is random) (4.2).
- [ ] **Tied embeddings:** before saving, check that the parameter count is unchanged and that `lm_head.weight`
      and `embed_tokens.weight` are still the same tensor. If they are split, the model silently grows by 0.64B.
- [ ] **Reload test:** load each saved model with the evaluation code before deleting anything.
- [ ] **Base-model scoring:** plain text, no chat template, the right digit token (8.3).
- [ ] **Training loss is not progress** on repeated textbooks (8.2).
- [ ] **Early dip** is expected (8.7).
- [ ] **tmux**, never notebooks, for training (6).
- [ ] **Debug off the GPU meter:** make the script work end to end with the smallest Qwen3.5 Base model and
      256-token blocks on Kaggle, Colab or CPU; start the g6e only when it runs.
- [ ] **Reproducibility record** in every W&B run (7).
- [ ] **Free fix afterwards:** if general skills dropped, average the last checkpoints or blend the CPT model with
      the base (e.g. 0.8 × CPT + 0.2 × base, chosen by validation loss). No training needed.

---

## 10. The machine: AWS g6e.xlarge

### 10.1 Why this machine

| | g6e.xlarge |
|---|---|
| GPU | 1× NVIDIA L40S, 48 GB — enough for full-parameter CPT of this model (section 3.3) |
| CPU / RAM | 4 vCPUs / 32 GB — enough, because the data is already tokenised |
| Price | **$1.861/h** On-Demand (us-east-1), billed per second → $100 of credits ≈ **53 GPU-hours** |
| Value | more compute per dollar than g6 (L4, 24 GB), and big enough for full training |

### 10.2 Do these first (they can take days)

1. **GPU quota.** "Running On-Demand G and VT instances" starts at **0 vCPUs** on new accounts.
   - Service Quotas → Amazon EC2 → *Running On-Demand G and VT instances* (code `L-DB2E81BA`) →
     request **4** (g6e.xlarge = 4 vCPUs), **in the region you will use**.
   - The Spot quota ("All G and VT Spot Instance Requests") is separate; we don't need it.
   - Approval can take up to ~48 hours, and new accounts are sometimes refused. Give a concrete reason, e.g.
     "one g6e.xlarge, about 40 hours, fine-tuning a 4B open model for an academic shared task".
   - If it is refused, fall back to RunPod, Vast.ai or Lambda (L40S/A100 by the hour).
2. **Free plan vs Paid plan.** Accounts created since July 2025 start on a "Free plan" that blocks some services.
   It is not clear whether it allows GPU instances. If the launch or the quota request is refused, upgrade to the
   Paid plan: credits are still used first, but after that you pay — so set up the budget alerts (10.6).

### 10.3 Launch settings

| Setting | Value |
|---|---|
| Region | One that offers g6e and has the approved quota. us-east-1 is the cheapest with the most capacity; ~250 ms from Sri Lanka is fine for VS Code Remote-SSH. |
| Machine image (AMI) | "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu 22.04 or 24.04)": GPU driver, CUDA and PyTorch installed. Login user `ubuntu`. |
| Instance type | `g6e.xlarge`, purchasing option **On-Demand** (the default) |
| Key pair | new ED25519 key, `.pem` (10.4) |
| Storage | root volume **gp3, 200 GB** (model cache, 2 full checkpoints, 3 epoch copies, environment) |
| Security group | inbound SSH (22) from **My IP** only |
| Advanced → Shutdown behavior | **Stop** (never Terminate) |
| Advanced → Termination protection | **On** |
| Elastic IP (optional) | ~$0.005/h (~$0.12/day): the address stays the same after every restart |

### 10.4 SSH from VS Code (Windows)

1. EC2 → Key pairs → create (ED25519, `.pem`) → save as `C:\Users\Asus\.ssh\sinhala-aws.pem`. Windows OpenSSH
   refuses keys that other users can read, so in PowerShell:
   ```powershell
   icacls "$env:USERPROFILE\.ssh\sinhala-aws.pem" /inheritance:r /grant:r "$($env:USERNAME):R"
   ```
2. Test from PowerShell: `ssh -i $env:USERPROFILE\.ssh\sinhala-aws.pem ubuntu@<Public IPv4 DNS>`, then
   `nvidia-smi` must show the L40S.
3. Install the VS Code extension **Remote - SSH** (Microsoft), and add to `C:\Users\Asus\.ssh\config`:
   ```
   Host sinhala-gpu
       HostName <Public IPv4 DNS or Elastic IP>
       User ubuntu
       IdentityFile C:\Users\Asus\.ssh\sinhala-aws.pem
       ServerAliveInterval 30
       ServerAliveCountMax 10
   ```
4. `F1` → **Remote-SSH: Connect to Host…** → `sinhala-gpu` → Linux → open `/home/ubuntu`. Install the Python
   extension "in SSH: sinhala-gpu".
5. Without an Elastic IP, the public DNS changes on every start: update `HostName`. If SSH suddenly times out,
   your home IP has probably changed: update the security-group rule ("My IP").

### 10.5 First-time setup on the machine

1. `git clone https://github.com/kanishkapg/Sinhala-LLM.git` (private repo: use a GitHub fine-grained token or
   `gh auth login`). The packed dataset is committed, so it comes along.
2. A virtual environment with the pinned versions: `torch` (the AMI's), `transformers==5.19.0`, `accelerate`,
   `datasets`, `torchao`, `flash-linear-attention`, `causal-conv1d`, `wandb`.
3. `hf auth login` and `wandb login`. Keep tokens in environment variables, never in the repo.
4. `tmux new -s train` for every run.

### 10.6 Paying only while training

- **On-Demand bills per second, only while the instance is running.** Stop it whenever it is not training, and
  write or debug code on your own machine, not on the GPU.
- **Still billed while stopped:** the EBS volume (200 GB ≈ $16/month ≈ $0.53/day), an Elastic IP if you made one
  (~$0.12/day), S3. When the project is done, push the final weights to Hugging Face, then terminate the instance
  and delete the volume.
- **Stop automatically when a run ends or crashes:** start it in tmux as
  `python train.py ... ; sudo shutdown -h now`. With shutdown behavior = Stop, the machine stops and the disk stays;
  after a crash you resume the next time.
- **Safety net if you forget:** a CloudWatch alarm on the instance — `CPUUtilization` < 5% for 60 minutes →
  *Stop this instance* (EC2 console → instance → Actions → Monitor and troubleshoot → Manage CloudWatch alarms).
  Training keeps one CPU busy, so it won't fire during a run.
- **AWS Budgets:** alerts at $25, $50 and $75 with **credits excluded** from the budget, so they track the real
  spending. Billing → Credits shows what is left.
- **Downloads:** the first 100 GB per month out of AWS are free. Move weights-only models (8.4 GB), never full
  optimizer checkpoints.
- **Spot** (~$0.87/h) is about half the price, but needs its own quota and can be interrupted. With the deadline,
  stay On-Demand.

### 10.7 Daily routine

```
edit code locally → git push → start the instance → connect from VS Code → git pull
→ tmux → python train.py ... ; sudo shutdown -h now → watch W&B → the instance stops itself
```

---

## 11. Budget

| Step | GPU-hours | ≈ Cost |
|---|---:|---:|
| Smoke test + resume test | 0.5 | $1 |
| Easy CPT (22.9M tokens) | 2–3 | $4–6 |
| Base vs CPT SFT comparison + probes (8.8) | 2–3 | $4–6 |
| Final cumulative CPT (all grades + replay, ~70–120M tokens) | 6–12 | $11–22 |
| Final SFT, evaluation, test predictions | 3–5 | $6–9 |
| Disk + IP, ~2 weeks | — | ~$9 |
| **Total** | | **~$35–55**, leaving ~$45 for reruns |

---

## 12. Order of work

1. **Today:** request the GPU quota; check the Free/Paid plan; set up the budget alerts.
2. Write the training script (settings of section 4, per-source eval, probe callback, epoch-copy callback,
   smoke-test mode). *Done: [scripts/train/](../scripts/train/README.md).*
3. Make Probe B (~300 + ~100 MCQs) and pick the Probe A subset. *Done: `data/probes/easy/`; the Sinhala
   reader review of `probe_b_review.csv` is still open.*
4. Debug the script end to end with a tiny Qwen3.5 model on Kaggle/Colab/CPU. *Done on CPU with a tiny random
   Qwen3.5 (`scripts/train/make_tiny_model.py`): train, eval, probes, gates, epoch copies, stop + resume, smoke mode.*
5. On the g6e: smoke test → resume test → full Easy run with the stop points of section 8.6.
6. Choose the epoch count from the epoch-boundary checkpoints (section 5).
7. Base vs CPT SFT comparison (section 8.8). Continue only if CPT wins.
8. Medium/Hard data → final cumulative CPT (from the base, or continuing the Easy run's pre-decay checkpoint).
9. SFT → evaluation → submission (see [cpt_overview.md](cpt_overview.md), stages 2–3).

---

## 13. Changes compared with cpt_overview.md

| Topic | cpt_overview.md said | Now |
|---|---|---|
| GPU | one 80 GB GPU for full training | one **48 GB** L40S, with 8-bit AdamW + stochastic rounding |
| Collator | `DataCollatorForLanguageModeling(mlm=False)` | **`labels = input_ids`** — that collator masks the `<|endoftext|>` separators |
| Schedule | cosine to 10% | **warm-up–stable–decay** (section 4.3) |
| Data order | shuffle on | **sequential** (section 4.2) |
| Checkpoints | every ~25% of the run, keep the best by validation loss | every 25 steps (keep 2) + weights-only copies at the epoch boundaries (section 6) |
| Global batch | 32–64 sequences | 32 sequences (more updates for a small dataset) |

---

## 14. References

- Biderman et al., 2024, "LoRA Learns Less and Forgets Less".
- Muennighoff et al., 2023, "Scaling Data-Constrained Language Models".
- Allen-Zhu & Li, "Physics of Language Models, Part 3.1: Knowledge Storage and Extraction".
- Guo et al., 2024, "Efficient Continual Pre-training by Mitigating the Stability Gap".
- Unsloth — Qwen3.5 fine-tuning guide: https://unsloth.ai/docs/models/qwen3.5/fine-tune
- VeOmni — Qwen3.5 training guide (FLA kernels): https://veomni.readthedocs.io/en/latest/examples/qwen3_5.html
- AWS — EC2 instance type quotas: https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html
- g6e.xlarge prices (third-party, check the AWS pricing page): https://compute.doit.com/spot/us-east-1/g6e.xlarge
