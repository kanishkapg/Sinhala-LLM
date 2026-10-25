# Continuing the Gemma model: the plan

Model: `sinhala-mmlu/gemma-4-e4b-sinhala-mmlu`, tag `v1.0.0` · Written 25 Oct 2026

Everything here is based on what we checked in the repo and in the weights (see `gemma/analysis/`).
Words in *italics* are explained in the glossary at the end.

---

## The plan in 10 lines

1. **Check the rules first.** This model was trained on 725 public SinhalaMMLU Easy questions, and some public
   questions are near-copies of test questions. Ask the organisers before relying on it.
2. **Measure before training.** Run it in *thinking mode* on our test sets, per level and per subject. Also run
   Google's original model to see what the first training actually added.
3. **Take the free win.** Pick the best way to run the model (thinking budget, voting). No training needed.
4. **Yes to SFT, a careful yes to CPT.** The model reasons well but learned almost no new facts. Facts must live
   in the weights, because retrieval is banned. So we add textbook knowledge (*CPT*) and practise answering (*SFT*).
5. **Do both in one LoRA run.** Mix textbook text with reasoning examples, so the model learns facts without
   forgetting how to think and answer. The model's own training script already supports this (`--docs`).
6. **Merge the LoRA, then blend with the current model.** Pick the blend by dev scores.
7. **Stay at 7.94B.** Anything that merges back into the weights costs nothing. Anything extra loaded at
   inference does (only 0.06B of room left).
8. **Keep `v1.0.0` as the fallback.** Submit a new model only if it beats `v1.0.0` on our dev sets.
9. **The test phase ends 27 Oct.** Steps 0–2 surely fit. One training round might fit. The rest is for later.
10. **Package it for the re-run.** The organisers re-run everything, so pin versions and seeds.

---

## 1. Where we start from

### What this model is (checked)

| Fact | Detail |
|---|---|
| Starting point | `google/gemma-4-E4B-it`, the **instruction-tuned** Gemma (not the plain base model) |
| Training done | **One** LoRA fine-tune (rank 64, 2 epochs, LR 5e-5), merged, then blended 50/50 with Google's model |
| Training data | 7,197 English reasoning traces for 4,980 MCQs (725 real public Easy questions + 4,255 generated from textbooks) |
| CPT done? | **No.** No textbook text was trained as plain text. |
| How much the weights moved | Only **0.1–0.4%** per weight matrix. Embeddings, vision and audio are unchanged. |
| Size | **7.94B** parameters. The limit is 8B. |
| Options | Only **4** options (fine: all shared-task test items have 4) |

### What it is good and bad at

| Good | Bad |
|---|---|
| **Reasoning.** Thinking mode beats direct answering by about 16–23 points on the same weights (67.3% vs 51.8% on its own Easy dev set). | **Knowledge.** It learned very few new facts. Reasoning can't recover a Sri Lankan history or dance fact the model never saw, and Humanities is about half the test. |
| Gemma's tokenizer reads Sinhala efficiently | Only Easy had real exam questions in training. "Hard" was only measured on translated MMLU. |
| One model, simple to run | Each answer needs hundreds of generated tokens (slow, and harder to reproduce exactly) |

### What nobody has measured yet

- What Google's untouched model scores on the same sets. So we don't know how much the first training helped.
- Scores per subject.
- Which 725 public questions were used.

---

## 2. The clock

- Today is **25 Oct**. The test phase ends **27 Oct** (check the exact closing time on Codabench).
- Up to **3 submissions per day**.
- Only a system we submitted before the end can be ranked. After that, the organisers re-run our code
  (28 Oct – 31 Nov).

| Step | Fits before 27 Oct? |
|---|---|
| 0 · Safety checks | **Yes, must do** |
| 1 · Measure | **Yes, must do** (about 1 GPU-hour) |
| 2 · Free accuracy (no training) | **Yes** |
| 3–5 · Training round 1 | **Maybe.** Only if the data and the run work the first time. |
| 6 · Later rounds | **No.** For the report, or if the deadline moves. |

> If we run out of time, **Steps 0–2 alone are a complete, valid result.** Training is a bonus, not a requirement.

---

## 3. Step 0: Safety checks (do these first)

### 0a. The 725 public questions (the biggest risk)

**Why it matters:** the rules say *"Training, tuning or prompt design on test questions is strictly prohibited."*
When we built Probe A, we found about **18 near-copies** of shared-task test questions inside the public Easy set.
If any of those are among the 725 this model trained on, the model has seen test questions.

**What to do:**

1. List the shared-task test questions that have a near-copy in the public Easy set (reuse the Probe A
   overlap check).
2. **Email the organisers today.** Say we want to use a model that was trained on public SinhalaMMLU Easy
   questions, some of which are near-copies of test items, and ask if that is allowed. Keep the reply.
3. If they say **no**, use **Plan B**: do this same plan, but start from `google/gemma-4-E4B-it`, which is
   clean. Step 1 shows how much we would lose. Since the first training moved the weights only 0.1–0.4%, the
   loss may be small.

### 0b. The parameter count

| Version | Parameters | Room under 8B |
|---|---|---|
| Full model (text + vision + audio) | **7.941B** | 0.059B |
| Text only (vision + audio removed) | 7.463B | 0.537B |

Where it goes: 3.97B transformer layers, 2.82B per-layer embedding table, 0.67B token embeddings,
0.31B audio, 0.17B vision. (Measured with `gemma/analysis/param_split.py`.)

**Golden rule: always merge LoRA into the weights before inference.** An unmerged adapter would push the model to about 8.09B, which is over the limit.

### 0c. Don't break what works

- Keep `v1.0.0` untouched. It is our fallback.
- Save new models under a **new tag or repo** (for example `v1.1.0`). Note that this HF repo is **public**.
- Never move or overwrite an existing tag.

---

## 4. Step 1: Measure before you train

### What to run

| Model | Mode | Why |
|---|---|---|
| `v1.0.0` | Thinking, greedy, 1 sample, budget 7168, `--context` | Its intended mode: our main baseline |
| `v1.0.0` | Thinking, 16 samples + option shuffling, majority vote | Its earlier submission recipe: does voting help? |
| `google/gemma-4-E4B-it` | Thinking, greedy, same settings | What did the first training add? (decides Plan B) |

**How:** use the repo's own `scripts/bakeoff_run.py` inside the Docker image `vllm/vllm-openai:v0.32.0-cu129`,
on Modal. It needs our files in its format, one JSON per line:
`{"id", "subject", "grade", "question", "choices": [4 strings], "answer": 1-4}`.

```bash
python3 scripts/bakeoff_run.py --model gemma4_e4b --hf-id <model or folder> \
  --inputs eval --sets shared_easy,shared_medium,shared_hard,probeb_heldout,gmmlu_dev \
  --out results --tag <name> --modes think --samples 1 --greedy --context \
  --think-budget 7168 --max-model-len 16384
```

**Look at:** accuracy per level and per subject, *parse rate* (should be ~100%), and average tokens per answer.
The weakest subjects tell us where training data should go.

### Which test sets are clean for this model

| Set | Clean for Gemma? | Use it for |
|---|---|---|
| Shared-task test (our *proxy key*, ~85–90% correct) | Never train on it | A final check of 1–2 finished candidates **only**. Choosing settings on test questions counts as "tuning on test questions". |
| Public SinhalaMMLU Easy (1,851) | **No.** 725 were in its training, so scores are inflated. | Don't use it to judge Gemma |
| Probe B held-out (100) | Yes | Does it know textbook facts? |
| Held-out part of our new SFT questions (split by chapter) | Yes | **Main dev set** for picking settings |
| Global-MMLU Sinhala | Yes, if we don't train on it | General reasoning, "Hard" stand-in |

> **Gap:** we have no clean Medium or Hard dev set yet. Global-MMLU is the stand-in until our own
> Medium/Hard questions exist.

---

## 5. Step 2: Free accuracy without training

These cost **0 parameters** and need no training:

| Setting | Effect | Cost |
|---|---|---|
| Thinking mode on | The biggest gain (+16–23 points vs direct) | Hundreds of tokens per question |
| Subject + grade in the prompt (`--context`) | Small gain | Free |
| Thinking budget 7168 tokens | In their runs, every answer finished within it | Free |
| 16 samples + shuffled options + majority vote | Usually a few points more | 16× the compute |

**How to choose:** use voting if it beats greedy by **2 or more points** on the dev sets. Otherwise use greedy,
because it is simpler and more reproducible. Either way, fix the seeds, pin the Docker image, and save every
prediction file.

---

## 6. Do we need CPT and SFT? (simple answers)

**Do we need SFT?** **Yes.**
- It is cheap, and this model was built for it (reasoning SFT).
- We can aim it at the model's weak spots: weak subjects, Medium/Hard, negation questions and "A and C" questions.
- After any CPT, we must re-practise the answer format anyway.

**Do we need CPT?** **Probably yes, but test it with one run first.**
- Retrieval is banned, so **facts must be inside the weights**.
- This model learned almost no new facts: weights moved only 0.1–0.4%, no textbook text was trained, and the
  embeddings didn't change.
- Thinking helps only when the fact is already known. It can't invent a missing fact.
- In our own Qwen run, textbook CPT **alone** took Probe A from **45.8% to 57.0%**. That is our strongest
  evidence that CPT adds knowledge.

**Why not CPT alone, with plain text only?** This is a chat/thinking model. Training it only on raw text can make it
forget how to chat, think and answer. So we **always mix reasoning examples into the CPT run.**

**Which order?** CPT before SFT, or both together. **Never SFT first and then CPT,** because that erodes the answer format.

**LoRA or full training?** **LoRA first.** It is cheap, the repo's script already does it, and it forgets less. Its
weakness is that it learns fewer new facts. If LoRA CPT helps too little, try full-parameter CPT later (Step 6).

---

## 7. Step 3: Build the training data

### 7a. CPT data (plain text, for knowledge)

| Source | Where / status | Use |
|---|---|---|
| Easy textbooks, grades 6–8, cleaned (3,060 sections) | Ready: `data/processed/cpt/easy/04_dedup/textbook.jsonl` | Yes |
| Easy CPT mix (textbooks ×3 + Sinhala Wikipedia + FineWeb-2 Sinhala + English FineWeb-Edu) | Ready: `data/processed/cpt/easy/mix/train.jsonl` (rows already have a `text` field) | **Use directly with `--docs`** |
| Medium textbooks, grades 9–11 | Raw PDFs in `data/raw/g9`–`g11`, not processed yet | Run `scripts/cpt` with `--level medium` (watch for new FM fonts) |
| Hard textbooks, grades 12–13 | **Not collected** | Download from edupub.gov.lk |
| Teacher guides, syllabi | Not collected | Nice to have |

Tips:
- For round 1, use **1–2 textbook copies, not 3** (filter `mix/train.jsonl` by its `epoch` field). LoRA plus
  mixed reasoning examples doesn't need 3 copies, and it saves time.
- Keep the replay text (Wikipedia, FineWeb) at about 30–50% of the text part. It protects general Sinhala and English skills.
- No re-packing needed: the Gemma script tokenizes the text itself.

### 7b. SFT data (questions + English reasoning)

Training row format (what `scripts/sft_lora.py` reads):

```json
{"subject": "History", "grade": 7, "question": "...", "choices": ["...", "...", "...", "..."], "answer": 2,
 "reasoning_en": "Short English reasoning that ends at the right option."}
```

**Where the questions come from:**

| Source | Status | Notes |
|---|---|---|
| Our SFT pipeline (`scripts/sft`): Claude-written, blind-validated, from our textbooks | 86 final items now; more as the waves finish | Best quality. Slow (Claude Pro limits). |
| An open teacher model writing MCQs from textbook sections (for example Gemma 4 26B-A4B) | To build | Much faster on a GPU. Still run our checker, validator and dedup steps. |
| Global-MMLU Sinhala (public, translated MMLU) | Download from HF (`CohereForAI/Global-MMLU`, `si`) | Good reasoning practice for STEM and "Hard". Keep a **separate** 1,000 aside for evaluation. |

**Where the reasoning comes from:**

1. **The model itself (*self-distillation*).** Ask `v1.0.0` each question 4–8 times in thinking mode
   (`bakeoff_run.py --samples 4 --save-text`, no `--greedy`). Keep **only the correct** answers' reasoning.
   Cheap, and it matches the model's own style.
2. **A bigger teacher, for questions the model never gets right.** Gemma 4 31B or 26B-A4B, shown the textbook
   passage (`bakeoff_run.py --reference`, with a `reference` field holding the section text). Keep only correct
   traces. This is how textbook facts get *into* the reasoning.
3. **Fallback:** the `explanation_en` field our SFT pipeline already writes.

**What to give more weight to:** the weak subjects found in Step 1, negation questions (`නොවන`, `වැරදි`, …),
"A and C" combination questions, Medium/Hard, and exactly 25% of answers on each position.

**Traps:**
- Strip Gemma's channel markers (the `<|channel>` … `<channel|>` thinking wrapper) from saved thinking text
  before using it as `reasoning_en`.
- Train and evaluate with **exactly** the same prompt. `sft_lora.py` imports `SYSTEM` and `user_message`
  from `bakeoff_run.py`, so keep them unchanged, and use `--context` in **both**.

### 7c. Never use

- Past papers, marking schemes, e-thaksalawa, pastpapers.wiki
- Shared-task test questions (even with our proxy key)
- Public SinhalaMMLU questions, for any new training
- Probe B items (keep them for evaluation)
- **Always** check new data for overlap with the test set and the public set (`scripts/sft/s06_dedup.py` does this).

---

## 8. Step 4: Training round 1 (one mixed run)

```
v1.0.0 ──LoRA on [textbook text + reasoning examples, mixed]──► merged ──blend with v1.0.0──► v1.1 candidate
```

### Starting settings (we adjust them by measuring)

| Setting | Value | Why |
|---|---|---|
| `--rank` / `--alpha` | 64 / 64 | Same as the first training |
| `--lr` | 2e-5 | Lower than the first run (5e-5). The model is already trained. |
| `--epochs` | 1 | Enough for round 1 |
| `--max-len` / `--doc-len` | 3072 / 2048 | Same as the first training |
| `--micro-batch` / `--batch-tokens` | 1 / 16384 | Fits one GPU |
| `--context` | on | Must match evaluation |
| `--ckpt-dir` | set it | Lets a stopped Modal job resume |
| Mix | About **half text, half reasoning** by tokens | The script prints both counts. Use `--qa-repeat` to balance. |

```bash
hf download sinhala-mmlu/gemma-4-e4b-sinhala-mmlu --revision v1.0.0 --local-dir gemma-v1
cd gemma-v1
python3 scripts/sft_lora.py --model . \
  --train ../round1/traces.jsonl \
  --docs ../round1/cpt_text.jsonl --doc-len 2048 \
  --out ../runs/r1 --ckpt-dir ../runs/r1/checkpoints \
  --epochs 1 --lr 2e-5 --rank 64 --alpha 64 \
  --micro-batch 1 --batch-tokens 16384 --max-len 3072 --context
```

- **Smoke test first** with `--limit 50`.
- **GPU:** one L40S 48 GB on Modal should fit (16 GB of weights, a small LoRA, gradient checkpointing).
  If it runs out of memory, use an A100 80 GB.
- **Watch:** the loss going down, no NaN, and the "too long" count staying small.

### Pass / fail gate (after the run)

| Check | Must be |
|---|---|
| Parse rate and "thinking closed" rate | ≥ 99% (the format is not broken) |
| Probe B held-out + our dev set | **Higher** than `v1.0.0` |
| Global-MMLU Sinhala | Not more than 2 points lower (general reasoning kept) |

**If it fails:** lower the LR, put more reasoning examples in the mix, or drop the text part and do SFT only.

---

## 9. Step 5: Merge and blend

1. The script merges the LoRA by itself (`runs/r1/merged`). The size stays at 7.94B.
2. **Blend** it with the current model: `new = v1.0 + a × (merged − v1.0)` for `a` = 0.3, 0.5, 0.7 and 1.0.
   Reuse `scripts/train/blend_weights.py`, changing its loader to `AutoModelForImageTextToText` (Gemma is a multimodal model class).
3. Pick `a` using the **dev sets**.
4. Check the 1–2 best candidates once on the proxy test sets. **Submit only if a candidate beats `v1.0.0`.**

Note: the repo's TRAINING.md says **not** to re-apply the original 50/50 blend with Google's model. We blend
with `v1.0.0`, which is different and fine.

---

## 10. Step 6: Later rounds (after the deadline, or if it moves)

In order of value:

1. **Medium and Hard knowledge.** Process grades 9–11, collect grades 12–13, then do one CPT data set with
   all grades + replay.
2. **More Medium/Hard questions** and more teacher reasoning for them.
3. **Repeat self-distillation:** train → ask again → keep the correct answers → train. Each round fixes more questions.
4. **DPO:** show the model a correct and a wrong reasoning for the same question, and teach it to prefer the correct one.
5. ***GRPO* (reinforcement learning):** reward 1 when the final answer is right. MCQs can be checked
   automatically, so no reward model is needed. It is strong for reasoning but more complex, so only do it
   after SFT is stable.
6. **Full-parameter CPT**, if LoRA CPT adds too little: freeze the 3.5B embedding tables and the vision/audio parts,
   and train the 3.97B transformer weights (our 8-bit optimizer recipe from the Qwen run, or an 80 GB GPU).
7. **Shorter thinking:** train on shorter correct traces. Cheaper, faster and easier to reproduce.

---

## 11. What training is allowed under the 8B limit?

| Method | OK? | Why |
|---|---|---|
| LoRA / QLoRA, **then merge** | ✅ | Merged into the same weights, so still 7.94B |
| Full fine-tuning / CPT / SFT | ✅ | Same weights, same size |
| Self-distillation, DPO, GRPO | ✅ | They only change the weights |
| Blending / averaging Gemma-4-E4B checkpoints | ✅ | The result is still one 7.94B model |
| Big teachers (Gemma 26B/31B, Claude, Codex) to **make data** | ✅ | Not used at inference. Closed models must be **disclosed**. |
| Thinking budget, voting, option shuffling | ✅ | 0 parameters |
| **Unmerged** LoRA on the full model | ❌ | 7.94 + ~0.15 ≈ 8.09B |
| A second model at inference (ensemble, verifier, reward model, router) | ❌ | No room (0.06B left) |
| A bigger base model (Gemma 4 26B, 31B) | ❌ | Far over 8B |
| Adding new Sinhala tokens | ⚠️ | Each token adds 13,312 parameters (2560 + 10752), so at most ~4,400. Not needed: Gemma already reads Sinhala well. |
| Removing the vision/audio parts (text only, 7.46B) | ⚠️ | Frees 0.54B, but only useful if we must load something extra. Check that vLLM loads it, and ask the organisers how they count it. |

---

## 12. Compute and cost (rough)

| Job | GPU | Rough time |
|---|---|---|
| Evaluate the 2,238 shared-task questions, thinking, greedy | 1× L40S or A100 | 10–20 min (×16 with voting) |
| Self-distillation traces: ~3,000 questions × 4 samples | 1× L40S | 30–60 min |
| Teacher traces (Gemma 4 31B or 26B-A4B) | 1× 80 GB GPU (A100/H100) | 1–2 h |
| Round 1 LoRA training (~10M tokens) | 1× L40S | A few hours |

Check the Modal credit first: $30/month free, and part of it already went on the Qwen CPT and the comparison runs.
Always smoke-test with `--limit` before a full run.

---

## 13. Risks and what we do about them

| Risk | What we do |
|---|---|
| The organisers reject the 725-question contamination | Plan B: same plan from `google/gemma-4-E4B-it` |
| New training makes the model worse | Keep `v1.0.0`. Use the gates in Step 4 and the blend in Step 5. Submit only real wins. |
| CPT breaks the thinking/answer format | Always mix in reasoning examples. Check the parse rate. Lower the LR. |
| The organisers' re-run gives different answers | Greedy or seeded voting, a pinned Docker image, saved predictions. The forced-answer fallback always gives an answer. |
| Going over 8B by accident | Always merge. Re-count with `gemma/analysis/param_split.py`. Never load a second model. |
| Running out of time | Steps 0–2 alone are a full result |
| Modal credit runs out | Smoke tests, small eval subsets, one training run at a time |
| Wrong teacher reasoning | Keep only traces with the correct final answer, on validated questions |
| Proxy-key noise | Differences under ~3 points on the shared sets are not real |

---

## 14. Timeline and checklist

**25 Oct (today)**
- [ ] Email the organisers about the 725 public questions (Step 0a)
- [ ] List the test near-copies in the public Easy set
- [ ] Write the converter to the `bakeoff_run.py` format and a Modal runner (vLLM 0.32 image + `peft==0.21.2`)
- [ ] Measure `v1.0.0` (greedy and 16 votes) and `google/gemma-4-E4B-it` (Step 1)

**26 Oct**
- [ ] Choose the inference recipe and **submit `v1.0.0` with it** (the free win, Step 2)
- [ ] Build the round 1 data: text from `mix/train.jsonl` + self-distilled reasoning (Step 3)
- [ ] Smoke test, then the full round 1 run (Step 4)

**27 Oct (last day)**
- [ ] Gate check, blend sweep, pick the best (Step 5)
- [ ] Submit only if it beats `v1.0.0` on the dev sets
- [ ] Freeze the final system: model tag, Docker image, commands, seeds, prediction files

**After 27 Oct (verification + report)**
- [ ] Reproduction package: model manifest, scripts, data manifests, exact commands
- [ ] Report: cite the SinhalaMMLU paper and **disclose all closed-model use** (theirs: Codex/gpt-6.1-sol and
      Claude Sonnet; ours: Claude for SFT questions and Probe B)
- [ ] Step 6 ideas, if there is time

### Scripts we still need to write (all under `gemma/`)

| Script | Job |
|---|---|
| `to_bakeoff_jsonl.py` | Convert our test and dev files to the `bakeoff_run.py` format |
| `modal_gemma.py` | Run `bakeoff_run.py` and `sft_lora.py` on Modal (image `vllm/vllm-openai:v0.32.0-cu129` + peft) |
| `make_traces.py` | Turn `--save-text` outputs into `reasoning_en` training rows (keep correct ones, strip markers) |
| `find_overlap.py` | Find near-copies between the shared-task test and the public Easy set |
| blend step | Adapt `scripts/train/blend_weights.py` to load Gemma |

---

## 15. Glossary

| Word | Meaning |
|---|---|
| **CPT** (continued pre-training) | Training on plain text, such as textbooks, so the model *learns facts* |
| **SFT** (supervised fine-tuning) | Training on question → answer examples, so the model *learns the task* |
| **LoRA** | Training a small add-on instead of the whole model. Cheap. Merged into the weights afterwards. |
| **Merge** | Folding the LoRA add-on into the main weights, so no extra parameters remain |
| **Blend** (WiSE-FT) | Averaging two versions of the same model's weights, to keep the best of both |
| **Thinking mode** | The model writes its reasoning first, then `Answer: N`. Switched on with `enable_thinking=True`. |
| **Direct mode** | The model answers immediately, with no reasoning |
| **Self-distillation** | The model answers many times. We keep its correct answers and train on them. |
| **Teacher** | A bigger model used only to make training data, never at test time |
| **Replay** | General text mixed into CPT so the model doesn't forget its general skills |
| **Parse rate** | The share of answers where we could read a clear `Answer: N` |
| **Proxy key** | Our own best-guess answers for the unlabelled test set (~85–90% right) |
| **DPO** | Teaching the model to prefer a good answer over a bad one |
| **GRPO** | Reinforcement learning: the model tries, and gets a reward when the final answer is right |
