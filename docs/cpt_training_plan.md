# Training Plan — CPT → SFT for `Qwen/Qwen3.5-4B-Base`

This is a high-level plan, not training code. It uses the dataset built by `scripts/cpt/` (see
[prepare_cpt_dataset.md](prepare_cpt_dataset.md) and [scripts/cpt/README.md](../scripts/cpt/README.md)).

---

## 1. Which kind of training for the CPT dataset?

**Use continued pre-training with a causal language-modelling loss on the packed blocks.** This means:
- plain next-token prediction on every token;
- no chat template and no prompt masking;
- `labels = input_ids`.

This is the same objective the base model was pre-trained with. It is **not** instruction fine-tuning.

There are two ways to update the weights:

| | **A. Full-parameter CPT** (recommended) | **B. LoRA CPT, then merge** (when GPU memory is short) |
|---|---|---|
| What is trained | All 4 B parameters | Low-rank adapters on every linear layer (q, k, v, o, gate, up, down), plus `embed_tokens` and `lm_head` trained fully |
| Why | CPT's job is to add new knowledge and a new language register. Full fine-tuning absorbs new knowledge best. | LoRA learns less new knowledge but also forgets less ("LoRA Learns Less and Forgets Less", Biderman et al., 2024). A high rank (r = 128–256) and trainable embeddings close most of the gap. |
| GPU memory (bf16, 2,048 tokens, gradient checkpointing) | About 60–80 GB (weights + gradients + AdamW states): 1× A100/H100 80 GB, or 2× 40–48 GB GPUs with FSDP / DeepSpeed ZeRO-3. An 8-bit / paged AdamW lowers it. | About 16–24 GB: 1× L4 / A10 / RTX 4090 / A100 40 GB |
| Starting learning rate | 1e-5 – 2e-5 peak, cosine decay to 10 %, 3–5 % warm-up | 1e-4 – 2e-4 for the adapters; 1e-5 for `embed_tokens` / `lm_head` |
| After training | The checkpoint *is* the new base model | `merge_and_unload()` → a normal full model (save it in bf16) |

Recommendation:
1. Rent one 80 GB GPU for the CPT runs and do full-parameter CPT. A run is short (section 2).
2. Use LoRA only if you can't get such a GPU.
3. In either case, keep the result as a **merged, full model**. SFT starts from it, and the competition re-runs it.

Your local GPU (MX330, 2 GB) can't train a 4 B model. Use it for the data pipeline, which only needs a CPU. Train on Colab Pro (A100), Kaggle, RunPod, Lambda, Vast.ai or a university GPU.

### Other settings (both options)

| Setting | Starting value | Note |
|---|---|---|
| Sequence length | 2,048 (= the packed block size) | Use 4,096 only if you re-pack with `--block 4096` |
| Global batch | 32–64 sequences ≈ 65 k–130 k tokens per step | ~20 M tokens → ~150–300 optimizer steps |
| Passes over the packed data | **1** | Repetition is already built in (textbooks ×3 inside the mixture) |
| Optimizer | AdamW, β = (0.9, 0.95), weight decay 0.1, gradient clip 1.0 | |
| Precision | bf16 | T4 / V100 have no bf16: prefer an Ampere or newer GPU |
| Shuffle | on | The packed train set already mixes sources inside blocks |
| Checkpoints | every ~25 % of the run; keep the best by validation loss | |
| Library | HF `transformers` Trainer with `DataCollatorForLanguageModeling(mlm=False)`, or LLaMA-Factory (`stage: pt`), or Unsloth (its CPT recipe trains embeddings + LoRA) | Check that your library version supports Qwen3.5 (your baseline notebooks already use `transformers` 5.x) |

---

## 2. How much compute? (and why the huge corpora are not a problem)

Training compute depends on **how many tokens you train on**, not on how big the source corpus is.
Rule of thumb: FLOPs ≈ 6 × parameters × tokens.

| Run | Tokens | FLOPs | 1× H100 (~40 % MFU) | 1× A100 80 GB |
|---|---:|---:|---:|---:|
| Easy CPT (this dataset) | ~20 M | 4.8 × 10¹⁷ | ~20–30 min | ~1–1.5 h |
| Final cumulative run (all grades + replay) | ~60–100 M | ~2 × 10¹⁸ | ~1.5–2.5 h | ~4–7 h |
| Optional Sinhala language-adaptation stage | 200–500 M | 5–12 × 10¹⁸ | ~4–9 h | ~12–28 h |

**The general Sinhala corpora are much smaller than the figures you saw, and you only need a slice of them:**
- **Sinhala Wikipedia:** `wikimedia/wikipedia`, config `20231101.si`, is **54 MB**: 23,065 articles, ~290 MB as text. The 78 GB figure is the whole multilingual repository.
- **FineWeb-2 Sinhala:** `HuggingFaceFW/fineweb-2`, config `sin_Sinh`, is **2.0 GB**: 1.19 M documents, ~11 GB as text. The 2 TB figure is all languages.
- **What `s07_replay.py` does:** it doesn't download even these. It opens the Sinhala Parquet files remotely, reads random row groups, filters them, and **stops at the token budget** in `config/mixture.json` (Easy: 4 M Wikipedia + 5 M FineWeb-2 + 3 M English tokens).
- **For more general Sinhala:** raise the budgets for a separate *language-adaptation* stage. Use **quality over quantity**: Wikipedia and FineWeb-2 first, HPLT / MADLAD-clean only after filtering.

---

## 3. The plan, stage by stage

```
 Qwen3.5-4B-Base
   │  (optional) Stage 1a: Sinhala language adaptation — 200–500 M tokens general Sinhala + 15 % English
   ▼
 Stage 1b: curriculum CPT — textbooks (×3) + replay           ← dataset from scripts/cpt
   │   Easy run first (pipeline check), final model = one cumulative run on all grades
   ▼
 Stage 2: SFT — teach the answer format (MCQ → option digit) + Sinhala instructions
   ▼
 Stage 3: (optional) merging — recover general skills / add instruction-following
   ▼
 Evaluation with scripts/eval_baseline.py → submission
```

### Stage 1a — Sinhala language adaptation (optional, do it if budget allows)
- **Why:** Qwen3.5 reads Sinhala at only ~1.3 characters per token and has seen little Sinhala.
- **What:** a larger general-Sinhala CPT, done once, improves fluency and reading for every later stage.
- **Data:** same scripts, larger `replay_tokens`, and no textbooks (or textbooks at a low share).
- **Settings:** same as section 1, LR 1e-5 – 2e-5.

### Stage 1b — curriculum CPT (the dataset you are building)
- **Easy run first:** use it to validate the whole pipeline end to end (data → training → evaluation). Look at:
  - validation loss on the held-out textbook chapters (should fall clearly);
  - validation loss on the replay slices (Wikipedia, FineWeb-2, FineWeb-Edu). It must not rise by more than ~5–10 %. If it does, the model is forgetting: lower the LR or the textbook epochs, or add more replay.
- **For the final model, prefer one cumulative run:** all grades' textbooks + replay, starting from the base (or from 1a).
- **Chained runs (Easy → Medium → Hard) also work** if every run keeps the earlier grades in its mixture (`mixture.json` already does this). Never train on new grades alone.

### Stage 2 — SFT (after CPT, never before)
- **Goal:** answer a SinhalaMMLU-style question with the right option number, using the official prompt template from `scripts/eval_baseline.py`.
- **Data:**
  - **Synthetic MCQs generated from the textbook sections in `docs.jsonl`:** question + 4 or 5 options + answer, made by a strong LLM. Generating them with a closed API is allowed but **must be disclosed**.
  - **General Sinhala instruction data:** for example, translated or native Sinhala instruction sets.
  - **Never:** past papers, e-thaksalawa, or the test questions.
- **Method:** LoRA SFT is enough here, because format learning is low-rank. Then merge it.
  - Loss on the answer tokens only.
  - 1–3 epochs.
  - LR ~1e-4 (LoRA).

### Stage 3 — merging ideas (optional, cheap, often helpful)
- **LoRA merge:** `merge_and_unload()` after each LoRA stage gives a single full model.
- **CPT / base interpolation:** if general skills dropped, average the weights, e.g. 0.8 × CPT + 0.2 × base (linear or SLERP, e.g. with `mergekit`). Choose the weight by validation loss.
- **"Chat vector":** add (Qwen3.5-4B instruct − Qwen3.5-4B-Base) to the CPT model. This gives instruction-following without much SFT data. It only works if both models share the architecture and tokenizer; check this first.

---

## 4. Evaluation and checkpoint selection

- **During CPT:** pick checkpoints by **validation loss** (textbook val ↓, replay val stable). Don't pick them by test accuracy.
- **After CPT** (base-style model): run a zero-shot multiple-choice evaluation with the official prompt, as plain text without the chat template, using the option-digit logits.
- **After SFT:** run `scripts/eval_baseline.py` as in the baseline notebooks and compare against the zero-shot baseline, per subject. Expect the biggest gains in History, Buddhism, Geography, Civics and Health.
  - Arts, Dancing, Drama and Eastern Music gain little until those textbooks are added.
- **Keep it reproducible:** record the seeds, the `manifest.json` of every dataset, the training configs and the checkpoint hashes. The competition re-runs your system.

---

## 5. Order of work (summary)

1. Build the Easy dataset (scripts/cpt; human review included).
2. Easy CPT run (full-parameter, ~20 M tokens) → check the validation losses → quick MCQ evaluation.
3. Medium and Hard datasets (same scripts, `--level medium/hard`).
4. *(Optional)* Sinhala language-adaptation run.
5. Final cumulative CPT (all grades + replay).
6. Synthetic-MCQ SFT (LoRA → merge).
7. *(Optional)* merging, then the final evaluation and the system report (licences, closed-API use and contamination numbers all come from the `manifest.json` files).
