# SinhalaMMLU Shared Task (ICTer 2026) — Project Plan

A complete, stage-by-stage plan: what the task is, what the rules force you to do,
what to build first, and how to finish with a reproducible, ranked submission.

---

## Table of contents

1. [The task in one page](#1-the-task-in-one-page)
2. [Rules that shape every design decision](#2-rules-that-shape-every-design-decision)
3. [What we already decided, and why](#3-what-we-already-decided-and-why)
4. [Traps that can cost you the competition](#4-traps-that-can-cost-you-the-competition)
5. [Phase 0 — Setup and blocking questions](#phase-0--setup-and-blocking-questions)
6. [Phase 1 — Evaluation harness (build this first)](#phase-1--evaluation-harness-build-this-first)
7. [Phase 2 — Base model selection](#phase-2--base-model-selection)
8. [Phase 3 — First real submission (SFT-only)](#phase-3--first-real-submission-sft-only)
9. [Phase 4 — Curriculum knowledge (CPT)](#phase-4--curriculum-knowledge-cpt)
10. [Phase 5 — Final model (CPT → SFT)](#phase-5--final-model-cpt--sft)
11. [Phase 6 — Free accuracy at inference time](#phase-6--free-accuracy-at-inference-time)
12. [Phase 7 — Packaging, reproduction and report](#phase-7--packaging-reproduction-and-report)
13. [Repository structure](#13-repository-structure)
14. [Parameter budget reference](#14-parameter-budget-reference)
15. [Experiment log template](#15-experiment-log-template)
16. [Reference links](#16-reference-links)

---

## 1. The task in one page

**Given a Sinhala question and its answer options, predict the correct option number.**

The questions come from Sri Lankan national and provincial exam papers, curated natively
in Sinhala (not translated). Each question carries a difficulty label:

| Level  | Grades | Exam stage      | Options | Count in full dataset |
|--------|--------|-----------------|---------|-----------------------|
| Easy   | 6–8    | Junior secondary| 4       | 1,893                 |
| Medium | 9–11   | O-Level         | 4       | 2,585                 |
| Hard   | 12–13  | A-Level         | 5       | 2,566                 |

Full dataset: **7,044 questions, 6 domains, 30 subjects.**

Domains: Humanities (3,419), Social Science (1,084), Other (1,038), STEM (629),
Business Studies (477), Language (397). Humanities dominates — and it is the hardest
domain for models, because it is culturally grounded (Sri Lankan history, dance,
oriental music, Buddhism, drama).

### Scoring

```
AccE  = accuracy on Easy
AccM  = accuracy on Medium
AccH  = accuracy on Hard
Macro = (AccE + AccM + AccH) / 3      ← this is the ranking metric
```

**Hard is worth a full third of your score.** Always look at per-level accuracy, never
just the overall number. Chance level is 25% on 4 options and 20% on 5.

### Submission format

`predictions.jsonl`, zipped as `predictions.zip`, one JSON object per line:

```json
{"q_no": "1", "subject": "Arts", "category": "humanities", "level": "easy", "answer": "2"}
{"q_no": "2", "subject": "Arts", "category": "humanities", "level": "easy", "answer": "4"}
```

Up to **3 submissions per day** during the test phase.

### Official phases

| Phase        | Dates              | What happens |
|--------------|--------------------|--------------|
| Development  | from 05 Oct        | Build on the dev set with the public baseline |
| Test         | 12 – 27 Oct        | Submit to the live Codabench leaderboard |
| Verification | 28 Oct – 31 Nov    | Organisers re-run your code; only reproduced runs are ranked |

### The five official steps

1. Register your team (1–4 members, one corresponding member).
2. Develop on the dev set with the public baseline.
3. Submit test predictions on Codabench.
4. Share your system and code to finalise your submission.
5. Present your work at ICTer 2026.

---

## 2. Rules that shape every design decision

### 2.1 The 8B parameter limit

- The **total parameters of every component used at inference** must not exceed 8 billion.
- Counts every neural component you load: generators, ensemble members, classifiers, routers.
- **MoE models count total parameters**, not active parameters.
- **LoRA and other adapters count in addition to their base model.**
- **Quantisation does not reduce the parameter count.**

### 2.2 Models

- Only **open-weight models released before 1 October 2026**.
- **No closed APIs** (GPT, Claude, Gemini) at inference time.

### 2.3 Training data

- Any publicly available data may be used for training.
- Synthetic data generated with open models is permitted.
- Closed APIs may be used **only to generate training data**, and this **must be disclosed**.
- **Training, tuning or prompt design on test questions is strictly prohibited.**

### 2.4 Inference — the rule that decides your whole architecture

- No internet access or live retrieval at inference.
- **No retrieval of any kind.** No RAG, knowledge bases, vector databases, search indexes,
  dictionaries, corpora or lookup tables — local or remote, neural or sparse (including BM25).
- The full pipeline must run from submitted code with **no human in the loop**.

> **Consequence:** the only place knowledge can live is *in the model weights*.
> This makes **fine-tuning your main lever**, not prompt engineering. Everything in
> Phases 4 and 5 follows from this single rule.

### 2.5 Reproducibility

- All ranked teams share code, scripts and a model manifest after the final deadline.
- Organisers re-run every final system. **Only reproduced runs enter the final ranking**
  and are eligible for prizes.
- All teams must submit a system description report.
- The report **must cite the SinhalaMMLU dataset paper** (Pramodya et al., EMNLP 2025).

> **Consequence:** your pipeline must be deterministic. This is why we score answers by
> probability instead of generating text — see Phase 1.

### 2.6 Integrity

The organisers may disqualify a team or revoke a prize for manual predictions, use of
test answers from any source, multiple Codabench accounts, or undisclosed closed-API use.
Anyone who has had access to the full dataset may not participate or advise teams.

---

## 3. What we already decided, and why

| Decision | Choice | Reason |
|---|---|---|
| Base model | **Gemma 3 4B** | Only small model with published Sinhala evidence. Its 262k vocabulary tokenizes Sinhala far more efficiently than Llama's. ~3.88B text-only leaves budget headroom. |
| Variant for zero-shot | **`gemma-3-4b-it`** | Paper shows instruction-tuned beats base in zero-shot for nearly every model family. Its chat template also makes each answer digit a single token. |
| Answer method | **Probability over option digits** | Deterministic (needed for verification), never fails to produce an answer, matches the paper's method for open models. |
| Prompt | **Official Sinhala template** from the task repo | Including the subject name improved performance in the paper; matching the template makes your scores comparable. |
| Adapter strategy | **One model, sequential stages** | Sinhala skill transfers across subjects; per-domain data is too small; no routing errors; easiest to reproduce. |
| Training order | **CPT first, then SFT** | SFT→CPT erodes the answer format. Knowledge is absorbed through pre-training-style text, not QA pairs. |
| Routing | **Rule-based, if any** | `subject`, `category` and `level` are given in the data. A neural router would cost parameters for nothing. |
| Rejected | Llama 3.2 3B + SinLlama tokenizer | 11,080 new token IDs with no trained embeddings; SinLlama's embeddings are 4096-dim (Llama-3-8B) and won't transfer to 3072-dim; would need ~300M tokens of CPT just to recover; base scores 22.14% (chance). |
| Rejected | Qwen3.5-9B, Llama-3.1-8B, SinLlama, Gemma 4 26B A4B | Over budget (9B / 8.03B / 8.12B / 25.2B total). |

### Backup base models (test before committing)

| Model | Counts as | Notes |
|---|---|---|
| Gemma 4 E2B | 5.1B total | Same 262k tokenizer, newer generation, Apache 2.0. Count 5.1B, **not** 2.3B "effective". |
| Qwen3.5-4B | ~4B LM (~5B with vision) | 201 languages, strong MMMLU/INCLUDE. Load text-only; disable thinking mode. |
| Gemma 4 E4B | 8B with embeddings | **Only if organisers confirm** and your measured text-only count is ≤ 8.0B. |

### Realistic expectations

| System | SinhalaMMLU average |
|---|---|
| Chance (4 options) | 25% |
| Qwen2.5-7B-chat | 27.23% |
| Llama-3.1-8B-chat | 25.28% |
| Aya-Expanse-8B | 22.62% |
| Gemma-3-4b-it (1,000-q sample, HelaLM card) | ~40.6% |
| GPT-4o | 62.95% |
| Claude 3.5 Sonnet | 67.65% |

Small open models from 2024 are essentially at chance. A modern base with a good Sinhala
tokenizer is already a large win; curriculum fine-tuning is the second.

---

## 4. Traps that can cost you the competition

### 4.1 Contamination — the one that gets teams disqualified

- The full benchmark is **not** fully public. Hugging Face (`naist-nlp/SinhalaMMLU`) gates
  access and the public portion is the **Easy test split only: 1,851 questions**
  (= 1,893 − 14 subjects × 3 few-shot examples). Medium and Hard are not public.
- The hidden test set is very likely drawn from the non-public portion.
- The dataset's primary source was **e-thaksalawa** (the Ministry of Education platform)
  plus other sites hosting government past papers and marking schemes.
- Therefore **"scrape O/L and A/L past papers for training" is the obvious idea and the
  most dangerous one.**

**Rules to follow:**

- Do **not** train on the 1,851 public Easy questions. Use them as extra validation only.
  (License is CC BY-NC-ND 4.0 anyway.)
- Do **not** train on past papers, marking schemes or e-thaksalawa quiz content.
- **Do** train on textbooks and teacher guides — the source material, not the exams.
- Run n-gram overlap decontamination against every question you can see.
- **Ask the organisers in writing before using any of this.**

### 4.2 A model named "8B" is often not ≤ 8B

Llama-3.1-8B is ~8.03B. Qwen3-8B is ~8.2B. Both exceed a strict reading of the limit.
Never trust the name — measure it.

### 4.3 Gemma's "E" sizes are not what they look like

"E" means *effective*. Gemma 4 E4B is 4.5B effective but **8B with embeddings**;
E2B is 2.3B effective but **5.1B with embeddings**. Count the larger number.

### 4.4 Comparing answers with the wrong type

The official accuracy script converts both sides with `pd.to_numeric` before comparing,
which means `answer` may be stored as the string `"2"`. A naive
`predicted_number == item["answer"]` compares `int` to `str` and is **always False** —
you get 0% accuracy from a working model. Always `int()` both sides.

### 4.5 Few-shot prompting is not where the gains are

The paper found 3-shot gave limited or inconsistent improvement, and instruction-tuned
models sometimes got *worse*. Don't spend your time here.

### 4.6 Free-form chain-of-thought breaks reproduction

Sampled generations are hard to reproduce exactly on someone else's GPU, and a run that
doesn't reproduce is dropped from the final ranking. Score option tokens instead.

### 4.7 Sinhala text handling

- **Keep the zero-width joiner (U+200D).** Sinhala uses it for conjuncts (yansaya,
  rakaransaya). Stripping it corrupts the text.
- Normalise to Unicode **NFC**.
- Many Sri Lankan PDFs use **legacy non-Unicode Sinhala fonts**. If extracted text looks
  garbled, convert or OCR it (Tesseract `sin`) before training.

### 4.8 Question types that break models

The paper measured both:

- **Negation questions** (`සාවද්‍ය`, `නොගැළපේ`, `වැරදි`, `අසත්‍ය`) — every model scored worse.
- **Sub-option questions** ("A හා C", ~9.8% of data) — every model scored worse.

Track these as separate buckets in your error analysis and over-generate them in SFT data.

---

## 16. Reference links

**Task**
- Task overview — https://sinhalammlu.ucsc.cmb.ac.lk/task.html
- Rules — https://sinhalammlu.ucsc.cmb.ac.lk/rules.html
- Evaluation — https://sinhalammlu.ucsc.cmb.ac.lk/evaluation.html
- Submission — https://sinhalammlu.ucsc.cmb.ac.lk/submission.html
- FAQ — https://sinhalammlu.ucsc.cmb.ac.lk/faq.html

**Dataset and paper**
- Paper (ACL Anthology) — https://aclanthology.org/2025.emnlp-main.1673/
- Paper (arXiv HTML) — https://arxiv.org/html/2509.03162v1
- Official code — https://github.com/naist-nlp/SinhalaMMLU
- Dataset (gated) — https://huggingface.co/datasets/naist-nlp/SinhalaMMLU

**Models**
- Gemma 3 4B — https://huggingface.co/google/gemma-3-4b-it
- Gemma 4 — https://huggingface.co/google/gemma-4-E4B
- Qwen3.5-4B — https://huggingface.co/Qwen/Qwen3.5-4B
- HelaLM-4B (Sinhala Gemma fine-tune, useful reference) — https://huggingface.co/Jainamshahhh/helalm-4b
- SinLlama — https://huggingface.co/polyglots/SinLlama_v01

**Data sources**
- EPD textbooks — https://edupub.gov.lk
- Cleaned Sinhala corpus — https://huggingface.co/datasets/polyglots/MADLAD_CulturaX_cleaned

---

