# Preparing the SFT Dataset — Easy level (Grades 6–8)

Target model: **CPT e3** (`kanush21/qwen3.5-4b-si-cpt-easy`, from `Qwen/Qwen3.5-4B-Base`), and the base model for
the control run · Stage: supervised fine-tuning (SFT) on multiple-choice questions · Scope: the dataset only, not
the training code

This guide goes from the cleaned textbook sections of the CPT pipeline to a **tokenised, train-ready SFT dataset**
of Sinhala MCQs. For each step it says what to do, why, and how to check it worked. It also explains why we generate
the data the way we do: a planner, small grounded generation jobs, a blind validator, and dataset-level bias
audits.

- Input: `data/processed/cpt/easy/04_dedup/textbook.jsonl` (built by `scripts/cpt/`, see
  [prepare_cpt_dataset.md](prepare_cpt_dataset.md)).
- Training and evaluation of the SFT run: [prepare_cpt_training_pipeline.md](prepare_cpt_training_pipeline.md),
  section 8.8 (the base-vs-CPT deciding test), and [cpt_overview.md](cpt_overview.md), stage 2.
- Implementation: [scripts/sft/](../scripts/sft/README.md) (built and tested 2026-10-23). The run guide, the
  manual checks and the Pro-plan guards are in its README.
- Written 2026-10-23. The test phase ends 2026-10-27. Medium and Hard follow the same procedure later (section 17).

---

## Contents

1. [Decisions at a glance](#1-decisions-at-a-glance)
2. [What SFT must teach and what that means for the data](#2-what-sft-must-teach-and-what-that-means-for-the-data)
3. [What we measured](#3-what-we-measured)
4. [Pipeline overview and folder layout](#4-pipeline-overview-and-folder-layout)
5. [Step 1 — Freeze the inputs and the exclusions](#step-1--freeze-the-inputs-and-the-exclusions)
6. [Step 2 — The blueprint: subjects, chapters, sections, question types](#step-2--the-blueprint-subjects-chapters-sections-question-types)
7. [Step 3 — Item-writing rules and the style guide](#step-3--item-writing-rules-and-the-style-guide)
8. [Step 4 — Generation jobs](#step-4--generation-jobs)
9. [Step 5 — Assemble the options in code](#step-5--assemble-the-options-in-code)
10. [Step 6 — Automatic per-item checks](#step-6--automatic-per-item-checks)
11. [Step 7 — Blind validation by a second model](#step-7--blind-validation-by-a-second-model)
12. [Step 8 — Deduplicate and decontaminate](#step-8--deduplicate-and-decontaminate)
13. [Step 9 — Dataset-level bias audit](#step-9--dataset-level-bias-audit)
14. [Step 10 — Human review](#step-10--human-review)
15. [Step 11 — Split, render and tokenise (the train-ready format)](#step-11--split-render-and-tokenise-the-train-ready-format)
16. [How much data, waves, cost and time](#16-how-much-data-waves-cost-and-time)
17. [Later: Medium and Hard](#17-later-medium-and-hard)
18. [Changes compared with earlier plans](#18-changes-compared-with-earlier-plans)
19. [Final checklist](#19-final-checklist)
20. [References](#20-references)

---

## 1. Decisions at a glance

| Question | Decision | Section |
|---|---|---|
| Where do the questions come from? | **Synthetic MCQs grounded in our own cleaned textbook sections.** No past papers, no e-thaksalawa, no SinhalaMMLU questions. | 2.3 |
| Who writes them? | **Claude** (Opus) as the generator and **a different model** (Sonnet) as the blind validator, run through the Claude Code CLI on the **claude.ai Pro subscription** (no API billing, extra usage off, waits for the usage-limit reset). Closed model used only to make training data, **disclosed in the report**. | 8, 11 |
| How many per call? | **One source section per call, 1–6 questions** (short sibling sections merged, at most 8). Not dozens or hundreds per call. | 8.2 |
| Who decides coverage and question types? | A **deterministic planner** (code, seed 42) assigns every question slot to a section, a question type and a cognitive level **before** generation. | 6 |
| Who decides the answer position? | **Code, never the model.** The generator returns the correct answer and 3 distractors as separate fields. The code places the key from an exactly balanced schedule (25% per position, per subject and type). | 9 |
| How is the key checked? | A **blind** validator answers each question from the section without seeing the key. Only an exact match with all quality flags passing is accepted. | 11 |
| Bias control | Per-item checks, then **dataset-level gates**: position, longest/shortest option, stem–key word overlap, option convergence, and a choices-only probe. | 10, 13 |
| Size | Easy: **~7,800 generated → ~6,400 accepted**. The pilot (wave 1) is ~3,200 accepted. | 16 |
| Output | `items.jsonl` (SinhalaMMLU schema plus provenance), and `train/val` as both readable JSONL (prompt + digit) and tokenised Parquet (`input_ids`, `labels`). | 15 |
| Prompt | Exactly the probe and inference prompt: `build_prompt` from [probe_common.py](../scripts/probes/probe_common.py), plain text, ending `පිළිතුර: ` (with the space). Loss is on the answer digit only. | 15 |

---

## 2. What SFT must teach and what that means for the data

### 2.1 How the model is used at inference

The model never generates text. It reads the official Sinhala prompt, and the prediction is the option digit
(`1`–`4`, token ids 16–19) with the highest next-token logit. SFT therefore trains **one token per question**:
the correct digit after `පිළිතුර: `.

### 2.2 What SFT can and cannot do

| SFT can | SFT cannot |
|---|---|
| Teach the answer format and the link "this option text → this digit" | Add much new knowledge. That is the job of CPT, and SFT on 6k questions is too little text for it. |
| Remove **position bias** and calibrate the four digits | Fix facts the model never saw (subjects with little textbook text stay weak) |
| Practise **extracting** facts stored during CPT. Allen-Zhu & Li: QA practice on some facts makes *other* stored facts easier to retrieve, if the facts were seen in varied wordings. | Make up for a bad question. Every wrong key teaches the wrong link. |
| Practise the question forms where every model fails: negation and statement combinations | |

### 2.3 Consequences for the dataset

| Fact | Consequence |
|---|---|
| The target is one digit | The **answer position** must carry no information: exactly 25% per digit, and no surface cue such as the longest option or a repeated stem word. A model learns any shortcut that a 6k-item dataset leaves in. |
| Knowledge is in the weights from CPT | Ground the questions in the **same textbook sections CPT trained on**. Then SFT practises retrieving knowledge the model has, instead of teaching it to guess. |
| Wrong keys are poison | Every key must be supported by a quote from the section and confirmed by an independent, blind check. Unverified items are dropped. |
| Exam questions are short and self-contained | No "according to the passage above". The model never sees the passage at test time. |
| The rules forbid training on test questions; the test comes from `pastpapers.wiki` (2022–2025) | **Never** use past papers, marking schemes, e-thaksalawa or any dataset that may contain them. Use only our textbook text. |
| Probe A (the public Easy questions) is our only real exam-style dev set | Never train on it, never put its questions into a generator prompt, and decontaminate against it. Measuring its **statistics** (lengths, forms, endings) is fine. |
| Probe B measures what CPT stored | Generated questions must not be near-copies of Probe B items. Questions from Probe B's source sections are tagged, and the pilot leaves them out (Step 1). |

---

## 3. What we measured

### 3.1 The source text

| Fact | Value |
|---|---|
| Textbook sections (`04_dedup/textbook.jsonl`) | 3,060 sections, 3.51M Qwen tokens; median 1,128 tokens, 10th percentile 534, 90th 1,716, max 1,865 |
| Sections in the 19 CPT held-out chapters | 110, **excluded for good**. They are the CPT textbook validation set and the Probe B held-out control. |
| Probe B "in training" source sections | 154 (11 per subject), **generated but tagged** `probe_b_source` |
| Sections under 150 tokens | 19, skipped (too little content) |
| **Eligible** | **2,931 sections, 3.39M tokens** |
| Grades | g6 835, g7 990, g8 1,235 sections |
| Thin subjects | **Art**: 45 sections in 3 chapters (69k tokens), because the books are mostly pictures. **Geography**: 88 sections (89k tokens). |
| Not in the shared-task Easy test | **Islam** (it is in Probe A and in the full dataset): kept at half weight |

### 3.2 The real exam style (measured on Probe A, 1,826 public Easy questions)

| Property | Measured | Use |
|---|---|---|
| Question length | median **57** characters, 90th percentile 113 | Ask for short stems. LLMs write much longer ones by default. |
| Option length | median **14** characters, 90th percentile 31 | Options are short phrases or terms, not sentences |
| Most common stem endings | `වන්නේ` (561), `කුමක්ද` (107), `තෝරන්න` (95), `ද` (73), `වනුයේ` (71), `නොවන්නේ` (52), `ලබන්නේ`, `හඳුන්වන්නේ`, `යනු`, `නම්` | Completion-style stems ("… වන්නේ,") are the dominant form |
| Question forms (regex estimate) | completion ~38%, descriptive identification ~30%, direct question ~12%, negation ~10%, fill-in-the-blank ~6%, statement selection ~3%, A/B/C combination **<1%** | The type quotas of Step 2 |
| Answer positions | 1: 465, 2: 495, 3: 498, 4: 368 (option 4 at only 20%) | Do **not** copy this skew. Train with exactly 25%. |
| Key is the uniquely longest option | **30%** (433 / 1,423), per subject 23–39% (chance is 25%) | Our gate: ≤ 30%, aim for 25% |
| Key is the uniquely shortest option | 21% | Our gate: ≥ 18% |
| Numeric option sets | 93 questions; only 27 in ascending order | Numeric options can be shuffled like any others |
| Stems that refer to a figure or table | 30 | We have no images: such items are forbidden |
| "All of the above"-style options | 16 | Allowed only inside combination items |

### 3.3 The prompt strings the test uses

At inference the prompt uses each test item's own `metadata.subject_original`. The shared-task Easy test files use
**different strings from Probe A for two subjects**:

| Subject | Probe A / `SUBJECT_MAP` | Shared-task test |
|---|---|---|
| Art | `චිත්‍ර` | `චිත්‍ර කලාව` |
| Health | `සෞඛ්‍ය හා ශාරීරික විද්‍යාව` | `සෞඛ්‍ය හා ශාරීරික අධ්‍යාපනය` and `සෞඛ්‍යය හා ශාරීරික අධ්‍යාපනය` |

The other 11 test subjects match. SFT prompts for Art and Health must therefore **mix all the variants**
(Step 11), so the model never meets an unseen subject string at test time. These strings are metadata, not
question content. All shared-test Easy items have 4 options.

---

## 4. Pipeline overview and folder layout

```
textbook.jsonl ─► s01 plan ─► s02 generate ─► s03 assemble ─► s04 check ─► s05 validate ─► s06 dedup
  (sections)     (quotas,      (Claude,        (code places     (rules in     (blind, 2nd      (vs itself,
                  slots, waves) per section)    the key)         code)         model)           probes, test)
                                                                                                   │
      s09 render ◄── s08 review ◄── s07 audit ◄────────────────────────────────────────────────────┘
      (prompt, tokens,  (Sinhala      (dataset-level
       train/val)        reader)       bias gates)
```

Scripts (to be written), in `scripts/sft/`:

| Script | Does | Needs |
|---|---|---|
| `s01_plan.py` | Eligible sections, subject quotas, per-section slots, type and cognitive schedule, waves, job list | textbook.jsonl, CPT manifest, Probe B sources |
| `s02_generate.py --wave N` | Sends the jobs through the Claude Code CLI on the claude.ai subscription (`claude_runner.py` guards); appends each finished job; resumable; waits out usage limits | claude.ai login, extra usage off |
| `s03_assemble.py` | Flattens the questions, builds the option lists, places the key from the position schedule | |
| `s04_check.py` | Automatic per-item rules (Step 6), with reason codes | |
| `s05_validate.py --wave N` | Blind validation (Step 7), with the accept rule (3 jobs per call) | claude.ai login, extra usage off |
| `s06_dedup.py` | Near-duplicates within the set and against the test, Probe A and Probe B | `probe_common.near_duplicates` |
| `s07_audit.py [--choices-only]` | Dataset-level gates (Step 9); the choices-only probe runs on Modal | |
| `s08_review.py` | Draws the review sample, reads the reviewer's verdicts back | |
| `s09_render.py --variant pilot\|final` | Prompt, tokens, labels, split, views, manifest | Qwen3.5 tokenizer |

Config and prompts: `scripts/sft/config/sft_easy.json` (quotas, seed, waves, gates),
`scripts/sft/prompts/generator.md`, `validator.md`, `scripts/sft/config/exemplars.json`.

Data:

```
data/processed/sft/easy/
  00_plan/        sections.jsonl  quotas.json  jobs.jsonl  jobs_topup.jsonl
  01_generated/   wave0.jsonl wave1.jsonl ...   (one line per finished job: questions, model, prompt sha1)
                  usage_log.jsonl  prompts/     (every CLI call's usage; a copy of each prompt version)
  02_items/       items_raw.jsonl  skipped.jsonl  assemble_rejected.jsonl
  03_checked/     passed.jsonl  rejected.jsonl  (reason codes)
  04_validated/   validation.jsonl  accepted.jsonl  rejected.jsonl
  05_final/       items.jsonl  audit.json  dedup_report.json  audit_drops.jsonl
  06_rendered/    pilot/{train,val}.jsonl + .parquet    final/{train,val}.jsonl + .parquet
  review/         review_<round>.csv  review_drops.jsonl
  manifest.json
```

**Every stage only appends or writes new files.** Nothing is edited in place. Every stage can be re-run, and skips
the job ids or item ids it has already done. This matters here because the D: drive and the network both drop now
and then.

---

## Step 1 — Freeze the inputs and the exclusions

1. Read `textbook.jsonl`. Record its sha1 and the CPT `manifest.json` sha1 in the SFT manifest.
2. **Exclude for good:** sections whose `parent_id` is one of the 19 held-out chapters in
   `manifest.json → mix_summary.val_chapters`, and sections under 150 tokens.
3. **Tag** `probe_b_source = true` on the 154 primary "train" sections in `data/probes/easy/probe_b_sources.jsonl`.
   They are generated like the others. The **pilot** render leaves them out, so the base-vs-CPT comparison on Probe
   B stays meaningful. The **final** render may include them, because Probe A is the deciding metric by then.
4. Exclude the sections the authored style exemplars were written from (Step 3.4).
5. Write `00_plan/sections.jsonl`: `id`, `parent_id`, subject, grade, chapter, section title, `n_tokens`, the
   tags, and the text.

**Check:** 2,931 eligible sections (minus the exemplar sections), 0 held-out sections, 154 tagged.

---

## Step 2 — The blueprint: subjects, chapters, sections, question types

The planner fixes **what** will be asked **where** before any model call. Coverage and quotas are then guaranteed
by construction, not left to the model. It is deterministic (seed 42), so the plan can be reproduced.

### 2.1 Subject quotas

Weight per subject = ½ × an equal share + ½ × its share of eligible tokens. Islam gets half weight (not in the
shared test). Then a density cap: **at most one question per ~280 tokens of source text**. Denser than that, the
questions become trivial or repeat each other.

| Subject | Sections | Chapters | Eligible tokens | **Questions to generate** | Tokens per question |
|---|---:|---:|---:|---:|---:|
| Art | 45 | 3 | 69k | **240** (capped) | 290 |
| Buddhism | 295 | 52 | 350k | **730** | 480 |
| Catholicism | 224 | 73 | 215k | **560** | 385 |
| Christianity | 217 | 61 | 245k | **600** | 410 |
| Civics | 227 | 11 | 246k | **600** | 410 |
| Dancing | 109 | 20 | 137k | **465** | 295 |
| Drama and Theatre | 153 | 5 | 198k | **540** | 365 |
| Eastern Music | 101 | 11 | 133k | **460** | 290 |
| Geography | 88 | 12 | 89k | **320** (capped) | 280 |
| Health | 267 | 25 | 330k | **700** | 470 |
| History | 174 | 14 | 201k | **545** | 370 |
| Islam (½ weight) | 234 | 82 | 277k | **320** | 865 |
| Science | 429 | 42 | 492k | **900** | 545 |
| Sinhala | 368 | 66 | 403k | **790** | 510 |
| **Total** | **2,931** | **477** | **3.39M** | **~7,800** | |

Expect ~6,400 to survive (Steps 6–8 drop ~15–20%). Art and Geography are capped by their text. If they end up
short, they can be topped up later from other sources (section 17), but never by over-asking the same sections.

### 2.2 Chapters and sections

- Inside a subject, divide the quota over sections **in proportion to their tokens** (largest-remainder rounding).
  Then every part of the curriculum is covered in proportion to its content, which avoids bias toward the first
  chapters or the longest books.
- Each section gets **1 to 6 slots**, and every eligible chapter gets at least 2.
- Sections under ~400 tokens in the same chapter are **merged into one job** of up to ~2,000 tokens and at most 8
  slots.

### 2.3 Question-type quotas (form)

| Type code | Form | Sinhala pattern | Measured in real Easy | **Quota** |
|---|---|---|---:|---:|
| `completion` | Stem is an unfinished sentence; the options complete it | "… වන්නේ," / "… වනුයේ," / "… යනු," / "… නම්," | ~38% | **33%** |
| `identification` | Stem describes something; the options are names or terms | "… ලෙස හඳුන්වන්නේ," / a description ending in "…කි." | ~30% | **20%** |
| `direct` | Wh-question | "… කුමක්ද?" / "කවුද?" / "කීයද?" / "කවරක්ද?" | ~12% | **15%** |
| `negation` | The key is the exception; the 3 distractors are **true** | "… නොවන්නේ," / "සාවද්‍ය ප්‍රකාශය තෝරන්න." / "… නොගැළපෙන්නේ," | ~10% | **12%** |
| `blank` | Fill in the dotted blank | "……… පිහිටා ඇත." | ~6% | **6%** |
| `statement` | Pick the one correct statement out of 4 | "පහත ප්‍රකාශ අතුරින් නිවැරදි ප්‍රකාශය තෝරන්න." | ~3% | **8%** |
| `combination` | 3–4 statements A–D in the stem; options are combinations | "A හා B පමණි" / "A, B හා C සියල්ලම" | <1% | **6%** |

Negation, statement and combination items are **over-weighted** compared with real Easy questions. The SinhalaMMLU
paper found every model weakest on negation and sub-option questions, and those forms are more common at Medium and
Hard. They stay small enough not to distort the Easy style.

### 2.4 Cognitive level (an independent tag)

| Code | Meaning | Quota |
|---|---|---:|
| `recall` | A fact, term, name, date or place stated in the section | 55% |
| `understanding` | A paraphrased concept, cause and effect, classification, comparison | 30% |
| `application` | Apply a rule or concept to a new short case, a simple calculation (Science), a grammar rule on a new word (Sinhala) | 15% |

### 2.5 The schedule

- For each subject, build the multiset of (type, cognitive) slots that matches the quotas exactly. Shuffle it
  (seed) and deal it to that subject's sections.
- Each section's slots get **different** types where the count allows.
- A section that cannot support a slot type (for example, a combination item needs at least 3 distinct facts) may
  return that slot as `skipped` with a reason. The planner refills skipped cells in a top-up wave (16.2).
- **Waves:** inside every chapter, alternate sections between wave 1 and wave 2. Wave 1 then covers every chapter
  of every subject, and gives the stratified pilot set.
- Output: `00_plan/jobs.jsonl`, one line per job: `job_id`, `wave`, `section_ids`, `slots: [{slot, qtype,
  cognitive}]`.

**Check:** the slot counts per subject × type match the quotas (±1); every eligible chapter has slots in wave 1;
no section has more than 6 slots (8 for a merged job).

---

## Step 3 — Item-writing rules and the style guide

These rules go into the generator prompt (Step 4), and are then enforced again by code (Step 6), the validator
(Step 7) and the audit (Step 9). They follow the standard multiple-choice item-writing guidelines (Haladyna,
Downing & Rodriguez, 2002), plus what we measured in section 3.2.

### 3.1 Content

1. **One clearly best answer**, stated in or directly implied by the section, with a **verbatim evidence quote**.
2. **Self-contained.** Never "ඉහත ඡේදයට අනුව", "පාඩමේ සඳහන් පරිදි", "රූපයේ", page or activity numbers. A student
   who studied the subject must be able to answer without this text.
3. **Worth knowing:** a concept, fact, term, person, place, process or rule the lesson teaches. Not the textbook's
   own scaffolding (the names of example children in an activity, the order of the exercises, the title of a box).
4. **No figures, tables, maps or music notation** (we have no images, and the notation text is often garbled).
5. **Paraphrase:** at least ~60% of stems must not copy the source sentence word for word. Real exams reword, and
   varied wording is what makes retrieval transfer (section 2.2).
6. **Grade-appropriate** vocabulary, and the **textbook's own terms** for each subject. This matters for religion:
   Catholic and Christian textbooks use different Sinhala names for the same biblical figures and books, so follow
   the section's spelling.
7. **No time-sensitive facts** (current officials, statistics that change) unless the textbook states them as a
   fixed fact.
8. Religious subjects: accurate to that tradition's textbook, respectful, no comparisons between religions.
9. Facts that are **not** in the section may be used only for distractors, and every distractor must be wrong in
   general, not only "not mentioned".

### 3.2 Options

10. **Exactly 4 options**, all distinct and mutually exclusive: no overlaps, no two synonyms, no option that contains
    another.
11. **Homogeneous:** same category (all kings, all districts, all organs), same grammatical form, and every option
    agrees grammatically with the stem. A distractor that does not fit the stem's grammar is a cue.
12. **Similar length and detail.** The key must not be the most precise or most qualified option.
13. **Plausible distractors:** common misconceptions, related terms from the same chapter or subject, neighbouring
    numbers, dates or places. Never joke options.
14. **No** "ඉහත සියල්ල" / "ඉහත කිසිවක් නොවේ" outside combination items.
15. **No absolute words** ("සෑම විටම", "කිසිවිටෙක", "පමණක්") that only distractors use.
16. **Same formatting** across the options: all end with "." or none do; same number format and units.
17. **No clang cue:** the key must not repeat a distinctive stem word that the distractors lack.

### 3.3 Type-specific rules

- **negation:** the negation word is explicit in the stem; the 3 distractors are **true** statements or members per
  the section; exactly one option is false or does not belong. This is the type where keys are most often wrong, so
  the validator must answer these with extra care.
- **combination:** the generator writes 3–4 labelled statements (A–D) and marks each true or false, each with its
  own evidence. **The code builds the options** (Step 5.2).
- **statement:** 4 complete, similar-length statements; exactly one true. The false ones are realistic
  misstatements: a swapped name, number or cause.
- **blank:** one blank written as a run of dots; the options fit the blank grammatically.
- **numeric:** same units and format; the distractors are realistic mistakes (wrong unit conversion, neighbouring
  value).

### 3.4 Style exemplars

- Author **one exemplar per question type (7 total)**, plus 2 more for Sinhala grammar and Science application.
  Write them once from sections that are then excluded (Step 1). Have the Sinhala reader approve them, and freeze
  them in `config/exemplars.json`.
- **Do not use Probe A questions as exemplars.** The generator would echo them and leak into our only real dev set.
- Put the measured statistics (section 3.2) into the prompt as numbers: "stems are usually 30–110 characters,
  options 5–30 characters".

### 3.5 Subject notes for the generator

| Subject | Ask about | Avoid |
|---|---|---|
| Buddhism | Teachings, suttas, Jataka stories, Pali terms as the book spells them, monks and lay disciples, festivals | Interpretations beyond the book |
| Catholicism / Christianity | Bible events and figures, sacraments, church practices, **each book's own spelling** | Mixing the two traditions' terms |
| Islam | Pillars, practices, terms as transliterated in the book | |
| Civics | Institutions, rights and duties, community services, laws | Current office-holders |
| Geography | Locations, climate, resources, landforms, Sri Lankan districts and provinces | Anything that needs a map |
| History | Kings, eras, places, inscriptions, chronology (ක්‍රි.පූ. / ක්‍රි.ව.) | Dates the book does not give |
| Health | Nutrition, body systems, first aid, sports rules, hygiene | Medical advice beyond the book |
| Science | Concepts, classification, units, simple calculations, experiments described in words | Diagram-based items |
| Sinhala | Grammar (නාම පද, ක්‍රියා පද, සන්ධි, සමාස), word meanings, synonyms and antonyms, spelling (ණ/න, ල/ළ), authors and characters of the book's texts | Questions about the exact wording of a poem line |
| Art, Dancing, Drama, Eastern Music | Forms, traditions, artists, costumes, instruments, ragas and thalas, terms | Notation, pictures, steps shown in figures |

---

## Step 4 — Generation jobs

### 4.1 Your batching idea, and what we keep

| Your idea | Kept? | Refinement |
|---|---|---|
| Small batches instead of hundreds of questions at once | **Yes** | Even smaller, and tied to the source: **one section → 1–6 questions** (at most 8 for merged short sections). |
| Append each batch to a storing file | **Yes** | Append one line per **job id**; skip the ids already done on restart; keep the raw response; never edit in place. |
| Cover all subjects and chapters without bias, and the type quotas | **Yes** | Decided by the planner in code (Step 2), not asked of the model. |
| A second agent checks the correct answer | **Yes** | Made **blind** (it never sees the key), run by **a different model**, with structured quality flags (Step 7). Code checks and dataset audits catch what an LLM cannot see, such as position and length bias across thousands of items. |

**Why one section per call, and not 15–20 questions per call:** the quality of a question depends on how many
distinct facts the source text in front of the model contains. A median section (~1,100 tokens) supports about 3–4
good questions. Asking for 15–20 from one section forces trivial questions and near-duplicates. Asking for 15–20
from many sections spreads the model's attention and invites hallucination.

### 4.2 One request

| Part | Content |
|---|---|
| System prompt (identical for every job, so it can be **prompt-cached**) | Role; the rules of Step 3; the type definitions; the exemplars; the measured style numbers; the output schema |
| User message | Subject (English and Sinhala), grade, chapter title, section title, **the section text**, and the slot list, for example `[{slot:1, qtype:"negation", cognitive:"recall"}, {slot:2, qtype:"completion", cognitive:"understanding"}, …]` |
| Instruction | "Write one question per slot. Each question must test a **different** fact and quote **different** evidence. If the section cannot support a slot's type, return it with `skip_reason`. Never invent facts that are not in the section." |
| Model and settings | `claude-opus-5-5`, adaptive thinking (it cannot be turned off on this model), `output_config.effort` = `high` for wave 0. Compare `medium` on the same jobs and keep it if the quality holds. Structured output (`output_config.format` with the JSON schema below). |

### 4.3 What the generator returns (per question)

```json
{
  "slot": 2,
  "qtype": "negation",
  "cognitive": "recall",
  "question": "<Sinhala stem; for combination: only the introduction line>",
  "closing": "<combination only: 'ඉහත ප්‍රකාශ අතුරින් නිවැරදි වන්නේ,'>",
  "correct": "<key text>",
  "distractors": ["<d1>", "<d2>", "<d3>"],
  "statements": [{"label": "A", "text": "...", "true": true, "evidence": "..."}],
  "evidence": "<verbatim quote from the section that proves the key>",
  "explanation_en": "<one or two English sentences: why the key is right and why each distractor is wrong>",
  "skip_reason": null
}
```

- **There are no positions and no option list.** The model cannot leak a position preference.
- `statements` only for `combination` (then `correct` and `distractors` are empty, and the code builds them).
- `explanation_en` is for the human reviewer and for adjudication. It is **never** shown to the validator and never
  used in training.

### 4.4 How to run it

- **Chosen backend: the Claude Code CLI on the claude.ai Pro subscription** (`claude -p` with `--json-schema`,
  no tools, safe mode, no session files), so the $20 plan is the only cost. `claude_runner.py`:
  - removes every API-key variable from the CLI's environment;
  - checks `claude auth status` (claude.ai login, Pro or Max);
  - stops unless each response confirms extra usage is off (`fast_mode_disabled_reason = extra_usage_disabled`);
  - when the 5-hour or weekly limit is reached, waits for the reset and continues by itself.
- Parallel workers (default 2); one job per generation call, three jobs per validation call.
- Each finished job is appended to `wave<N>.jsonl` at once. A re-run skips finished jobs and retries failed ones.
- Every record stores the model, effort and instruction hash; `usage_log.jsonl` stores every call's usage, and a
  copy of each prompt version is kept. This is the closed-model disclosure record.
- The Message Batches API (half price, but billed separately from Pro) is not used. It can be added later as a
  second backend if API credit is bought.

---

## Step 5 — Assemble the options in code

### 5.1 Position schedule (all types except combination)

- Inside each **wave × subject × type** cell, take the items in a seeded random order and assign key positions by
  cycling `1, 2, 3, 4`. Every cell is then balanced to within 1 item. Random shuffling only gives balance on
  average.
- Checks and validation drop some items, which breaks this exact balance. **s09 therefore re-balances exactly**
  per split × subject × type at render time. Moving the options does not change the item, so the validation
  stays valid.
- Place the key at its assigned position; fill the other three positions with the distractors in random (seeded)
  order.
- Numeric option sets are treated the same way: real Easy exams do not sort them (27 of 93 ascending).

### 5.2 Combination items

- From the A–D truth labels, the true set (for example `{A, C}`) is the key. Distractor combinations are other
  plausible subsets of the same size, plus one size-different subset ("A, B හා C සියල්ලම").
- Options are written in the conventional exam forms: "A හා B පමණි", "A හා C පමණි", "B හා C පමණි",
  "A, B හා C සියල්ලම".
- Choose the three distractor subsets so that, **with the options in canonical order** (the natural order exams
  use), the key lands on the scheduled position. Real exams keep combination options in a natural order, so the
  order stays natural while the position still follows the schedule.
- Reject an item if its statements have no true statement, if all are true and no "සියල්ලම" option fits, or if two
  options name the same set.

Write `02_items/items_raw.jsonl` with the SinhalaMMLU fields (`question`, `choices`, `answer`) plus all
provenance.

---

## Step 6 — Automatic per-item checks

Cheap code checks run before any validation call, so the validator only pays for plausible items. Each failure gets
a reason code in `03_checked/rejected.jsonl`.

| Check | Rule |
|---|---|
| Schema | 4 non-empty, distinct options (compared after `norm()`); `answer` in 1–4; type and slot as planned |
| Script and encoding | NFC; ZWJ kept; Sinhala share of letters ≥ 80% (Latin allowed for A–D labels, units and symbols); no FM artefacts (`•`, stray ASCII-mapped characters) in the stem or options |
| Evidence | The quote is found in the section after `norm()` (the Probe B check). For combination items, every statement's evidence too. |
| Self-contained | No banned references: `ඉහත ඡේද`, `පාඩමේ`, `පාඩමෙහි`, `රූපය`, `රූප සටහන`, `වගුව`, `ක්‍රියාකාරකම`, `පිටුව`, `ඉහත කොටස` |
| Length | Stem 15–250 characters (combination up to 500); each option ≤ 80 characters |
| Length cue | The key is not longer than 1.5 × the longest distractor (unless all options are under 12 characters) |
| Clang cue | The key does not share a distinctive content word with the stem that no distractor shares |
| "All / none of the above" | Only in combination items |
| Negation | The `negation` type has a negation marker in the stem (`නොවන`, `නොවේ`, `සාවද්‍ය`, `වැරදි`, `අසත්‍ය`, `නොගැළපෙන`, …) |
| Blank | The `blank` type has a run of at least 3 dots or `…` |
| Duplicates within a job | No two stems from the same job with a `SequenceMatcher` ratio above 0.8 |
| Not answerable from the stem alone | The key text does not appear verbatim in the stem |

**Check:** the reject rate per reason and per subject. Above 10% for any single reason means the prompt should be
fixed, not just the items dropped.

---

## Step 7 — Blind validation by a second model

### 7.1 Why blind, and why a different model

- If the validator sees the key, it tends to confirm it (confirmation bias), and it misses exactly the cases we care
  about: two defensible answers, or a wrong key.
- A **different model** makes errors that are less correlated with the generator's.
- An **open-book** validator (it sees the section) checks the key against the source. The "self-contained" flag
  checks the closed-book side.

### 7.2 One validation request

- One request per **job** (the section once, then its 1–8 items), `claude-sonnet-5-5`, effort `medium`, structured
  output. Also through the Batches API.
- Items are shown exactly as they will be trained, with their assigned positions, **without** the key, the
  evidence, the explanation or the type label.

It returns, per item:

```json
{
  "item_id": "...",
  "answer": 3,
  "confidence": "high",
  "single_best": true,
  "also_defensible": [],
  "supported_by_section": true,
  "self_contained": true,
  "needs_figure": false,
  "language_ok": true,
  "grade_ok": true,
  "implausible_distractors": 0,
  "source_garbled": false,
  "issues": []
}
```

`answer` = 0 means "no option is correct".

### 7.3 Accept rule

Accept only if **all** of these hold:
- `answer == key`
- `single_best`, with an empty `also_defensible`
- `supported_by_section`, `self_contained`, `language_ok`, `grade_ok`
- not `needs_figure`, not `source_garbled`
- `confidence` is not `low`
- `implausible_distractors` ≤ 1

Everything else is **dropped, not repaired**. Repairing costs as much as generating anew and needs validating
again. Drops are refilled by the top-up wave.

### 7.4 Monitoring the validator

| Signal | Action |
|---|---|
| Disagreement (`answer ≠ key`) above 15% in a subject or type | Read 20 of them. If the keys are wrong, fix the generator prompt. If the validator is wrong, change its instructions. Do this before the next wave. |
| Negation items disagree much more often | Usually the generator made a distractor false. Tighten rule 3.3. |
| The human review (Step 10) also covers 30 **rejected** items | To measure how often the validator throws away good items |

---

## Step 8 — Deduplicate and decontaminate

Use `near_duplicates()` and `norm()` from [probe_common.py](../scripts/probes/probe_common.py), the same matching
that found the 18 hidden near-copies in Probe A.

| Against | Drop if | Flag for a look if |
|---|---|---|
| Itself (accepted items) | stem similarity ≥ 0.8 and ≥ 2 identical options, or stem similarity ≥ 0.9 → keep one | stem similarity ≥ 0.7 within the same section |
| **Shared-task test** (all three levels) | stem similarity ≥ 0.8 and ≥ 2 shared options, or stem ≥ 0.9 | stem ≥ 0.6, or ≥ 3 shared options |
| **Probe A** (full + excluded) | same thresholds | same |
| **Probe B** (405) | same thresholds | same; also any item from a `probe_b_source` section whose evidence overlaps a Probe B evidence quote |
| Exemplars | same thresholds | |

The generator never sees the test, so matches can only come from shared textbook facts. They are dropped anyway:
the cost is a few items, the benefit is a clean contamination record. Write every count to
`05_final/dedup_report.json`. The system report needs them.

---

## Step 9 — Dataset-level bias audit

Shortcuts that no single item shows become visible over thousands of items. `s07_audit.py` computes these on
`05_final/items.jsonl` and writes `audit.json`. **A failed gate blocks the render.**

| Metric | Target | Gate | Fix if it fails |
|---|---|---|---|
| Key position share, per split, subject and type | 25% each | exact by construction (±1 item per cell) | bug in Step 5 |
| Key is the uniquely longest option | ≈ 25% | ≤ 30% overall and in every subject (real Easy: 30%) | drop the worst items (key ≥ 1.3 × the next) until it passes; tighten rule 12 |
| Key is the uniquely shortest option | ≈ 25% | ≥ 18% | same |
| Key has the highest word overlap with the stem | ≈ 25% | ≤ 35% | rule 17 |
| Key is the option most similar to the other options ("convergence" cue) | ≈ 25% | ≤ 35% | ask for more varied distractors |
| **Choices-only probe:** CPT e3 scores 1,000 sampled items from the options alone, no stem | close to Probe A's own choices-only score | ≤ Probe A's value + 5 points, and ≤ 40% | the options carry a cue: find it by type and subject |
| Type shares | the quotas of 2.3 | within ±3 points | top-up wave |
| Subject shares | the quotas of 2.1 | within ±10% relative | top-up wave |
| Chapter coverage | every eligible chapter | ≥ 1 accepted item per chapter | top-up wave |
| Validator acceptance overall | ≥ 80% | ≥ 75% | prompt problem: stop and fix |
| Matches with the test, Probe A or Probe B after Step 8 | 0 | 0 | |

The choices-only probe follows Balepur et al. (2024): if a model beats chance without the question, the dataset
has artefacts the fine-tuned model will learn instead of the content. Running the same probe on Probe A gives the
reference for "how much cue real exams have". It is ~1,000 short forward passes on Modal.

---

## Step 10 — Human review

- `s08_review.py` draws **100 accepted items** (stratified by subject and type) plus **30 rejected items** into
  `review/review_sample.csv`. It shows the stem, the 4 options, the key, the evidence and the English explanation,
  and opens in Excel.
- The Sinhala reader marks each item: key correct? / one best answer? / natural Sinhala? / fair distractors? / any
  cue?
- **Gate:** at least 90 of the 100 accepted items fully good, and no systematic flaw. Read the rejected sample to
  see whether the validator is too strict. Above 50% wrongly rejected is wasted budget: relax the rule.
- Do this for **wave 0 (all ~120 items)** before anything larger, and once more on the pilot.

---

## Step 11 — Split, render and tokenise (the train-ready format)

### 11.1 Final item format (`05_final/items.jsonl`)

The SinhalaMMLU schema, so `build_prompt` and the probe scoring code work unchanged, plus provenance:

```json
{"id": "sft-easy-000123", "subject": "History", "category": "humanities",
 "question": "...", "choices": ["...", "...", "...", "..."], "answer": 3,
 "metadata": {"subject_original": "ඉතිහාසය", "difficulty": "easy", "grade": 7,
              "source_id": "history_g7_si/ch02/s05", "chapter_id": "history_g7_si/ch02",
              "qtype": "negation", "cognitive": "recall", "evidence": "...", "explanation_en": "...",
              "wave": 1, "job_id": "...", "probe_b_source": false,
              "generator": "claude-opus-5-5", "validator": "claude-sonnet-5-5", "validator_confidence": "high"}}
```

`answer` is always an int 1–4 (not a string; see trap 4.4 of the initial plan).

### 11.2 Split

- **Validation set = 5% of sections**, grouped by `source_id` (all questions from a section go to the same side),
  stratified by subject, and chosen by a seeded hash so it is **the same for pilot and final**.
- The SFT validation loss only shows the fit to *synthetic* questions. **Probe A remains the deciding metric.**

### 11.3 Variants

| Variant | Items | Use |
|---|---|---|
| `pilot` | wave 1 accepted, **without** `probe_b_source` | Section 8.8 deciding test: the same LoRA SFT on base and on CPT e3, compared on Probe A and Probe B |
| `final` | all waves accepted, including `probe_b_source` | The submission model |

### 11.4 Rendering

1. **Prompt** = `build_prompt(item)` imported from `scripts/probes/probe_common.py`, never re-implemented. It is the
   official Easy template as plain text (no chat template), ending with `පිළිතුර: ` including the space.
2. **Subject-string variants:** for Art, choose `චිත්‍ර` / `චිත්‍ර කලාව`, and for Health choose among the three
   strings of section 3.3, by a seeded hash of the item id. Every other subject uses its single official string.
3. **Completion** = the answer digit `"1"`–`"4"`. No EOS after it: the model is scored on this one token, and
   training on EOS adds nothing.
4. **Views (training split only):** each item is rendered **twice**. View 0 uses the scheduled positions. View 1
   moves the key two places (`(p + 1) % 4 + 1`, i.e. 1→3, 2→4, 3→1, 4→2) and shuffles the distractors (seeded).
   This keeps the 25% balance exactly, and stops the model tying an item's content to one digit. The validation
   split gets view 0 only. Combination items get no second view: their options keep the natural order.
5. **Tokenise** with the Qwen3.5 tokenizer: `input_ids = tok(prompt) + [digit_id]`,
   `labels = [-100] × len(tok(prompt)) + [digit_id]`.
   - Assert for every item that `tok(prompt + digit) == tok(prompt) + [digit_id]` and that `digit_id` is in
     16–19.
   - Drop and log anything over 1,024 tokens (Easy prompts are mostly 150–400 tokens).
   - The collator must pad with `labels = -100` and `attention_mask = 0` for pads. Don't rely on the pad id
     (pad = EOS in Qwen3.5, the same trap as in CPT).

### 11.5 Files (`06_rendered/<variant>/`)

| File | Content |
|---|---|
| `train.jsonl`, `val.jsonl` | `id`, `item_id`, `view`, `split`, `subject`, `qtype`, `probe_b_source`, `prompt`, `completion`, `answer`. Human-readable, and usable with TRL's prompt–completion format. |
| `train.parquet`, `val.parquet` | the same keys plus `input_ids`, `labels`, `n_tokens`: ready for the Trainer with a padding collator |
| `../../manifest.json` | sha1 of the inputs (textbook.jsonl, CPT manifest, prompt files, exemplars); seed; models and efforts; counts per stage and reject reasons; dedup counts; audit results; token statistics; per-variant item and example counts; dates. **This is the reproduction and disclosure record.** |

**Check:** label distribution exactly 25% per digit in each split; every prompt identical to what `build_prompt`
gives for the item; round-trip one example per subject through `probe_eval.py`'s scoring path, so the training
target and the evaluation read the same token.

---

## 16. How much data, waves, cost and time

### 16.1 How many questions

| | Unique accepted items | Training examples (2 views) |
|---|---:|---:|
| Pilot (wave 1, no Probe B sections) | ~3,000–3,200 | ~6,000 |
| Final Easy | **~6,400** | ~12,000 |

Why this size:

- **Format and calibration** (answer with a digit, no position preference) are learned from roughly 1–3k
  examples. That is why the pilot is enough for the base-vs-CPT decision.
- Beyond that, extra questions help only if they bring **new facts and new wordings**. That is the
  knowledge-extraction effect, so the ceiling is set by the textbook text, not by our budget. At ~1 question per
  280–550 tokens, every section is used without being over-mined.
- Above this density the same sections yield near-duplicates, and the model starts learning the generator's style
  instead of the content. More Easy data should come from **more source text**, not more questions per section.
- Quality beats quantity: 6k clean items beat 15k items of which 10% have wrong keys.

### 16.2 Waves

| Wave | Jobs | Purpose | Gate to continue |
|---|---|---|---|
| **0 — calibration** | ~30 jobs (~2 per subject, every type), `--sync` | Tune the generator and validator prompts; compare effort `high` vs `medium`; measure the tokens and the cost | Human review of all ~120 items ≥ 90% good; validator acceptance ≥ 75%; no gate of Step 9 clearly failing |
| **1 — pilot** | half of the sections of every chapter (~3,900 slots) | The pilot set for the deciding test | Step 9 gates on wave 1; human review of 100 items |
| **2 — rest** | the other half (~3,900 slots) | The final set | Step 9 gates on everything |
| **Top-up** | only the subject × type × chapter cells below 90% of quota, on sections with spare capacity | Fill the gaps left by skips and drops | |

Prompts are frozen after wave 0. If they change later, the change and the wave it starts at go into the manifest.

### 16.3 Cost (API-price estimate; with the Pro backend this is usage, not money)

List prices on 2026-10-06 (check the pricing page): `claude-opus-5-5` $4 input / $20 output per million tokens,
`claude-sonnet-5-5` $2 / $10, and the Batches API at half price. Cached reads of the shared system prompt cost
much less.

| | Requests | Assumed tokens per request (in / out, thinking included) | Batch cost |
|---|---:|---|---:|
| Generation (Opus 5.5) | ~1,950 jobs | ~5k / ~5k | ~$20 + ~$100 |
| Validation (Sonnet 5.5) | ~1,950 jobs | ~3k / ~1.5k | ~$6 + ~$15 |
| **Total Easy** | | | **~$100–200** |

Sinhala is token-expensive, and the thinking length is unknown before wave 0. **With the Pro backend nothing is
billed per call**: these numbers only show how much Pro usage the job needs. The full set will take many
5-hour windows. `usage_summary` reports the calls completed before the first limit hit, which is the real
throughput per window. If that is too slow for the deadline:

- render the pilot only;
- switch the generator to Sonnet;
- lower the effort to `medium`.

### 16.4 Time (fits the 2026-10-27 deadline)

| When | What |
|---|---|
| Oct 23 | Write `s01`–`s04`; author the exemplars; run and review wave 0 |
| Oct 24, morning | Wave 1 batch (most batches finish within about an hour), validation, dedup, audit, render `pilot`, then the deciding SFT test on Modal |
| Oct 24–25 | Wave 2 + top-up → render `final` → final SFT |
| Meanwhile | You prepare the Medium and Hard raw data (section 17) |

---

## 17. Later: Medium and Hard

The same pipeline, run with `--level medium|hard`. What changes:

| Topic | Medium (O/L, grades 9–11) | Hard (A/L, grades 12–13) |
|---|---|---|
| Source text | Grade 9–11 textbooks through `scripts/cpt` (s01–s08). SFT generation can start as soon as `04_dedup/textbook.jsonl` exists, even before a Medium CPT run. | Few or no textbooks for most test subjects (Biology, Chemistry, Physics, Economics, Business, Agriculture, Media…). Add the [Global-MMLU](https://huggingface.co/datasets/CohereForAI/Global-MMLU) `si` split (Apache-2.0, translated MMLU) for the high-school science, economics and business subjects, decontaminated the same way. |
| Instruction string | `eval_baseline.py`'s Medium string (it contains a double space, so copy it exactly) | The baseline's Hard string says "1, 2, 3, 4, 5", but all shared-test Hard items have **4 options**. **Confirm the official template before generating**, and train with exactly the string used at inference. |
| `subject_original` | Collect every variant from the test files' metadata (as in 3.3), e.g. `කෘෂි  විද්‍යාව` with a double space | same |
| Type quotas | Measure first (no public Medium set exists: use the paper's statistics). Raise `combination` to ~10%. | Raise `combination` and `application`; longer stems allowed |
| Stems and options | Longer, more reasoning | Longer; more numeric and multi-step items |
| Probe | Make a Probe B for each level (it is the only signal for these levels besides the leaderboard) | same |
| Past papers | Still forbidden: the Medium and Hard test items come from O/L and A/L papers | same |

Do not use the "exam-style MCQ" part of the HelaLM dataset: it may be derived from past papers.

---

## 18. Changes compared with earlier plans

| Earlier (the plan sent in chat, cpt_overview.md stage 2) | Now | Why |
|---|---|---|
| "3–4 public Easy questions as style examples" | Authored exemplars plus measured statistics | Public Easy is Probe A, our only real dev set. A generator that sees them echoes them. |
| "Reshuffle so positions are uniform" | The generator never sees positions; code assigns an exactly balanced schedule per subject × type | Exact balance, and no generator preference can leak |
| "Second model answers with the section" | **Blind** validation by a **different** model, structured flags, a strict accept rule, and human review of rejected items too | Avoids confirmation bias; measures the validator itself |
| Combination ~10%, negation ~15% | Easy: combination 6%, negation 12%, statement 8% | Real Easy has <1% combination items; over-weight, but not so far that the style is distorted. Raised for Medium and Hard. |
| Exclude the Probe B sections | Generate them, tag them, and choose per render variant | The pilot needs them out; the final model can use them |
| 15–20 questions per call (your idea) | 1–6 per section job (merged short sections ≤ 8) | Quality is bounded by the distinct facts in the source text, not by the call size |
| Nothing about the subject strings | Mix all `subject_original` variants for Art and Health | The test uses strings that Probe A does not |
| Easy 5–7k | ~7,800 generated → ~6,400 accepted, pilot ~3,100 | Measured text capacity, ~1 question per 280–550 tokens |
| Global-MMLU for general practice | Not for Easy; only for Hard subjects without textbooks | Easy has textbook coverage for every test subject |

---

## 19. Final checklist

- [ ] Held-out CPT chapters excluded; Probe B sources tagged; exemplar sources excluded.
- [ ] Plan: subject × type counts match the quotas; every chapter has wave-1 slots.
- [ ] Wave 0 reviewed by the Sinhala reader (≥ 90% good); prompts frozen and hashed.
- [ ] Every job recorded with model, effort, usage and prompt sha1 (disclosure).
- [ ] Positions placed by code: 25% per digit in every split, subject and type.
- [ ] Automatic checks: reject reasons each < 10%.
- [ ] Blind validation: acceptance ≥ 75%; disagreements read and explained.
- [ ] Dedup: 0 matches with the shared-task test, Probe A or Probe B.
- [ ] Audit gates passed (longest key ≤ 30%, choices-only ≤ Probe A + 5 points, coverage).
- [ ] Render: the prompt equals `build_prompt`; tokenisation round-trip asserted; labels only on the digit; Art
      and Health subject variants mixed.
- [ ] `manifest.json` complete; the closed-API use (generator and validator models, dates) noted for the system
      report.
- [ ] No Probe A or Probe B question, and no past-paper content, anywhere in the training files.

---

## 20. References

- Haladyna, Downing & Rodriguez (2002), "A Review of Multiple-Choice Item-Writing Guidelines for Classroom
  Assessment", *Applied Measurement in Education* 15(3).
- Allen-Zhu & Li, "Physics of Language Models, Part 3.1: Knowledge Storage and Extraction".
- Zheng et al. (2024), "Large Language Models Are Not Robust Multiple Choice Selectors" (ICLR): position and label
  bias.
- Balepur, Ravichander & Rudinger (2024), "Artifacts or Abduction: How Do LLMs Answer Multiple-Choice Questions
  Without the Question?" (ACL): the choices-only probe.
- Pramodya et al. (2025), "SinhalaMMLU" (EMNLP): negation and sub-option questions are the hardest:
  https://arxiv.org/abs/2509.03162
- Global-MMLU (Singh et al., 2024), `si` split: https://huggingface.co/datasets/CohereForAI/Global-MMLU
- Anthropic, Message Batches API and structured outputs: https://docs.claude.com/en/docs/build-with-claude/batch-processing
