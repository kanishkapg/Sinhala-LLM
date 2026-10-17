# Preparing the CPT Dataset — Easy level (Grades 6–8)

Target model: **`Qwen/Qwen3.5-4B-Base`** · Stage: Continued Pre-Training (CPT) · Scope: dataset only (no training code)

This guide takes you from the textbook PDFs in `data/raw/g6`, `g7`, `g8` to a **tokenized, packed,
train-ready dataset**. Every step lists what to do, why it matters, and how to check it worked.

> **Implementation:** each step is a script in [`scripts/cpt/`](../scripts/cpt/).
> [`scripts/cpt/README.md`](../scripts/cpt/README.md) gives the run order, the manual-review stops and how to restart.
> The training plan (full CPT vs LoRA, compute, SFT, merging) is in [cpt_overview.md](cpt_overview.md).

---

## Contents

1. [What CPT is and why we do it](#1-what-cpt-is-and-why-we-do-it)
2. [What we know about the raw data](#2-what-we-know-about-the-raw-data)
3. [Pipeline overview and folder layout](#3-pipeline-overview-and-folder-layout)
4. [Step 1 — Collect and inventory the sources](#step-1--collect-and-inventory-the-sources)
5. [Step 2 — Extract text span by span (with font names)](#step-2--extract-text-span-by-span-with-font-names)
6. [Step 3 — Convert legacy FM text to Unicode Sinhala](#step-3--convert-legacy-fm-text-to-unicode-sinhala)
7. [Step 4 — OCR the scanned pages](#step-4--ocr-the-scanned-pages)
8. [Step 5 — Clean and normalise](#step-5--clean-and-normalise)
9. [Step 6 — Quality review (automatic + human)](#step-6--quality-review-automatic--human)
10. [Step 7 — Build the document store (`docs.jsonl`)](#step-7--build-the-document-store-docsjsonl)
11. [Step 8 — Deduplicate and decontaminate](#step-8--deduplicate-and-decontaminate)
12. [Step 9 — Collect replay data (general Sinhala + English)](#step-9--collect-replay-data-general-sinhala--english)
13. [Step 10 — Build the mixture and render training text](#step-10--build-the-mixture-and-render-training-text)
14. [Step 11 — Tokenize, pack and split (the train-ready format)](#step-11--tokenize-pack-and-split-the-train-ready-format)
15. [Step 12 — Final checks and the manifest](#step-12--final-checks-and-the-manifest)
16. [Appendix A — Review of the sample `train.jsonl`](#appendix-a--review-of-the-sample-trainjsonl)
17. [Appendix B — Links](#appendix-b--links)

---

## 1. What CPT is and why we do it

**CPT = continue the model's original training objective (next-token prediction) on raw text from our domain.**
There are no questions, no answers, no chat template — the model simply reads the textbooks, and the loss is
computed on every token.

**Why we need it here.** The competition forbids retrieval of any kind at inference time (no RAG, no lookup
tables). The only place curriculum knowledge can live is **inside the model weights**. CPT is how we put Sri
Lankan school knowledge (history, Buddhism, geography, civics, science in Sinhala) and better Sinhala fluency
into the weights.

**What CPT does *not* do.** It does not teach the model to answer MCQs in the required format. That is the job
of SFT, which comes after CPT (order: **CPT → SFT**; doing SFT first and CPT after erodes the answer format).

**Dataset consequences of how CPT trains:**

| Training fact | What it means for the dataset |
|---|---|
| Text is packed into fixed-length blocks (e.g. 2,048 tokens) | Documents should be cut at natural boundaries and each piece should be self-contained |
| An EOS token separates documents | Every document ends with `<|endoftext|>` |
| Loss is on all tokens | Garbage text (bad conversion, captions, page numbers) is learned too — cleaning matters |
| The corpus is small (~3M tokens) | It will be repeated 2–3 epochs, so it must be mixed with **replay** data to avoid forgetting |

---

## 2. What we know about the raw data

These were measured on the actual PDFs (PyMuPDF + the real Qwen3.5 tokenizer):

| Fact | Value / consequence |
|---|---|
| Files | 32 PDFs (g6: 10, g7: 11, g8: 11) |
| Unicode Sinhala in the PDFs | **0 %** — all text uses legacy **FM fonts** (FMAbhaya, FMMalithi, FMEmanee, FMSamantha, FMDerana, SARAWIPULA, …) that store Sinhala as ASCII/Latin-1 codes. Plain extraction gives `b;sydih` instead of `ඉතිහාසය`. **Must be converted** (Step 3). |
| Scanned book | `g8/health g-8 S.pdf` (216 pages, ~21 chars/page) is image-only → **needs OCR** (Step 4) |
| Size after conversion | ≈ 4.27 M characters ≈ **3.24 M Qwen3.5 tokens**; after cleaning (scripts, before review): 3.67 M characters → **2,547 sections, 2.82 M tokens** |
| Tokenizer efficiency | Qwen3.5 encodes Sinhala at only **1.32 chars/token** (~1 token per syllable) |
| Qwen3.5 special tokens | `bos_token = null` (no BOS), `eos_token = <|endoftext|>`, `add_bos_token = false` |
| Coverage gap | The Easy test set includes **Arts (30), Dancing (39), Drama (60), Eastern Music (44)** = 173 / 642 questions (**27 %**) — and we have **no textbooks** for these subjects |
| Extra subject | Islam textbooks exist, but Islam is not an Easy test subject (keep at normal weight; Sinhala text is still useful) |

---

## 3. Pipeline overview and folder layout

```
 raw PDFs ──► [1] inventory ──► [2] span extraction ──► [3] FM→Unicode ──► [4] OCR (scanned)
                                                                              │
 docs.jsonl ◄── [7] segment ◄── [6] quality review ◄── [5] clean/normalise ◄──┘
     │
     ├──► [8] dedup + decontam ──► [9] + replay data ──► [10] mixture + render ──► [11] tokenize/pack/split
     │                                                                                   │
     └─────────────────────────────── [12] manifest + final checks ◄─────────────────────┘
```

```
data/processed/cpt/easy/
  00_inventory/      inventory.json/.csv: pages, fonts, scanned pages per book          (s01)
  01_extracted/      one .jsonl per book: page, line, font, raw text, converted text   (s02, s03)
  02_clean/          one .jsonl + .txt per book after cleaning (headings, paragraphs)  (s04)
  03_review/         automatic scores, review samples + page images, verdicts          (s05)
  docs.jsonl         textbook document store (one lesson-section per line, with metadata)   (s06)
  replay/            general-Sinhala and English replay docs (same schema family)      (s07)
  04_dedup/          textbook + replay docs after dedup/decontamination, report.json   (s08)
  mix/               train.jsonl / val.jsonl: rendered training text after mixing      (s09)
  packed/
    train/           tokenized + packed (Parquet shards)  ← this is what the trainer reads
    val/
  manifest.json      sources, licences, ratios, token counts, seeds, tokenizer hash    (s10)
```

Treat every stage as **re-runnable from the previous stage** (a script per stage, fixed random seeds). The
competition re-runs your system, and you will re-run this pipeline for Medium (g9–11) and Hard (g12–13).

---

## Step 1 — Collect and inventory the sources

1. **List every book** with grade, subject, medium, page count, and file name. Give each a stable `book_id`
   in the form `{subject}_g{grade}_{medium}`, e.g. `history_g7_si`.
2. **Close the coverage gap first.** Download the Sinhala-medium Grade 6–8 textbooks for **Art, Dancing,
   Drama & Theatre, and Oriental (Eastern) Music** from the Educational Publications Department
   (see [Appendix B](#appendix-b--links)). These four subjects are 27 % of the Easy test.
3. Optional but valuable: **NIE teachers' guides** for the same grades (they explain the same content in
   different words — good for CPT).
4. **Do not collect** past papers, marking schemes, model papers, or e-thaksalawa quiz content. That is the
   source of the benchmark; training on it is contamination.

**Check:** every Easy test subject maps to at least one textbook.

| Easy test subject | Questions | Textbooks we have |
|---|---:|---|
| Science | 92 | g6, g7 (p-I, p-II), g8 (p-I, p-II) |
| Sinhala Language | 83 | g6, g7, g8 |
| Drama | 60 | **missing** |
| Christianity | 58 | g6, g7, g8 |
| Catholicism | 50 | g6, g7, g8 |
| Geography | 50 | g6, g7, g8 |
| Eastern Music | 44 | **missing** |
| History | 40 | g6, g7, g8 |
| Dancing | 39 | **missing** |
| Health | 36 | g6, g7, g8 (g8 scanned) |
| Arts | 30 | **missing** |
| Buddhism | 30 | g6, g7, g8 |
| Civics | 30 | g6, g7, g8 |

---

## Step 2 — Extract text span by span (with font names)

**Why:** a single line often mixes FM-font Sinhala with normal fonts (Times, Minion, Myriad) used for
numbers, units and English (`1.1`, `15 cm`). If you convert a whole line, the Latin parts get corrupted.
So keep the **font name of every span** and decide per span.

**How (PyMuPDF):**

```python
import pymupdf  # pip install pymupdf

FLAGS = pymupdf.TEXTFLAGS_TEXT | pymupdf.TEXT_CID_FOR_UNKNOWN_UNICODE  # keep Latin-1 glyph codes (ú, ï, §, ...)

for page_no, page in enumerate(pymupdf.open(pdf_path), start=1):
    for block in page.get_text("dict", flags=FLAGS, sort=True)["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                font = span["font"].split("+")[-1]   # drop subset prefix "ABCDEF+FMAbhaya"
                record(page_no, block["number"], line["bbox"], font, span["size"], span["flags"], span["text"])
```

Save **page number, bbox, font, size and bold flag** too — you will use them later to detect headings
(larger/bold), headers/footers (top/bottom of page) and captions (small font, inside figures).

**Check:** for every page, total extracted characters > ~100. Pages below that are images → OCR (Step 4).

---

## Step 3 — Convert legacy FM text to Unicode Sinhala

### 3.1 Decide which spans to convert (font whitelist)

| Font family | Action |
|---|---|
| `FM*` (FMAbhaya, FMAbhayax, FMAbabldBold, FMMalithix, FMEmaneex, FMSamanthax, FMDeranax, FMGanganeex, FMBasurux, FMSLRAbhayax, …) | **Convert** |
| `SARAWIPULA` | **Convert** (it is FM-encoded too: `foúhka jykafia` → දෙවියන් වහන්සේ) |
| `F2`, `0KDBOLIDDAbold` (rare) | Inspect a few spans by eye, then decide |
| `IskoolaPota` | Already Unicode — **keep as is** |
| Times / Minion / Myriad / Arial / Garamond | Latin, digits, symbols — **keep as is** |
| `SHREE-TAM*` | Legacy Tamil — **drop** |
| Arabic fonts (Islam books) | Arabic script — drop or keep untouched (not converted) |

Use an **explicit whitelist**, not "starts with FM", so new fonts in later grades are noticed rather than
silently converted wrongly. Log every font you have not classified.

The full inventory of the Easy books found 66 more font names. Legacy Sinhala also hides behind
non-FM names: `F2`, `0KDBOLIDDAbold`, `R1`, `KELANI-Plain`, `FM-ABILIC`. Legacy Tamil fonts are
`NCNitusha`, `TAM-Tamil003`, `Tharmini-Plain` and `Baamini-Plain`. The classified list is in
`scripts/cpt/config/fonts.json`.

### 3.2 The converter

Start from the open-source **Open-SL `fm_abaya_to_unicode`** mapping (MIT licence) — an *ordered* list of
~500 string replacements (longer patterns first: `ff;%` → ත්‍රෛ, … , single keys last). Port it to Python
as a list of `(pattern, replacement)` pairs and apply them in order.

Then apply these **patches**, found by auditing all 32 books:

| # | Patch | Reason |
|---|---|---|
| 1 | `f\da` → `f\|da` and `f\d` → `f\|d` | Upstream regex bug: `\d` means "any digit" |
| 2 | Before the main table, map the nasal prefix: `` `v ``→ඬ, `` `o ``→ඳ, `` `. ``→ඟ, `` `n ``→ඹ, `` `M ``→ළු, `` `¥ ``→ඳූ, `` `÷ ``→ඳු | Not handled upstream (~4,800 occurrences, e.g. `ස`දහා` instead of සඳහා) |
| 3 | `¥` → දූ and `¨` → ලු (not ලූ) | Missing / wrong (`ගැටලූව` should be ගැටලුව) |
| 4 | Delete the `•` → `x` rule; map `¡`, `²` → `•` | Bullets were turning into `x` (~7,000 occurrences) |
| 5 | `%` must output `්‍ර` (no leading ZWJ); remove stray `=` in outputs like `ඡ=`, `රැ=`, `ඳැ=`, `ඤු=`, `ඤූ=` | Bad Unicode in upstream outputs |
| 6 | Verify `(`, `)`, `&`, `^`, `'`, `"` against the printed page **per font** | Punctuation differs between FM fonts (in the science book `)` became `*`; in Christianity `(` is `:` for verse numbers like `1:31`) |
| 7 | Replace ligatures `ﬁ`→`fi`, `ﬂ`→`fl` *before* converting | Typesetting software sometimes stores these as ligatures |
| 8 | Delete the early `&`→`)` rule | Its output became `*` through the later `)`→`*` rule, so every `)` was printed as `*` |
| 9 | Map nasal prefix forms to the FM code of the pre-nasalised letter (`` `o ``→`\|`, `` `ÿ ``→`÷`, `` `â ``→`~a`, …), not to Unicode | Keeps the `f` (ෙ) reordering rules working (`f`o` → ඳෙ); also handles `` ` `` written before the vowel sign (`y`so` → හිඳ) |
| 10 | Rare glyphs `ª`→ඳු, `ý`→ඡි, `\`→ඳ, `³`/`⋆`→•, `·5`→×5; vowel sign before `්‍ර`/`්‍ය` → after it (ගාබි්‍රයෙල් → ගාබ්‍රියෙල්) | Found by the leftover-character audit |

**Expected quality:** with these patches, about 90 of the 3.3 M characters in FM spans remain
unconverted (Ω, ¼ and a few rare forms), and malformed syllables are < 0.5 per 1,000 characters in every book.
All patches are listed in `scripts/cpt/fm_converter.py` (`PATCHES`) and copied into `manifest.json`.

**Check:** run the automatic checks of Step 6.1 per book; every leftover character that is not Sinhala,
digit or punctuation must be explained (e.g. `I V X i v` are intentional roman numerals from `¸ ¹ º µ ¶`).

---

## Step 4 — OCR the scanned pages

Applies to `g8/health g-8 S.pdf` and any page from Step 2 with < ~100 characters.

1. Render each page to an image at **300 dpi** (`page.get_pixmap(dpi=300)`).
2. OCR with one of:
   - **Tesseract** with the `sin` model from `tessdata_best` (free, offline; decent on clean print, weaker on
     layout). Command: `tesseract page.png out -l sin --psm 4`.
   - A stronger OCR service or vision model. The rules allow closed APIs **only for building training data,
     and this must be disclosed** in the system report. Record which tool produced which pages.
3. Mark these documents with `"extraction": "ocr"` so you can down-weight or re-check them.

**Check:** a reviewer compares 10 random OCR pages against the images (Step 6.2).

---

## Step 5 — Clean and normalise

Apply in this order, per book:

1. **Drop whole sections that are not curriculum content:** the national anthem, the preface/message pages, the
   table of contents, the publisher/copyright page, and the empty pages at the back. Keep **glossaries**
   (word lists are useful).
2. **Remove headers, footers and page numbers.**
   - Rule: a line that repeats on > 30 % of a book's pages is a running header/footer → drop it.
   - Known footer: `නොමිලේ බෙදා හැරීම සඳහා ය.`
   - Lines that are only a number at the top/bottom of a page → page numbers → drop.
3. **Remove figure / map / diagram labels.** These are short lines inside images: `ඔලිව් ගසක්`,
   `තහඩුව`, `පොලිතින් කොළය`, map labels like `ඇ ප | ෆොලොරන්ස් | ි නයින`.
   - Use the bbox + font size from Step 2: text inside image regions, or with a smaller font than body
     text, is a label.
   - Keep real **captions** (`2.12 රූපය සරල ධාරිත්‍රකය`) as one line, or drop them all — be consistent.
   - Drop **flattened mind-maps** (many short phrases run together on one line).
4. **Rejoin hard-wrapped lines into paragraphs.** PDF lines break mid-sentence, and paragraphs break across
   pages. Join line *n* to line *n+1* when line *n* does not end in `. ? ! :` and is not a heading. Also join
   across page breaks.
5. **Mark headings.** Detect headings from font size / bold / numbering (`2.5`, `5.2`) and write them as
   Markdown headings (`## ධාරිත්‍රක`). Remove a heading line that is immediately duplicated (caption +
   heading of the same name).
   - **Chapters:** in these books every chapter opens with a big stand-alone number (`01`, `4`, `4.`) or
     a big numbered title (`3. …`). The chapter title is the largest text on that page. The numbers
     must run 1, 2, 3 … through the book; that excludes big numbered list items.
   - Text before chapter 1 (title page, credits, preface, contents) is dropped.
   - Missed chapter openers can be added by hand (`chapter_pages` in `config/books_<level>.json`).
   - Some text is printed twice (shadow effect) → drop the second copy.
   - Rejoin words hyphenated across lines (`හේ-` + `න්` → `හේන්`).
6. **Normalise lists and exercises.** Bullets → `• `, numbered items → `1. `. Fill-in-the-blank dot leaders
   (`''''''''` → `........`) → `______`. **Keep** activities (`ක්‍රියාකාරකම`) and exercises — they are
   curriculum content.
7. **Unicode normalisation:**
   - NFC normalisation.
   - Consistent ZWJ conjuncts: `්‍ය` (yansaya) and `්‍ර` (rakaransaya) as `U+0DCA U+200D U+0DBA/U+0DBB`.
   - Remove control characters, ` `, `\xa0`, zero-width spaces (keep the ZWJ `U+200D`).
   - Collapse runs of spaces; at most one blank line between paragraphs.
8. **Language filter:** drop paragraphs with < 50 % Sinhala characters (excluding digits/punctuation),
   unless they are intentional English terms inside Sinhala text.

**Check:** open `02_clean/*.txt` side by side with the PDF for one chapter per book.

---

## Step 6 — Quality review (automatic + human)

"Have a Sinhala reader skim" means **a person who reads Sinhala fluently** checks samples of the
converted text against the original pages. No app can fully replace that, but tools make it fast and
make the automatic checks point the reviewer at the worst text first.

### 6.1 Automatic checks (run on every book, every time)

| Check | How | Flag when |
|---|---|---|
| Malformed syllables | Regex: vowel sign / hal at word start `(?:^|[\s(])[ා-ෟ්]`; two vowel signs in a row `[ා-ෟ]{2,}` (excluding valid pairs) | > 1 per 1,000 chars |
| Unmapped glyphs | Count chars in Sinhala paragraphs that are not Sinhala, digits, whitespace or common punctuation (e.g. `` ` ¥ ¨ ú ï § ``) | any — each must be explained |
| Perplexity outliers | Train a small **KenLM** 5-gram model on Sinhala Wikipedia (or use Qwen3.5-4B-Base itself) and score every paragraph | top 2 % highest-perplexity paragraphs → send to human review |
| Spelling / out-of-vocabulary rate | Build a word-frequency list from Sinhala Wikipedia + FineWeb-2 Sinhala; or use the Hunspell `si_LK` dictionary / SinSpell | OOV rate much higher than the book average |
| OCR cross-check (conversion errors) | Render the page → Tesseract `sin` → character error rate (CER) vs your converted text | pages with CER far above the book median (Tesseract has its own errors, so use it *relatively*) |

### 6.2 Human review (the "Sinhala reader skim")

**Who:** one or two team members (or volunteers) who read Sinhala fluently.

**What to sample (fixed seed, saved to `03_review/`):**
- 20 random paragraphs **per book**, plus
- the 20 worst paragraphs per book from the automatic checks, plus
- 10 random pages from each OCR'd book.

**Tool options (pick one):**

| Tool | Best for | Setup |
|---|---|---|
| **Label Studio** (free, open source) | Best choice: shows the **PDF page image next to the converted text**, with buttons for the verdict | `pip install label-studio` → `label-studio start` → create a project with the template below → import a JSON file of tasks |
| **Argilla** (free, open source, from Hugging Face) | Review that also pushes results back to a Hugging Face dataset | `pip install argilla`, run the server (Docker or HF Space), create a dataset with a text field + rating question |
| **Google Sheets / Excel** | Fastest to start, no install | Columns: `id`, `book`, `page`, link to the page image, `text`, `verdict`, `error_type`, `notes` |
| **Small Streamlit app** | A custom side-by-side viewer for your own team | ~40 lines: image of the page on the left, text on the right, radio buttons that save to CSV |

**Label Studio template** (image + text + verdict):

```xml
<View>
  <Image name="page" value="$image"/>
  <Text name="text" value="$text"/>
  <Choices name="verdict" toName="text" choice="single" showInline="true">
    <Choice value="OK"/>
    <Choice value="Minor errors"/>
    <Choice value="Broken"/>
  </Choices>
  <Choices name="error_type" toName="text" choice="multiple" showInline="true">
    <Choice value="Wrong letters / conversion"/>
    <Choice value="Broken sentence order (layout)"/>
    <Choice value="Caption / label noise"/>
    <Choice value="Header / footer / page number"/>
    <Choice value="Missing text"/>
  </Choices>
  <TextArea name="notes" toName="text" placeholder="Wrong form → correct form, e.g. ගැටලූව → ගැටලුව"/>
</View>
```

Each task is `{"image": "<url or local path of the page PNG>", "text": "<paragraph>", "id": "...", "page": 12}`.
Render page PNGs with `page.get_pixmap(dpi=120).save(...)`. Serve local files with Label Studio's local
storage, or put them on any static file host.

**What the reviewer does for each sample:**
1. Read the converted text while looking at the page image.
2. Choose a verdict and the error types.
3. For wrong letters, write `wrong → correct` in notes (e.g. `ස`දහා → සඳහා`). These notes are how you find
   missing converter rules.

**Acceptance rule (per book):**
- ≥ 95 % of samples `OK`, **and**
- no error that repeats (the same wrong form appearing in several samples).

**When a book fails:** add a converter rule or cleaning rule → re-run Steps 3–5 for that book → draw a
**new** sample → review again. Record every rule you add (it goes in the system report).

---

## Step 7 — Build the document store (`docs.jsonl`)

The document store is the **clean text + metadata**, one document per line. It is *not* yet the
train-ready format (that is Step 11), but it is the source of truth that all later steps read.

### 7.1 Unit of a document: a lesson **section**, not a whole chapter

Whole chapters are 10 k–27 k tokens, much longer than a 2,048–4,096-token training block. If you store whole
chapters, packing cuts them into many blocks and **only the first block sees the subject/grade/chapter
header**. So:

- Split each chapter at its headings (`##`) into **sections of ~800–1,800 tokens**.
- If a section is longer, split it at paragraph boundaries. If it is very short (< 150 tokens), merge it with
  the next section.
- Keep the chapter as the parent (`parent_id`), so you can still split train/val by chapter.

### 7.2 Recommended schema

```json
{
  "id": "history_g7_si/ch01/s03",
  "parent_id": "history_g7_si/ch01",
  "book_id": "history_g7_si",
  "source": "textbook",
  "subject": "History",
  "subject_si": "ඉතිහාසය",
  "grade": 7,
  "level": "easy",
  "medium": "si",
  "lang": "si",
  "kind": "section",
  "chapter_no": 1,
  "chapter": "අපේ පැරණි ජන ජීවිතය",
  "section": "ජන ජීවිතය",
  "page_start": 13,
  "page_end": 15,
  "extraction": "fm_convert",
  "text": "## ජන ජීවිතය\n\nඅතීතයේ අපේ රටේ ජනයා ...",
  "n_chars": 4210,
  "n_tokens": 3190,
  "sha1": "3f2a…",
  "license": "Educational Publications Department, Sri Lanka (free distribution textbook)",
  "pipeline_version": "cpt-easy-v1"
}
```

| Field | Why it is there |
|---|---|
| `id` (unique, stable) | Joins review results, dedup clusters and splits; the pattern `{book_id}/ch{NN}/s{NN}` sorts naturally |
| `parent_id` | Train/val split **by chapter**, so validation never shares a chapter with training |
| `source` | Mixing by source (`textbook`, `wikipedia`, `fineweb2`, …) in Step 10 |
| `subject`, `subject_si`, `grade`, `level`, `chapter`, `section` | Used to render the metadata header (Step 10) and for per-subject statistics |
| `extraction` | `fm_convert` / `unicode` / `ocr` — lets you down-weight or re-check OCR text |
| `text` | Clean body text only, **without** the metadata header (added at render time, so you can change its format without rebuilding the store) |
| `n_chars`, `n_tokens` | **Computed from `text`** by the script (never copied); `n_tokens` uses the Qwen3.5 tokenizer |
| `sha1` | Exact-duplicate detection and change tracking |
| `license`, `pipeline_version` | Reproducibility and the system report |

**JSONL rules:** UTF-8, exactly one JSON object per line, **no blank lines**, `ensure_ascii=False` (keeps
Sinhala readable), unique `id`s, LF line endings.

See [Appendix A](#appendix-a--review-of-the-sample-trainjsonl) for the review of the sample file.

---

## Step 8 — Deduplicate and decontaminate

### 8.1 Deduplication
1. **Exact:** drop documents with an identical `sha1` of the normalised text.
2. **Near-duplicate:** MinHash LSH on character 5-grams, Jaccard ≥ 0.8 (e.g. the `datasketch` library).
   Expected hits: Bible passages shared by Catholicism and Christianity, and repeated front matter. Keep one
   copy (prefer the higher grade or the cleaner extraction).
3. Run dedup **again after adding replay data** (Step 9). Web corpora often contain copies of textbook
   passages.

### 8.2 Contamination check (against test questions)
1. Build an n-gram index (13-grams of whitespace tokens, or 50-character shingles) from **every question and
   option you can see**: the shared-task test set in `data/SinhalaMMLU-SharedTask-Test/` and the public
   `naist-nlp/SinhalaMMLU` split.
2. Scan all textbook and replay documents.
   - **Textbooks** are allowed source material. Record the overlap rate for the report. Only remove a
     passage if it is actually exam-style MCQ content.
   - **Replay/web data:** **drop any document** that contains an overlapping n-gram with a question. Web
     crawls can contain past papers.
3. Also drop replay documents whose URL is a past-paper or quiz site (e-thaksalawa, past-paper sites) when
   the source gives a URL.
4. Save the numbers (docs scanned, docs dropped, per source) to `manifest.json` — the system report needs
   them.

> Never look at or tune on the test questions beyond this automatic overlap check.

---

## Step 9 — Collect replay data (general Sinhala + English)

**Why:** ~2.7 M textbook tokens repeated 2–3× is a very narrow diet for a 4 B model. Without replay the
model overfits the textbook style and forgets general skills. Replay = high-quality general text mixed into
every CPT run.

### 9.1 Sources

All are on Hugging Face unless noted. Licences are taken from the dataset cards — re-check each card before
use, and put the licence of every source you use in `manifest.json`.

**General Sinhala**

| Priority | Source | HF id / config | Licence | Notes |
|---|---|---|---|---|
| ★★★ | **Sinhala Wikipedia** | `wikimedia/wikipedia`, config `20231101.si` (or a fresh dump from dumps.wikimedia.org/siwiki) | CC BY-SA 3.0 / GFDL | Highest quality; strong on Sri Lankan history, Buddhism, geography. 23 k articles (~60–80 M tokens): sample 4 M per Easy run, and use more of it in the later cumulative runs |
| ★★★ | **FineWeb-2 Sinhala** | `HuggingFaceFW/fineweb-2`, config `sin_Sinh` | ODC-By 1.0 | Already filtered and deduplicated web text; the best general web source |
| ★★ | **HPLT 2.0 cleaned** | `HPLT/HPLT2.0_cleaned`, config `sin_Sinh` | CC0 | Largest Sinhala web source; noisier — apply the filters in 9.2 |
| ★★ | **MADLAD-400 (clean)** | `allenai/MADLAD-400`, files `data/si/si_clean_*.jsonl.gz` | ODC-By | Use `si_clean` only, never `si_noisy` |
| ★★ | **CulturaX** | `uonlp/CulturaX`, config `si` | mC4 (ODC-By) + OSCAR terms | Gated — accept the terms on HF first |
| ★★ | **Sinhala-1.5B-corpus** | `sinhala-nlp/sinhala-1.5B-corpus` | "other" (mix: CC0, ODC-By, CC BY, **CC BY-NC-SA** for NSina news) | One combined download: HPLT 2.0 + FineWeb-2 + NSina news + SinMin + Facebook/Twitter. Has a `source` field — use it to pick subsets (e.g. news, SinMin). **SinMin includes school textbooks and gazettes → decontaminate** |
| ★ | **MADLAD + CulturaX cleaned (SinLlama corpus)** | `polyglots/MADLAD_CulturaX_cleaned` | not stated on the card | ~10.7 M rows, the corpus used for SinLlama; already deduplicated. Check its licence before use |
| ★ | **mC4 Sinhala** | `allenai/c4`, config `si` | ODC-By | Older and noisier; overlaps with MADLAD/CulturaX |
| ★ | **FinePDFs Sinhala** | `HuggingFaceFW/finepdfs`, config `sin_Sinh` | ODC-By | Text extracted from PDFs; likely includes educational documents and **past papers → strict decontamination**, or skip |
| ★ | Serendib CPT corpus | `Chamaka8/serendip-cpt-sinhala` | CC BY-4.0 | Large deduplicated Sinhala CPT corpus (student project); overlaps the web sources above |

**English (to keep general ability)**

| Source | HF id / config | Licence |
|---|---|---|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` (e.g. `sample-10BT`) | ODC-By |
| English Wikipedia | `wikimedia/wikipedia`, config `20231101.en` | CC BY-SA 3.0 |

**Recommended for the Easy run:** Sinhala Wikipedia + FineWeb-2 `sin_Sinh` as the core,
HPLT 2.0 / MADLAD-clean as extra volume if needed, and FineWeb-Edu for English.

**Size — you only need a slice:**

| Source (Sinhala part only) | Files | Documents | Text | Easy budget |
|---|---:|---:|---:|---:|
| Wikipedia `20231101.si` | 54 MB Parquet | 23,065 | ~290 MB | 4 M tokens |
| FineWeb-2 `sin_Sinh` | 2.0 GB Parquet | 1.19 M | ~11 GB | 5 M tokens |
| FineWeb-Edu `sample/10BT` (English) | 28 GB Parquet | ~9.7 M | — | 3 M tokens |

Figures like "78 GB" (Wikipedia) or "2 TB" (FineWeb-2) are the **whole multilingual repositories**; you never
download those. `scripts/cpt/s07_replay.py` doesn't even download the Sinhala files. It opens them
remotely, reads **random row groups**, filters the documents (9.2), and stops at the token budget set in
`config/mixture.json`. Training cost depends on the tokens you train on, not on the corpus size
(see [cpt_overview.md](cpt_overview.md#2-how-much-compute-and-why-the-huge-corpora-are-not-a-problem)).

### 9.2 Filters for replay documents
Apply in order and log how many documents each filter removes:

1. Language: Sinhala script ratio ≥ 0.7 (for Sinhala sources).
2. Length: 200 – 20,000 characters.
3. Quality heuristics: mean line length, fraction of lines ending in punctuation, ratio of repeated lines,
   no lists of links/menus, no `lorem`, no cookie/boilerplate text.
4. Optional: KenLM perplexity (the model from Step 6.1) — drop the worst 20–30 %.
5. Dedup against itself and against the textbooks (Step 8.1).
6. **Decontaminate against test questions (Step 8.2) — mandatory for every replay source.**
7. Convert to the same schema as `docs.jsonl` (`source` = `wikipedia`/`fineweb2`/…, `subject`/`grade` =
   `null`).

---

## Step 10 — Build the mixture and render training text

### 10.1 Mixture (starting point for the Easy run)

| Share of tokens | Source | Approx. tokens |
|---:|---|---:|
| ~40 % | Textbooks g6–8 (repeat ×3 across epochs) | ~8 M (2.7 M × 3) |
| ~45 % | General Sinhala (Wikipedia + FineWeb-2 + others) | ~9 M |
| ~15 % | English (FineWeb-Edu) | ~3 M |
| | **Total per run** | **~20 M tokens** |

- Count tokens with the **Qwen3.5 tokenizer**, not characters or words.
- Repeat textbooks by duplicating them into different epochs (shuffled differently each time). Do not repeat
  replay data — sample fresh documents instead.
- **Later grade bands:** keep the mixture **cumulative**: Medium run = Easy textbooks + Medium textbooks +
  replay; final run = all grades + replay. Or do one final run on everything. Training only on new
  grades erodes earlier gains.

### 10.2 Render the training text
At render time, prepend a short Sinhala metadata header to **every textbook section** (not to replay docs):

```
ශ්‍රේණිය: 7 | විෂය: ඉතිහාසය | පාඩම 1: අපේ පැරණි ජන ජීවිතය | ජන ජීවිතය

## ජන ජීවිතය

අතීතයේ අපේ රටේ ජනයා ...
```

This ties every fact to its subject and grade. Use the **same Sinhala subject names** as the official
evaluation prompt template, so the subject name in a test prompt points the model at the right knowledge.

Write the result to `mix/*.jsonl` with just `{"id", "source", "text"}` (+ `parent_id` for the split).

---

## Step 11 — Tokenize, pack and split (the train-ready format)

**The train-ready format is not JSONL text — it is tokenized, packed blocks stored as Parquet** (or Arrow via
`datasets.save_to_disk`). The trainer only reads `input_ids`; labels are a copy of `input_ids` (the data
collator can create them).

### 11.1 Split first (by chapter / document), then pack
- **Validation:** hold out ~2–3 % of textbook chapters (`parent_id`), stratified by subject and grade, plus a
  small slice (~0.5 %) of each replay source. Splitting **before** packing makes sure no validation text
  appears inside a training block.
- Use `val` to track perplexity on unseen curriculum text during CPT.

### 11.2 Tokenize and pack

```python
from transformers import AutoTokenizer
from datasets import load_dataset

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base")
EOS = tok.eos_token_id          # <|endoftext|>; Qwen3.5 has no BOS
BLOCK = 2048                     # 4096 if GPU memory allows

def tokenize(batch):
    # raw text + EOS per document; no chat template, no special prompt
    return {"input_ids": [tok(t, add_special_tokens=False)["input_ids"] + [EOS] for t in batch["text"]]}

def pack(batch):
    ids = [t for doc in batch["input_ids"] for t in doc]
    n = len(ids) // BLOCK * BLOCK                       # drop the remainder of each batch
    return {"input_ids": [ids[i:i + BLOCK] for i in range(0, n, BLOCK)]}

ds = load_dataset("json", data_files={"train": "mix/train.jsonl", "val": "mix/val.jsonl"})
ds = ds.shuffle(seed=42)                                # shuffle documents across sources BEFORE packing
ds = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)
ds = ds.map(pack, batched=True, batch_size=1000)
ds["train"].to_parquet("packed/train/train.parquet")
ds["val"].to_parquet("packed/val/val.parquet")
```

Rules:
- `add_special_tokens=False` and **no chat template** — CPT is plain text.
- **One EOS after every document**, so the model learns document boundaries.
- Shuffle **documents** before packing (mixes sources inside blocks), with a fixed seed.
- Every block has exactly `BLOCK` tokens → no padding is needed.
- Cross-document attention inside a block is standard practice for CPT; optional refinements (resetting
  `position_ids` per document) are a training-side choice and do not change the dataset.

### 11.3 Final train-ready layout

```
packed/train/*.parquet   column: input_ids (list<int32>, length = BLOCK)
packed/val/*.parquet     column: input_ids
manifest.json
```

---

## Step 12 — Final checks and the manifest

**Checks before training:**

1. **Decode round-trip:** decode 5 random blocks with `tok.decode()` and read them. You should see clean
   Sinhala, headers and EOS boundaries.
2. **Token counts per source** match the planned mixture (± 5 %).
3. **No `val` `parent_id` appears in `train`.**
4. **Contamination report** is saved (Step 8.2).
5. **Reproducibility:** re-running the pipeline with the same seeds gives identical `sha1`s for `docs.jsonl`
   and identical Parquet row counts.

**`manifest.json` — record:**
- pipeline version and seeds
- the tokenizer name and the hash of `tokenizer.json`
- `BLOCK` size
- for every source: name, HF id/config or file list, licence, documents and tokens before/after each filter
- dedup and decontamination counts
- the converter rule list (with your patches)
- OCR tool used and pages OCR'd
- any closed-API use (must be disclosed)

---

## Appendix A — Review of the sample `train.jsonl`

The sample file (`history_g7_si`, chapters 1, 2, 3, 5) was checked line by line.

### What is already good
- Clean **Unicode** Sinhala, NFC-normalised, correct ZWJ conjuncts (`ප්‍ර`, `ශ්‍රේ`), no leftover
  FM characters, no footers or page numbers, almost no malformed syllables (1 in ch05).
- Good identifiers and metadata: `id`, `book_id`, `subject` + `subject_si`, `grade`, `medium`, `kind`,
  `chapter`, `chapter_no`, page range, `n_chars`.
- Paragraphs separated by blank lines; activities (`ක්‍රියාකාරකම`) kept.

**Verdict:** the structure is a **good basis for the document store** (Step 7). It is not yet the train-ready
format — that is the packed Parquet from Step 11.

### Problems to fix

| # | Problem found | Fix |
|---|---|---|
| 1 | The file has 7 lines but only 5 records — **blank lines** (lines 5, 6, 8) | Write exactly one object per line, no blank lines |
| 2 | **Duplicate `id`**: line 7 repeats `history_g7_si/ch01` with a truncated `text` (171 chars ending in `...`) but `n_chars = 19407` | Remove it; enforce unique `id`s; compute `n_chars` from `text` |
| 3 | **Chapter 4 is missing** (ch01, 02, 03, 05) | Check extraction/segmentation for that chapter |
| 4 | Documents are **whole chapters** (10,460 – 26,979 tokens) | Split into sections of ~800–1,800 tokens (Step 7.1), with `parent_id` = chapter |
| 5 | **Figure captions and map labels** inside the text: `ඔලිව් ගසක්`, `ඇතීනා දෙවඟන`, `ඇ ප \| ෆොලොරන්ස් \| ි නයින \| ...` | Step 5.3 (bbox/font-size based) |
| 6 | **Flattened mind-map**: `ආරම්භය ප්‍රධාන ලක්‍ෂණ ලෝකයට උරුම වූ දායාද රූපය 5.9 කවපෙත්ත විසි කරන්නා` | Drop lines that are runs of short phrases / diagram text |
| 7 | **Paragraphs broken across lines/pages**: `3. ශ්‍රී ලංකාවේ විවිධ ස්ථානවලින් හමු වූ` ⏎⏎ `පුරාණ රෝම කාසි ...`; bullet `• සුවඳ විලවුන්, අලංකාර මැටි` cut | Step 5.4 line re-joining |
| 8 | **Headings are not marked** and some repeat (`ග්‍රීක ශිෂ්ටාචාරය` twice, `ඔගස්ටස් සීසර්` twice) | Write headings as `## …`; remove duplicated caption/heading lines |
| 9 | `chapter` includes the number (`"1. අපේ පැරණි ජන ජීවිතය"`) and also has `chapter_no` | Keep the title only in `chapter` (the number is in `chapter_no`) |
| 10 | Missing fields for mixing and reproducibility | Add `parent_id`, `source`, `level`, `lang`, `section`, `extraction`, `n_tokens`, `sha1`, `license`, `pipeline_version` |
| 11 | File name `train.jsonl` for the document store | Call it `docs.jsonl`; train/val is decided by chapter in Step 11 |

---

## Appendix B — Links

**Textbooks and curriculum**
- Educational Publications Department — e-books (free textbooks): http://www.edupub.gov.lk/BooksDownload.php
- National Institute of Education (teachers' guides, syllabi): https://nie.lk

**FM → Unicode conversion and PDF tools**
- Open-SL Sinhala Unicode converter (MIT; `src/translators/fm_abaya_to_unicode.ts`): https://github.com/Open-SL/sinhala-unicode-converter
- Pandukabhaya (Python FM Abhaya converter, alternative, untested by us): https://pypi.org/project/pandukabhaya
- PyMuPDF: https://pymupdf.readthedocs.io

**OCR**
- Tesseract: https://github.com/tesseract-ocr/tesseract
- Tesseract `sin` model (best quality): https://github.com/tesseract-ocr/tessdata_best

**Quality review**
- Label Studio: https://labelstud.io · https://github.com/HumanSignal/label-studio
- Argilla: https://github.com/argilla-io/argilla
- Streamlit: https://streamlit.io
- KenLM (n-gram perplexity): https://github.com/kpu/kenlm
- Sinhala Hunspell dictionary (si_LK, 2010): https://extensions.openoffice.org/en/project/sinhala-ubiquitous-spell-checker
- SinSpell (Sinhala spell checker paper): https://arxiv.org/abs/2107.02983
- datasketch (MinHash dedup): https://github.com/ekzhu/datasketch

**Replay data (Hugging Face)**
- Sinhala Wikipedia: https://huggingface.co/datasets/wikimedia/wikipedia (config `20231101.si`) · dumps: https://dumps.wikimedia.org/siwiki/
- FineWeb-2: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 (config `sin_Sinh`)
- HPLT 2.0 cleaned: https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned (config `sin_Sinh`)
- MADLAD-400: https://huggingface.co/datasets/allenai/MADLAD-400 (`data/si/si_clean_*`)
- CulturaX: https://huggingface.co/datasets/uonlp/CulturaX (config `si`, gated)
- Sinhala-1.5B-corpus: https://huggingface.co/datasets/sinhala-nlp/sinhala-1.5B-corpus
- MADLAD + CulturaX cleaned (SinLlama): https://huggingface.co/datasets/polyglots/MADLAD_CulturaX_cleaned
- mC4 / C4 multilingual: https://huggingface.co/datasets/allenai/c4 (config `si`)
- FinePDFs: https://huggingface.co/datasets/HuggingFaceFW/finepdfs (config `sin_Sinh`)
- Serendib CPT Sinhala corpus: https://huggingface.co/datasets/Chamaka8/serendip-cpt-sinhala
- FineWeb-Edu (English): https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

**Papers**
- SinhalaMMLU (must be cited in the report): Pramodya et al., EMNLP 2025
- SinLlama (Sinhala CPT of Llama-3-8B, MADLAD + CulturaX corpus): https://arxiv.org/abs/2508.09115
- FineWeb2: "FineWeb2: One Pipeline to Scale Them All" (2025)
