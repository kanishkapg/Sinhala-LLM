# CPT dataset pipeline — how to run it

Scripts that implement [docs/prepare_cpt_dataset.md](../../docs/prepare_cpt_dataset.md): textbook PDFs in
`data/raw/gN/` → tokenized, packed Parquet blocks for continued pre-training of `Qwen/Qwen3.5-4B-Base`.

Every stage reads the previous stage's files and writes its own folder under
`data/processed/cpt/<level>/`, so any stage can be re-run on its own. The scripts are deterministic (seed 42).

## 0. Setup (once)

```powershell
cd "D:\Competetions\Sinhala MMLU"
python -m venv .venv-cpt
.\.venv-cpt\Scripts\Activate.ps1
pip install -r scripts/cpt/requirements-cpt.txt
huggingface-cli login            # optional, but makes the replay downloads faster
```

For the scanned book (`g8/health g-8 S.pdf`), install **Tesseract 5**: on Windows, use the UB Mannheim installer.
Then copy `sin.traineddata` from <https://github.com/tesseract-ocr/tessdata_best> into Tesseract's `tessdata` folder.

For the human review, install **Label Studio** in a separate environment: `pip install label-studio`.

Run every command from the repository root.

## 1. Run order

| # | Command | Writes | Time (Easy) |
|---|---|---|---|
| 1 | `python scripts/cpt/s01_inventory.py --level easy` | `00_inventory/`, `config/books_easy.json` | ~30 s |
| 2 | `python scripts/cpt/s02_extract.py --level easy` | `01_extracted/*.jsonl`, `leftovers.json` | ~1.5 min |
| 3 | `python scripts/cpt/s03_ocr.py --level easy` | adds OCR lines to `01_extracted/` | ~20–40 min (216 pages) |
| 4 | `python scripts/cpt/s04_clean.py --level easy` | `02_clean/*.jsonl` + `*.txt` | ~10 s |
| 5a | `python scripts/cpt/s05_review.py wordlist` (once) | `data/processed/cpt/common/si_wordfreq.json` | ~1 min |
| 5b | `python scripts/cpt/s05_review.py auto --level easy` | `03_review/auto_scores.jsonl` | ~5 s |
| 5c | `python scripts/cpt/s05_review.py sample --level easy --round 1` | `03_review/round1/`, `03_review/pages/` | ~1 min |
| 5d | *human review in Label Studio or the CSV* | your export file | a few hours |
| 5e | `python scripts/cpt/s05_review.py report --level easy --round 1 --labels <export>` | `03_review/review_status.json` | seconds |
| 6 | `python scripts/cpt/s06_build_docs.py --level easy` | `docs.jsonl` | ~10 s |
| 7 | `python scripts/cpt/s07_replay.py --level easy` | `replay/*.jsonl` | 10–60 min (network) |
| 8 | `python scripts/cpt/s08_dedup_decontam.py --level easy` | `04_dedup/*.jsonl`, `report.json` | ~1 min |
| 9 | `python scripts/cpt/s09_mix.py --level easy` | `mix/train.jsonl`, `mix/val.jsonl` | ~1 min |
| 10 | `python scripts/cpt/s10_pack.py --level easy --block 2048` | `packed/train/*.parquet`, `packed/val/`, `manifest.json` | ~1 min |

Stage 7 does not depend on stages 1–6. You can start it early and let it run in the background.

## 2. Where to stop for manual review

| Stop | After | What to do | Then |
|---|---|---|---|
| **A** | s01 | If it prints **UNKNOWN FONTS**: classify each font in `config/fonts.json` (`convert` = legacy FM Sinhala, `keep` = Unicode/Latin, `drop` = Tamil/Arabic/dingbats). Use the sample text it prints: ASCII gibberish like `yels±hs` means legacy Sinhala. If it prints **MISSING SUBJECTS**: download those textbooks into `data/raw/gN/`, named like `drama g-6 S.pdf`. | re-run s01 |
| **B** | s02 | Read the **"Characters left after conversion"** list (`01_extracted/leftovers.json`). Fix each repeated one in `fm_converter.py` (`PRE` for FM codes, `POST` for Unicode output), or note why it is fine (e.g. Ω, ¼, roman numerals). | `s02 --force` |
| **C** | s04 | Open 2–3 `02_clean/*.txt` files next to the PDFs. Check chapters (`# ` lines), headings (`## `), paragraph joins, removed captions and labels. Fix in `config/books_easy.json`: `skip_pages`, `chapter_pages` (missed chapter openers, e.g. `{"12": 1, "17": 2}` for buddhism_g6), `chapter_min_size`. Then read the warnings s04 prints. | re-run s04 for those books: `--books <id>` |
| **D** | 5c → 5e | **The Sinhala reader review** (section 3 below). Books that fail: add the reviewer's `wrong -> right` fixes to `fm_converter.py POST` (or cleaning rules), then re-run. | see "restart" |
| **E** | s07 | Read the per-filter counts. If a source keeps too little, check `stats.json`. Open 20 random lines of each `replay/*.jsonl` and make sure they are real prose, not menus or ads. | `s07 --only <src> --force` |
| **F** | s08 | Check `04_dedup/report.json`. Replay documents that overlap the test are already dropped. Note the textbook overlap rate for the system report. | — |
| **G** | s10 | Read the 3 decoded blocks it prints: clean Sinhala, headers like `ශ්‍රේණිය: 7 \| විෂය: ඉතිහාසය \| …`, EOS between documents. Check the token shares in `mix/summary.json` against `config/mixture.json`. | done |

## 3. The human review (Stop D)

1. `s05_review.py sample` picks 20 random + 20 "worst" paragraphs per book (+10 more for OCR'd books). It renders the page images and writes:
   - `03_review/round1/tasks_labelstudio.json`
   - `03_review/round1/review_sheet.csv`, for Google Sheets / Excel if you don't use Label Studio.
2. Label Studio:
   ```powershell
   $env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"
   $env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="D:\Competetions\Sinhala MMLU"
   label-studio start
   ```
   - Create a project.
   - Under *Labeling Interface → Code*, paste `scripts/cpt/config/labelstudio.xml`.
   - Under *Cloud Storage → Add Source Storage → Local files*, use the path `D:\Competetions\Sinhala MMLU\data\processed\cpt\easy\03_review\pages`.
   - Import `tasks_labelstudio.json`.
3. For each sample, the reviewer compares the text with the page image and does three things:
   - choose **OK / Minor errors / Broken**;
   - tick the error types;
   - write fixes as `wrong -> right`, one per line, e.g. `ගැටලූව -> ගැටලුව`.
4. Export the results (*Export → JSON*) and run `s05_review.py report ... --labels export.json`.

   A book passes with **≥ 95 % OK and no `wrong -> right` fix seen twice**.

## 4. Restarting after a fix

Re-run only the stage you fixed and everything after it.

| You changed | Re-run |
|---|---|
| `config/fonts.json` or `fm_converter.py` | `s02 --force` (or `--books <id>`) → s03 (OCR'd books only, uses its cache) → s04 → s05 auto → new review round for the failed books → s06 → s08 → s09 → s10 |
| `config/books_easy.json` (skip / chapter pages) | s04 → s05 auto → s06 → s08 → s09 → s10 |
| Review verdicts only | s05 report → s06 → s08 → s09 → s10 |
| `config/replay_sources.json` or replay budgets | `s07 --force` (or `--only <src>`) → s08 → s09 → s10 |
| `config/mixture.json` only | s09 → s10 (if budgets went up: s07 → s08 first) |
| Block size | s10 only |

**New review round for failed books:**
- `s05_review.py sample --level easy --round 2`. It samples again only the books that have not passed, with a new seed.
- Review, then `report --round 2`.

**Quick dry run before the review:**
- `s06_build_docs.py --allow-unreviewed` includes every book, so you can test s07–s10 end to end.
- `s07_replay.py --limit-tokens 50000` collects only a small replay sample.

**Medium / Hard:** put the g9–g11 (or g12–13) PDFs in `data/raw/g9…`, then run the same commands with `--level medium` (or `hard`). `mixture.json` already makes the Medium and Hard runs cumulative: their `textbook_levels` include the earlier levels.

## 5. Final train-ready output

```
data/processed/cpt/easy/packed/train/train-00000.parquet ...   input_ids: list<int32>[2048], epoch: int16
data/processed/cpt/easy/packed/val/val-00000.parquet
data/processed/cpt/easy/manifest.json   (sources, licences, counts, converter patches, tokenizer hash ...)
```

Load it with `datasets.load_dataset("parquet", data_files={"train": ".../packed/train/*.parquet", "validation": ".../packed/val/*.parquet"})`. Labels are `input_ids`.

## Measured on the Easy books (2026-10-15, before the human review)

- 32 PDFs: 31 text PDFs plus 1 scanned (health g8, needs OCR).
- After cleaning: 3.67 M characters → **2,547 sections, 2.82 M Qwen3.5 tokens** (median 1,070 tokens per section).
- Characters left unconverted in FM spans: about 90 out of 3.3 M.
- With 3 textbook epochs, the textbook part of the mixture is ~8.5 M tokens.
- Only 4 textbook sections share a 13-gram with a visible test question.
