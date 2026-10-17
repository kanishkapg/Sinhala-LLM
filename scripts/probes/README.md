# MCQ probes for CPT monitoring

The two multiple-choice probes of [docs/prepare_cpt_training_pipeline.md](../../docs/prepare_cpt_training_pipeline.md),
section 8.3. Run every command from the repo root with the `sin-llm` environment (it has `tokenizers`).

## Files (`data/probes/easy/`)

| File | Rows | Use |
|---|---:|---|
| `probe_a_subset.jsonl` | 504 | Probe A, 36 per subject. Every 50 steps. |
| `probe_a_full.jsonl` | 1,826 | Probe A, all usable questions. At the epoch boundaries. |
| `probe_a_excluded.jsonl` | 25 | Left out of Probe A, with the reason (18 near-copies of shared-task test questions, 7 with two identical options). |
| `probe_b.jsonl` | 405 | Probe B: `split` = `train` (305, from textbook sections that are in training) or `heldout` (100, from the 19 held-out chapters). |
| `probe_b_review.csv` | 60 | Random sample (40 train + 20 held-out) for the Sinhala reader. Opens in Excel. |
| `probe_b_authored/*.jsonl` | 405 | Probe B as written: correct answer, 3 distractors, evidence quote. Edit these, never `probe_b.jsonl`. |
| `probe_b_sources.jsonl` | 202 | The sections Probe B was written from (`role: alternate` = spares, unused). |

One JSON object per line, the SinhalaMMLU fields plus `id` and `probe`:

```json
{"id": "B-train-History-003", "probe": "B", "split": "train", "subject": "History", "category": "humanities",
 "question": "...", "choices": ["...", "...", "...", "..."], "answer": 2,
 "metadata": {"subject_original": "ඉතිහාසය", "difficulty": "easy", "grade": 7,
              "source_id": "history_g7_si/ch02/s05", "evidence": "<quote from the section>", "author": "claude-opus-5-5"}}
```

`answer` is always an int 1–4. Probe B gold positions are balanced (each position ~25% per split).

## Scoring (for the training callback)

```python
from probe_common import build_prompt          # scripts/probes/probe_common.py
prompt = build_prompt(item)                    # official Sinhala prompt, plain text, ends with "පිළිතුර:" + a space
# next-token logits at the end of the prompt; prediction = argmax over the token ids of "1".."4" (16..19)
```

The prompt ends with `පිළිතුර:` **followed by a space**: Qwen tokenizes `: 3` as `:` + ` ` + `3`, so after the
space the next token is the bare digit. `check_probes.py --tokenizer` verifies this.

Report Probe B per split (`train` vs `heldout`). The held-out set has no Art and more Islam (3 held-out
chapters), so for the train-vs-held-out comparison also report it on the 13 subjects both splits share.

## Rebuild and check

```bash
python scripts/probes/select_probe_b_sources.py --dump <folder>   # only to pick new sections (deterministic)
python scripts/probes/build_probes.py                             # authored/*.jsonl + Probe A sources -> data/probes/easy/
python scripts/probes/check_probes.py --tokenizer                 # must end with "OK: no errors"
```

## Probe B authorship

Probe B was written by Claude (`claude-opus-5-5`, a closed model) from the section text, one evidence quote per
question. Disclose this in the system report. Never put probe questions into SFT data.
