# SFT dataset pipeline — how to run it (Easy)

Scripts that implement [docs/prepare_sft_dataset.md](../../docs/prepare_sft_dataset.md): cleaned textbook sections
→ synthetic Sinhala MCQs written by Claude → automatic checks → blind validation → dedup → bias audit → human
review → tokenised train/val files for the SFT run.

Every stage writes its own folder under `data/processed/sft/easy/`. Stages that call Claude **append** each finished
job immediately and **resume** where they stopped. Everything else rewrites its output from the previous stage, so
any stage can be re-run. Deterministic: seed 42.

## 0. Setup (once)

- **Python:** use the `sin-llm` conda env. It has `tokenizers`, `pyarrow` and `datasketch`; nothing else is
  needed. In PowerShell:

  ```powershell
  cd "D:\Competetions\Sinhala MMLU"
  conda activate sin-llm
  $env:PYTHONUTF8 = "1"          # Sinhala in the console
  ```

- **Claude Code CLI:** the scripts find the copy bundled with the VS Code extension
  (`~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude.exe`). To use another copy, set
  `$env:CLAUDE_EXE` or `guard.claude_exe` in `config/sft_easy.json`.
- **Log in with your claude.ai Pro account:** `claude auth status` must show `"authMethod": "claude.ai"` and
  `"subscriptionType": "pro"`.
- **Turn extra usage off:** claude.ai → Settings → Usage → Extra usage: **off**. The scripts check this on every
  call and stop if they can't confirm it.

## 1. How the $20 Pro plan is protected (no extra charges)

Generation (s02) and validation (s05) run through the **Claude Code CLI on your claude.ai subscription**, never
through the paid API. `claude_runner.py` enforces this:

| Guard | What it does |
|---|---|
| No API key reaches the CLI | `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL` and the Bedrock/Vertex/Foundry switches are removed from the child process. `--bare`, which forces API-key auth, is never used. |
| Pre-flight | Stops unless `claude auth status` reports `authMethod: claude.ai` and a `pro`/`max` subscription. |
| Extra usage must be OFF | Every CLI response reports `fast_mode_disabled_reason`. Unless it is `extra_usage_disabled`, the run stops **before the next call**. (Verified on your account on 2026-10-23: it is off.) |
| Limit reached → wait | When the 5-hour or weekly limit hits, the CLI refuses the call (nothing is billed). All workers pause until the reset time in the message, or 30 min if none is given, then continue by themselves. Ctrl+C is safe at any time. |
| Optional own cap | Set `guard.max_notional_usd_per_window` (e.g. `2.0`) to stop early in each 5-hour window and keep part of your Pro allowance for chat. It is measured with the CLI's `total_cost_usd`, an API-price *estimate*: nothing is charged per call on the subscription. |

**Throughput.** Pro limits are not published in tokens or dollars. The full Easy set is ~2,900 generation jobs
plus ~1,000 validation calls, which will take **many** 5-hour windows. Wave 0 and the first limit hit show
the real rate: `usage_summary` prints "calls before the first limit hit". Use that number to decide what fits
before the deadline:

- wave 1 only = the pilot;
- `generator.model` → `sonnet`, which uses the Pro allowance more slowly than Opus;
- `generator.effort` → `medium`.

If your plan can't use Opus in Claude Code, the first call fails with a model error: set `"model": "sonnet"` for the
generator and `"haiku"` or `"sonnet"` for the validator. **Keep the two models different.**

## 2. Run order

All commands run from the repo root; all take `--level easy` (the default).

| # | Command | Writes | Claude? | Time |
|---|---|---|---|---|
| 1 | `python scripts/sft/s01_plan.py` | `00_plan/` sections, quotas, 2,924 jobs (7,594 slots) | no | ~10 s |
| 2a | `python scripts/sft/s02_generate.py --wave 0 --dry-run` | prints one full prompt | no | 1 s |
| 2b | `python scripts/sft/s02_generate.py --wave 0` | `01_generated/wave0.jsonl` (26 jobs, ~100 questions) | **yes** | ~15–40 min |
| 3 | `python scripts/sft/s03_assemble.py` | `02_items/items_raw.jsonl`, `skipped.jsonl` | no | s |
| 4 | `python scripts/sft/s04_check.py` | `03_checked/passed.jsonl`, `rejected.jsonl` | no | s |
| 5 | `python scripts/sft/s05_validate.py --wave 0` | `04_validated/` + accept/reject decisions | **yes** | ~5–15 min |
| 6 | `python scripts/sft/s06_dedup.py` | `05_final/items.jsonl`, `dedup_report.json` | no | ~1 min |
| 7 | `python scripts/sft/s07_audit.py --waves 0` | `05_final/audit.json` (gates) | no | s |
| 8 | `python scripts/sft/s08_review.py sample --round wave0 --waves 0 --all` → review → `... report --round wave0` | `review/` | no | human: 1–2 h |
| — | **Gate: wave 0.** Continue only if it passes (section 3). | | | |
| 9 | `s02 --wave 1` → `s03` → `s04` → `s05 --wave 1` → `s06` → `s07 --waves 1` → `s08 sample --round pilot --waves 1` → review → `s08 report --round pilot` | the pilot | **yes** | depends on Pro limits |
| 9b | *(optional)* choices-only probe (section 4) | | GPU | ~10 min |
| 10 | `python scripts/sft/s09_render.py --variant pilot` | `06_rendered/pilot/` **train-ready** | no | ~1 min |
| 11 | `s02 --wave 2` → `s03` → `s04` → `s05 --wave 2` → `s06` | | **yes** | |
| 12 | `python scripts/sft/s01_plan.py --topup` → `s02 --wave 3` → `s03` → `s04` → `s05 --wave 3` → `s06` | fills the gaps | **yes** | |
| 13 | `s07_audit.py` (all waves) → `s09_render.py --variant final` | `06_rendered/final/` **train-ready** | no | |

Useful flags:

- `--workers 2`: parallel CLI calls. The limit is shared, so more workers only finish faster.
- `--max-calls N`: try a few calls first.
- `--model`, `--effort`: override the config for one run.
- `s05 --decide-only`: re-apply the accept rule without any calls.

## 3. Where you must look (manual checks)

| # | When | What to check | If it fails |
|---|---|---|---|
| **M1** | **before wave 0** | **The 9 style exemplars** in `config/exemplars_easy.json`. They were written by Claude; a Sinhala reader must confirm the Sinhala is natural, each key is right, and the distractors are fair. Then set `"approved": true`. They are shown to the generator in **every** call, so a mistake in them repeats thousands of times. | Fix the text (each `evidence` must stay a verbatim quote: s01 checks it). |
| **M2** | after s01 | The subject table: no subject with 0 slots, Art ~246, Geography ~315, total ~7,600. | — |
| **M3** | after s02 wave 0 | Read the `--dry-run` prompt once, and 10 raw questions in `wave0.jsonl`. | Edit `prompts/generator.md`. |
| **M4** | after s04 | Reasons flagged "above 10%": the prompt is causing them. Common ones: `evidence_not_in_section` (the quote was paraphrased), `key_much_longer`, `clang_cue`, `fm_artefact`. | Fix the prompt, then **move `wave0.jsonl` away and regenerate wave 0**. |
| **M5** | after s05 | Acceptance ≥ 75%; any subject or type with key disagreement > 15%: open 20 of them in `04_validated/rejected.jsonl` and decide who was wrong, the generator or the validator. | Generator wrong → `generator.md`; validator wrong → `validator.md`. |
| **M6** | after s06 | `dedup_report.json → flagged`: items similar to a test, Probe A or Probe B question. Expect almost none. If one is a real copy, add its id to `05_final/audit_drops.jsonl`. | — |
| **M7** | after s07 | Every gate PASS. `key_longest` failing → `s07_audit.py --fix-length`, then re-run the audit. `qtype_shares` / `subject_shares` / `chapter_coverage` failing → top-up wave (step 12). For the pilot, audit with `--waves 1` (coverage counts only wave-1 chapters). | — |
| **M8** | s08, wave 0 and pilot | **The Sinhala-reader review.** Open `review/review_<round>.csv` in Excel and fill in `key_correct`, `single_best`, `natural_sinhala`, `fair_distractors` (Y/N), `cue` (Y/N) and `comment`. Save as **CSV UTF-8**. Gate: ≥ 90% of accepted items fully good. The `rejected` rows show whether the validator is too strict. Items you mark bad go to `review/review_drops.jsonl` and are left out. | Below 90%: fix the prompts before the next wave. |
| **M9** | after s09 | The printed example prompt must end with `පිළිතුර: ` followed by the digit; labels 25% each; Art and Health use several subject strings. | — |
| **M10** | at the end | Record the closed-model use for the system report: `manifest.json` has the generator and validator models, the prompt hashes and the dates. Copies of the exact prompts are in `01_generated/prompts/`. | — |

**Changing a prompt after wave 0:** prompts are hashed into every record. Regenerate wave 0 with the new prompt, or
keep it and note the change in the report. Later waves use the frozen prompt.

## 4. Optional: the choices-only probe (audit gate)

This shows whether the options alone give the answer away. It needs the 4B model, so run it on Modal:

```powershell
python scripts/sft/s07_audit.py --export-choices-only 1000
# add "data/processed/sft/easy/05_final/choices_only" to DATA_PATHS in scripts/train/modal_cpt.py,
# commit + push, set DATA_REV to the new commit, then:
modal run scripts/train/modal_cpt.py --cmd "probe_eval.py --model kanush21/qwen3.5-4b-si-cpt-easy --subfolder e3 --probe-dir data/processed/sft/easy/05_final/choices_only --out outputs/probes/choices_only"
modal volume get sinhala-cpt-outputs /outputs/probes/choices_only outputs/probes/
python scripts/sft/s07_audit.py --waves 1 --choices-only-summary outputs/probes/choices_only/summary.json
```

In the summary, "probe_b" is the SFT sample and "probe_a" is Probe A, both scored without their questions.

## 5. What you get

- `06_rendered/<variant>/train.parquet`, `val.parquet` hold `input_ids` (prompt + answer digit) and `labels`
  (`-100` on the prompt, the digit id 16–19 as the only target), plus `prompt`, `completion`, `answer`,
  `subject`, `qtype`, `view` and `n_tokens`.
- `train.jsonl` and `val.jsonl` are the same rows without the tokens: readable, and usable as TRL
  prompt–completion data.
- The training collator must pad with `labels = -100` and `attention_mask = 0`. Don't rely on the pad id: in
  Qwen3.5, pad = EOS.

Expected sizes:

| Variant | Items | Training examples |
|---|---|---|
| Pilot | ~3,000 | ~5,800 (2 views per item) |
| Final | ~6,000 | ~11,500 |

Median ~200 tokens per example.

## 6. Files

| File | Role |
|---|---|
| `sft_common.py` | paths (`SFT_DATA_ROOT` env var redirects all outputs, for tests), config, helpers |
| `claude_runner.py` | guarded Claude CLI calls (section 1) |
| `config/sft_easy.json` | quotas, models, guard settings, check thresholds, gates |
| `config/exemplars_easy.json` | 9 authored style exemplars (needs approval, M1) |
| `prompts/generator.md`, `prompts/validator.md` | the instructions sent to Claude |
| `s01`–`s09` | the stages above |

Tested on 2026-10-23:

- **Offline:** s01–s09 ran on fake generator and validator output in a scratch folder.
- **Live:** one guarded call each to s02 and s05 (Haiku, scratch folder). Sinhala in and out was intact, the
  structured output was correct, and the extra-usage check passed.

The full-size Opus and Sonnet runs, and a real usage-limit wait, have not been exercised yet.
