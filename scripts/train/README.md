# CPT training — `Qwen/Qwen3.5-4B-Base`, full-parameter, one 48 GB GPU

Code for [docs/prepare_cpt_training_pipeline.md](../../docs/prepare_cpt_training_pipeline.md) (the plan and the reasons
behind every setting). Run every command from the repo root. Train inside `tmux`, never from a notebook.

## Files

| File | What it does |
|---|---|
| `train_cpt.py` | The training run: data, model, `TrainingArguments` of section 4, callbacks, smoke / resume-test / resume modes, W&B reproducibility record |
| `cpt_data.py` | Packed train blocks read **in order**; per-source validation sets from `mix/val.jsonl`; collator with `labels = input_ids` |
| `cpt_callbacks.py` | Epoch copies (+ HF push), probes, stop-point gates, throughput / ETA / GPU-memory logging, stop-at-step |
| `probe_eval.py` | Probe A / Probe B / fixed samples. The same code runs in training and from the command line (reload test, epoch comparison, Base vs CPT) |
| `config/cpt_easy.json` | The Easy run (all settings of the plan) |
| `config/debug_tiny.json` | Dry run with a tiny random model on CPU / Colab / Kaggle |
| `config/sample_prompts.json` | The 5 fixed prompts of section 8.4 |
| `make_tiny_model.py` | Builds the tiny random Qwen3.5 (same tokenizer and layer mix) for the dry run; `--real-heads` keeps the 4B head sizes for GPU tests |
| `gpu_check.py` | GPU checks: DeltaNet kernels vs the PyTorch reference (outputs + gradients), 8-bit AdamW stochastic rounding |
| `colab_gpu_check.ipynb` | Free-GPU (Colab / Kaggle T4) check of the installs, `gpu_check.py` and the dry run with the real optimizer |
| `blend_weights.py` | Weighted average of models: the "free fix" of section 9 |
| `setup_machine.sh` | First-time setup on the g6e (section 10.5) |
| `launch.sh` | Runs `train_cpt.py` with a log file, then stops the machine (section 10.6) |
| `modal_cpt.py` | Runs any of these scripts on a Modal L40S, with `outputs/` on a Modal Volume (section "On Modal") |
| `requirements-train.txt` | Packages on top of the AMI's torch |

Outputs go to `outputs/` (git-ignored). For a run with `output_dir = outputs/cpt-easy-v1`:

```
outputs/cpt-easy-v1/
  checkpoint-325/ checkpoint-350/   full checkpoints (weights + optimizer + schedule + RNG + data position), last 2 kept
  epochs/e1 e2 e3/                  weights-only copies at steps 117 / 234 / 350 (+ cpt_info.json), pushed to hub_repo
  probes/step_0000.json ...         probe metrics, per-subject accuracy, samples; *_probe_a/b.jsonl = every prediction
  repro.json, pip_freeze.txt        reproducibility record (also in the W&B run)
  wandb_run_id.txt                  lets a resumed run continue the same W&B charts
```

## 1. Dry run off the GPU meter (section 9)

On your own machine, Colab or Kaggle (CPU is enough, ~3 minutes):

```bash
python scripts/train/make_tiny_model.py                                                    # -> outputs/tiny-qwen3.5
python scripts/train/train_cpt.py --config scripts/train/config/debug_tiny.json --stop-at-step 6
python scripts/train/train_cpt.py --config scripts/train/config/debug_tiny.json --resume
python scripts/train/train_cpt.py --config scripts/train/config/debug_tiny.json --smoke
```

12 steps over 3 tiny epoch segments: step-0 eval + probes, epoch copies at steps 4 / 8 / 12, both gates, the stop, the
resume. The numbers mean nothing (random model); kernel-fallback warnings are expected on CPU. Delete
`outputs/debug-tiny*` before the next dry run (an existing checkpoint makes a new run refuse to start without `--resume`).

**On a free GPU (Colab / Kaggle T4):** open [colab_gpu_check.ipynb](colab_gpu_check.ipynb)
(`https://colab.research.google.com/github/kanishkapg/Sinhala-LLM/blob/main/scripts/train/colab_gpu_check.ipynb`)
and run all cells: package installs, `gpu_check.py` (DeltaNet kernels vs the PyTorch reference, 8-bit AdamW
stochastic rounding), then the same dry run on the GPU with the real optimizer and kernels
(`make_tiny_model.py --real-heads` keeps the 4B head sizes). A T4 has no native bf16: bfloat16-only failures there
are the T4's limit (`--set model_dtype=float32 bf16=false` runs the rest).

## 2. On the g6e: setup and smoke test

```bash
bash scripts/train/setup_machine.sh          # once; then: hf auth login, wandb login
python scripts/train/gpu_check.py            # kernels + optimizer on this machine's torch / CUDA: all PASS
tmux new -s train
NO_SHUTDOWN=1 bash scripts/train/launch.sh --config scripts/train/config/cpt_easy.json --smoke
```

Set `hub_repo` in `config/cpt_easy.json` (e.g. `"<hf-user>/qwen3.5-4b-si-cpt-easy"`, created private) before the real
run. The smoke run (~10 min) evaluates step 0, probes 100 + 100 questions, trains `smoke_steps` (10) updates with the
real schedule, saves nothing, writes to `outputs/cpt-easy-v1-smoke`, and prints:

| Line | Pass if | If not |
|---|---|---|
| first training loss vs step-0 val loss | close; ≈ 12.4 (= ln 248,320) means a data/label bug | fix the setup |
| DeltaNet kernels, no fallback warning | both kernels installed, no "falling back" warning | install `flash-linear-attention` / `causal-conv1d` |
| GPU memory headroom ≥ 2 GB | peak reserved ≤ total − 2 GB | keep micro-batch 2 (or 1 + `global_batch` unchanged) |
| throughput / ETA | — | tokens/s → hours for the full run, evaluation included |

With headroom left, try `--set per_device_train_batch_size=4` (accumulation becomes 8, global batch stays 32).
Delete `outputs/cpt-easy-v1-smoke` afterwards.

## 3. Resume test, then the full run (section 6)

```bash
NO_SHUTDOWN=1 bash scripts/train/launch.sh --config scripts/train/config/cpt_easy.json --stop-at-step 25
bash scripts/train/launch.sh --config scripts/train/config/cpt_easy.json --resume
```

The first command trains to step 25, saves `checkpoint-25` and stops. The second resumes it: check in the log that it
starts at step 26 and that the loss carries on from the W&B curve, then let it run. It **is** the full run (no wasted
steps); `launch.sh` stops the machine when it ends or crashes (`sudo shutdown -c` within a minute cancels).

After a crash or a stopped machine: start it, `tmux new -s train`, run the `--resume` line again. After a loss spike:
delete the newest `checkpoint-*` (resume then uses the older one) and lower the LR with `--set learning_rate=1e-5`.
Keep batch size, sequence length and data unchanged across resumes.

## On Modal (no g6e capacity)

[modal_cpt.py](modal_cpt.py) runs steps 2–3 and 5 on a Modal L40S. Every command starts a new container, so there is no
tmux, no `launch.sh` and no shutdown: billing stops when the script ends. `outputs/` is the Volume `sinhala-cpt-outputs`.

```bash
pip install modal && modal setup                                     # once
modal secret create sinhala-cpt HF_TOKEN=hf_... WANDB_API_KEY=...    # once; HF token with write access
modal run scripts/train/modal_cpt.py --step check                    # builds the image (once), then gpu_check.py
modal run --detach scripts/train/modal_cpt.py --step smoke
modal run --detach scripts/train/modal_cpt.py --step stop25
modal run --detach scripts/train/modal_cpt.py --step resume          # the full run
modal run --detach scripts/train/modal_cpt.py --cmd "probe_eval.py --model outputs/cpt-easy-v1/epochs/e1 --out outputs/probes/e1"
modal volume ls sinhala-cpt-outputs /outputs/cpt-easy-v1
modal volume get sinhala-cpt-outputs /outputs/probes outputs/        # results to your machine
modal volume rm -r sinhala-cpt-outputs /outputs/cpt-easy-v1-smoke    # instead of deleting the smoke folder
```

Logs: the terminal while it stays attached, and always the app page on modal.com and W&B. `modal app stop sinhala-cpt`
ends a run. GPU containers can be preempted: Modal then restarts the same command, so `--step resume` carries on from the
last checkpoint by itself.

## 4. What is logged, and the stop points (sections 7-8)

W&B project `sinhala-mmlu-cpt`, `train/` and `eval/` panels:

| When | What |
|---|---|
| every step | loss, learning rate, grad norm, tokens/s, step time, ETA (training only), peak GPU memory |
| step 0, every 25 steps | `eval/textbook_loss`, `wiki_loss`, `fineweb2_loss`, `english_loss` |
| step 0, every 50 steps | `eval/probe_a_acc` (504), `probe_b_train_acc`, `probe_b_heldout_acc`, `*_shared` (13 common subjects), digit mass, answer-position share; `samples` table |
| step 0 + epoch boundaries | `eval/probe_a_full_acc` (1,826) |
| Check 1 / Check 2 | `eval/gate_check1`, `eval/gate_check2` (1 = pass) |

- **Check 1** runs at the first evaluation at or after 20 % of the steps (step 75): textbook loss down, replay losses
  within +5 % of step 0.
- **Check 2** runs at the end of epoch 1 (step 117): Probe B "in training" up by more than 2 SE, Probe A (all 1,826)
  not down by more than 2 SE.

A failed check prints a banner and sends a W&B alert (turn on e-mail / app alerts in the W&B settings). With
`"gate_action": "stop"` it also ends the run. The default is `"alert"`, because a noisy probe should not end a good
run while you sleep. The early dip in replay losses and Probe A during the first 10–20 % of steps is normal (8.7).

## 5. After the run

```bash
# reload test + all probes on each saved copy (same scoring as in training); also the base model as reference
python scripts/train/probe_eval.py --model Qwen/Qwen3.5-4B-Base --out outputs/probes/base
python scripts/train/probe_eval.py --model outputs/cpt-easy-v1/epochs/e1 --out outputs/probes/e1   # e2, e3 likewise
python scripts/train/probe_eval.py --model <hf-user>/<repo> --subfolder e3 --out outputs/probes/e3-hub
python scripts/train/probe_eval.py --compare outputs/probes/base outputs/probes/e1 outputs/probes/e2 outputs/probes/e3
```

- **Epoch count (section 5):** read the e1 / e2 / e3 columns: still rising after e3 → 4–5 textbook epochs; flat after
  e2 → 2; replay losses rising → fewer epochs, lower LR or more replay.
- **Free fix (section 9):** if general skills dropped,
  `python scripts/train/blend_weights.py --models outputs/cpt-easy-v1/epochs/e3 Qwen/Qwen3.5-4B-Base --weights 0.8 0.2 --out outputs/blend-0.8`,
  then probe it like the others.
- **Deciding test (8.8):** the same short LoRA SFT on the base and on the CPT model, then compare with `--compare`.
- Delete optimizer checkpoints only after the epoch copies reload and are on the Hub.

## Config keys (`config/cpt_easy.json`)

| Key | Default | Meaning |
|---|---|---|
| `model` | `Qwen/Qwen3.5-4B-Base` | base model (loads as text-only `Qwen3_5ForCausalLM`) |
| `output_dir`, `run_name` | `outputs/cpt-easy-v1` | on the EBS volume, never `/opt/dlami/nvme` |
| `hub_repo` | `null` | private HF repo for the epoch copies (`null` = no push) |
| `global_batch`, `per_device_train_batch_size` | 32, 2 | accumulation = 32 / 2 = 16 → 350 steps |
| `learning_rate`, `warmup_ratio`, `decay_ratio`, `min_lr_ratio` | 2e-5, 0.05, 0.2, 0.1 | WSD: 18 warm-up, 70 decay steps to 2e-6 |
| `optim`, `optim_args` | `adamw_torch_8bit`, `bf16_stochastic_round=True` | torchao 8-bit AdamW with stochastic rounding |
| `eval_steps`, `save_steps`, `save_total_limit` | 25, 25, 2 | validation losses and full checkpoints |
| `probe_steps`, `probe_full_at_start` | 50, true | probe cadence; full Probe A at step 0 as reference |
| `check1_fraction`, `replay_loss_tolerance`, `gate_action` | 0.2, 0.05, `alert` | stop points |
| `limit_blocks_per_epoch`, `val_limit_blocks`, `probe_limit`, `seq_len` | null / 2048 | dry-run shrinkers |
| `allow_slow_kernels` | false | refuse to train on the GPU without the DeltaNet kernels |

Any key can be overridden on the command line: `--set learning_rate=1e-5 run_name=cpt-easy-lr1e-5 output_dir=outputs/cpt-easy-lr1e-5`.

## Trap checklist (section 9) → where it is handled

| Trap | Handled by |
|---|---|
| labels = input_ids, no `DataCollatorForLanguageModeling` | `cpt_data.collate` |
| DeltaNet kernels | `train_cpt.py` refuses to start on a GPU without them; the smoke test reports any fallback warning |
| bf16 rounding | `optim=adamw_torch_8bit` + `bf16_stochastic_round=True` |
| sequential order | `train_sampling_strategy="sequential"`; `load_train` checks the epoch column is in order |
| tied embeddings | checked at load and before every epoch copy (alert if split or the parameter count changed) |
| reload test | `probe_eval.py` on each `epochs/eN` |
| base-model scoring | `probe_eval.digit_token_ids` checks that prompt + "1" tokenizes as prompt + the digit token |
| reproducibility record | `repro.json` + `pip_freeze.txt` in the output folder and the W&B run |
| `use_cache` | training runs with `use_cache=False`; epoch copies are saved with `True` for inference |
