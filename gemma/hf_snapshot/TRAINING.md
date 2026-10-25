# Further training from the published checkpoint

The repository contains full merged BF16 weights. Start a new optimizer/LoRA run from these weights; no original adapter is required. The original optimizer, scheduler and pre-interpolation LoRA state are not included.

## Runtime

The original LoRA implementation used Transformers 5.19.0, PEFT 0.21.2 and PyTorch 2.13.0+cu129, with two A100 40 GB GPUs. The supplied script also supports one CUDA GPU when the chosen sequence length and batch fit in memory.

Use the validated `vllm/vllm-openai:v0.32.0-cu129` environment, install `peft==0.21.2` if needed, and run commands from the downloaded model repository root. `requirements.txt` lists the package versions. The script uses CUDA, BF16, SDPA attention and gradient checkpointing.

## Data format

Provide a UTF-8 JSONL file with one MCQ per line:

```json
{"subject":"Mathematics","grade":6,"question":"දෙක සහ තුන එකතු කළ විට ලැබෙන අගය කීයද?","choices":["4","5","6","7"],"answer":2,"reasoning_en":"Adding 2 and 3 gives 5. The value 5 is the second option, so the answer is option 2."}
```

`answer` is a 1-based integer, and `choices` contains exactly four strings. `reasoning_en` is the desired correct reasoning trace. `subject` and `grade` are used with `--context`.

`examples/demo.jsonl` is a newly authored format example, not a meaningful training dataset. Replace it with your own correctly checked, appropriately licensed data. Keep an independent held-out evaluation set.

The script uses the parent's thinking chat template, masks the system/user prompt from the loss, and trains on the assistant's reasoning and final answer. Over-length examples are counted and skipped. Inspect this count and adjust `--max-len` if needed.

## Start a new LoRA run

This is a conservative starter configuration for a new dataset, not a claim that these hyperparameters are optimal:

```bash
python3 scripts/sft_lora.py \
  --model . \
  --train /path/to/your_train.jsonl \
  --out runs/continued \
  --ckpt-dir runs/continued/checkpoints \
  --epochs 1 --lr 1e-5 --rank 32 --alpha 32 \
  --micro-batch 1 --batch-tokens 8192 \
  --max-len 3072 --context
```

Use `--model sinhala-mmlu/gemma-4-e4b-sinhala-mmlu` to load directly from the Hub instead of the current local directory. To pin the parent release, first download `--revision v1.0.0` and train with `--model .` from that snapshot.

For multiple GPUs, replace `python3` with:

```bash
torchrun --standalone --nproc_per_node=2 scripts/sft_lora.py [the same arguments]
```

The script places LoRA only on language-model linear layers. Other parent weights remain frozen during the new LoRA run. The original base/interpolated model is already merged; do not reapply the original 50/50 interpolation automatically after your new run.

Optional `--docs /path/to/passages.jsonl --doc-len 2048` mixes plain language-model loss on JSONL rows with a `text` field alongside the MCQs. The script expects a non-empty MCQ training file; for a pure continued-pretraining objective, use a compatible language-model trainer directly with this full checkpoint.

## Resume your own interrupted run

With `--ckpt-dir`, the script saves LoRA parameters, optimizer, scheduler, step counter and training log to `ckpt.pt` every 60 steps by default. It reconstructs the minibatch order from the seed and step counter. Run the same command with the same data, settings and checkpoint directory to resume. Change the save interval with `--ckpt-every`.

That resume state belongs to your new run. It is separate from the original model-building run, whose optimizer state is not part of this release.

## Outputs and evaluation

- `runs/continued/merged/`: full weights with the new adapter merged, plus tokenizer/configuration and a new model card.
- `runs/continued/train_log.json`: training settings and loss history.
- `runs/continued/checkpoints/ckpt.pt`: resumable state when checkpointing is enabled.

Evaluate the new checkpoint using an independent MCQ JSONL file with `id`, `subject`, `question`, `choices` and optionally an integer `answer` and `grade`:

```bash
python3 scripts/bakeoff_run.py \
  --model gemma4_e4b --hf-id runs/continued/merged \
  --inputs /path/to/eval_directory --sets my_eval --out runs/continued/eval \
  --modes think --samples 1 --greedy --context \
  --think-budget 7168 --max-model-len 16384
```

The input is `/path/to/eval_directory/my_eval.jsonl`. Rows without gold answers yield predictions without an accuracy score. The parent's published validation scores do not apply to new weights; report your own measured results.

When sharing your new checkpoint, retain the Apache 2.0 license and base attribution, describe your additional modifications and data, and give it its own repository or version tag.
