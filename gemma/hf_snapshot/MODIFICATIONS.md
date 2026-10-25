# Modification notice

The model is derived from `google/gemma-4-E4B-it`. It is a community fine-tune, not an unchanged copy of the upstream model.

The checkpoint shards distributed here contain the merged Sinhala MCQ reasoning-distillation LoRA and the subsequent 50/50 WiSE-FT interpolation:

```text
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
```

The language-model linear weights were fine-tuned. Vision/audio towers were retained from the upstream model. Transformers serialization omitted unused KV-shared-layer entries that the saved model does not require.

All four shards are byte-identical to the corresponding files in the validated `SinhalaMMLU_final_model.zip`. Their SHA-256 hashes are recorded in `provenance.json` and `SHA256SUMS`.

The tokenizer, chat template, model configuration and generation settings are unchanged from the validated archive. The informational `soup.json` has been rewritten to replace an environment-specific training path with a public description; this does not alter any model tensor.

This release adds a model card, Apache 2.0 license/attribution, aggregate evaluation results, a newly authored arithmetic format example, and training/inference scripts. The evaluator adds a `--greedy` option and defaults to this published checkpoint. The LoRA trainer accepts a Hub ID or local checkpoint path, retains attribution files, and writes a fresh model card for a further fine-tune.

No original training/evaluation question files, optimizer state, access credentials or unmerged pre-interpolation adapter are included.
