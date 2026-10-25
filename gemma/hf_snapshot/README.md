---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
library_name: transformers
pipeline_tag: image-text-to-text
base_model: google/gemma-4-E4B-it
base_model_relation: finetune
language:
  - si
  - en
tags:
  - gemma4
  - sinhala
  - reasoning
  - multiple-choice
  - text-generation
  - lora
  - safetensors
model_name: Gemma 4 E4B SinhalaMMLU
---

# Gemma 4 E4B SinhalaMMLU

A Sinhala curriculum multiple-choice model derived from [Google Gemma 4 E4B-it](https://huggingface.co/google/gemma-4-E4B-it), using reasoning distillation, merged LoRA, and 50/50 weight interpolation with the base model (WiSE-FT).

**This repository contains the full merged BF16 checkpoint, tokenizer and chat template. No separate adapter or base-model download is needed to use the checkpoint.** You can start a new fine-tuning run directly from these weights; see [TRAINING.md](TRAINING.md).

- **Size:** approximately **7.94B total parameters/state elements**. The upstream `E4B` name does not mean this checkpoint has only 4B total parameters. It is above 7B and below 8B.
- **Primary task:** Sinhala school-curriculum MCQs, grades 6–13, with English reasoning and a numbered final answer.
- **Recommended inference:** native chat template with `enable_thinking=True`.
- **License:** Apache 2.0, following the [upstream Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license). Attribution and modification details are in [NOTICE](NOTICE) and [MODIFICATIONS.md](MODIFICATIONS.md).
- **Release checkpoint:** the same tensor weights as `SinhalaMMLU_final_model.zip`. [provenance.json](provenance.json) and [SHA256SUMS](SHA256SUMS) identify the files.

## Try the model

The validated runtime was `vllm/vllm-openai:v0.32.0-cu129`, with vLLM 0.32.0, PyTorch 2.13.0+cu129 and Transformers 5.19.0, on an A100 40 GB. [requirements.txt](requirements.txt) records the package versions. Memory requirements depend on context length, batching and backend.

Download the published snapshot:

```bash
hf download sinhala-mmlu/gemma-4-e4b-sinhala-mmlu --revision v1.0.0 --local-dir gemma-sinhala-mmlu
cd gemma-sinhala-mmlu
```

In the validated GPU environment, run the newly authored arithmetic example:

```bash
python3 scripts/bakeoff_run.py \
  --model gemma4_e4b --hf-id . \
  --inputs examples --sets demo --out results_demo \
  --modes think --samples 1 --greedy --context \
  --think-budget 7168 --max-model-len 16384
```

Predictions are written to `results_demo/gemma4_e4b/think_demo.jsonl`. The example is a format demonstration, not an evaluation set. If the vLLM image raises a native `torchcodec` import error for this text-only task, remove the unused package with `pip uninstall -y torchcodec`, as in the validation environment.

### Transformers example

```python
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

model_id = "sinhala-mmlu/gemma-4-e4b-sinhala-mmlu"
revision = "v1.0.0"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    revision=revision,
    dtype=torch.bfloat16,
    device_map="auto",
).eval()

messages = [
    {
        "role": "system",
        "content": (
            "You are an expert teacher of the Sri Lankan national school curriculum, "
            "fluent in Sinhala. Answer the multiple-choice question carefully. "
            "You may reason in English. Finish with 'Answer: <number>'."
        ),
    },
    {
        "role": "user",
        "content": (
            "විෂය (Subject): Mathematics | ශ්‍රේණිය (Grade): 6\n\n"
            "ප්‍රශ්නය (Question):\nදෙක සහ තුන එකතු කළ විට ලැබෙන අගය කීයද?\n\n"
            "විකල්ප (Options):\n1. 4\n2. 5\n3. 6\n4. 7\n\n"
            "Which option is correct? End your reply with 'Answer: <number>'."
        ),
    },
]
inputs = tokenizer.apply_chat_template(
    messages,
    enable_thinking=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=7168, do_sample=False)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=False))
```

The output can contain reasoning-channel markers before the final `Answer: N`. For the exact MCQ prompt, parsing and forced-option scoring used in evaluation, use the supplied `scripts/bakeoff_run.py`. The shorter Transformers example above demonstrates loading and generation; it is not the complete benchmark harness.

## Validation results

Measured on 24 October 2026 using the checkpoint in this repository, BF16, seed 1234, one greedy thinking sample, a 7,168-token budget, subject/grade metadata, and no option shuffling:

| Set | Questions | Accuracy |
|---|---:|---:|
| Locked SinhalaMMLU Easy development subset | 600 | 67.33% |
| Global-MMLU Sinhala harder-domain proxy | 1,000 | 74.30% |
| Equal-weight mean of those two sets | 1,600 total | **70.82%** |

**This is not the official SinhalaMMLU Easy/Medium/Hard competition score.** The Global-MMLU subset is a proxy, not the official Hard test. The locked 600 Easy questions were excluded from training. An exact ID and normalized question/options overlap check found no overlap with the final distillation training file.

Gemma native direct-answer scoring without reasoning averaged 51.37% on the same two primary sets; using a raw completion prompt averaged 36.10%. Enable the reasoning template to reproduce the intended inference behavior.

The model also scored 87.18%, 82.62%, 87.64% and 83.69% on supplementary generated curriculum Easy/Medium/Hard/Aesthetic sets (195/282/267/282 questions). These were generated using Gemma teachers and may favor this model, so they are excluded from the primary average.

All 2,626 reasoning outputs closed their thinking block and produced a parseable answer; none reached the token cap. Generation plus final option scoring took 553.3 seconds in the measured A100 batch run, excluding startup, weight transfer and verification. This is not single-request latency. Aggregate results and input fingerprints are in [evaluation/metrics.json](evaluation/metrics.json) and [evaluation/input_fingerprints.json](evaluation/input_fingerprints.json). Evaluation questions and answers are not distributed here.

The earlier submission recipe used 16 sampled votes, option shuffling, subject/grade context and a 7,168-token budget. That is a different inference configuration from the one-sample greedy validation numbers above. The included evaluator supports that recipe with `--samples 16 --shuffle --context --think-budget 7168`, without `--greedy`.

## How this checkpoint was built

1. Sinhala curriculum material from public textbook, teacher-guide and syllabus sources was cleaned and converted to Unicode where needed.
2. Curriculum MCQs were generated from source passages. Closed models (Codex/gpt-6.1-sol and Claude Sonnet) were used for training-question generation/checking; they are not used at inference.
3. Open Gemma 4 26B-A4B and 31B teachers produced reasoning traces without the source passages in the student's prompt. Correct, completed traces were retained.
4. The final training file contained **7,197 traces over 4,980 unique questions**. This included 725 unique SinhalaMMLU development questions outside the locked 600-question evaluation subset; the remaining questions were generated curriculum questions.
5. LoRA rank/alpha 64, dropout 0.05, all language-model linear layers, two epochs, learning rate `5e-5`, maximum sequence length 3,072 and micro-batch 1 per GPU. Vision/audio towers were not fine-tuned.
6. The adapter was merged and the weights interpolated as `W_final = W_base + 0.5 * (W_distilled - W_base)`.

Only one merged model is loaded for inference. There is no retrieval, tool use or teacher-model call in the inference path.

## Continue training

See [TRAINING.md](TRAINING.md) for a new LoRA run, dataset schema, checkpointing and merging. The full weights can also be used with another compatible fine-tuning framework.

This is a model-weight checkpoint, not an exact training-resume checkpoint: it does not contain the original optimizer/scheduler state or the unmerged pre-interpolation adapter. Further training starts with a new optimizer from the published merged weights. The supplied script can save resumable state for your own new run.

## Scope and limitations

- The evaluated specialization is text-based Sinhala MCQs. General chat, vision and audio quality were not evaluated for this derivative; retained multimodal weights come from the base model.
- Responses can be incorrect. English reasoning is expected even when the question is in Sinhala.
- The validation numbers are for this exact release and inference configuration; they do not transfer automatically to a further fine-tune.
- Original training and evaluation datasets are not part of this repository and retain their respective licenses and access conditions. Obtain suitable data from its original sources for your own training.

Base-model credit: Google/Google DeepMind, [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it). This is a community derivative maintained by the `sinhala-mmlu` organization, not an official Google release.
