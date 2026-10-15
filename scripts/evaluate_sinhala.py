import os
import glob
import json
import zipfile


def load_env(path=".env"):
    """Put HF_TOKEN (and friends) into the environment before the Hub is touched.

    Gemma 3 is a gated repo, so huggingface_hub needs a token with access granted.
    An env var takes priority over the token cached by `hf auth login`, which means
    this also works around a stale/revoked token sitting in ~/.cache/huggingface.
    """
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))



load_env(override=True)  # put HF_TOKEN into the environment before the Hub is touched

import torch
from huggingface_hub.errors import GatedRepoError
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# 1. Load model and processor
# google/* is gated and needs the licence accepted on the model page by the account
# owning HF_TOKEN. Set MODEL_ID to override, e.g. the ungated unsloth/gemma-3-4b-it
# mirror of the same weights.
model_id = os.environ.get("MODEL_ID", "google/gemma-3-4b-it")

if model_id.startswith("google/") and not os.environ.get("HF_TOKEN"):
    raise SystemExit(
        f"{model_id} is gated but HF_TOKEN is not set. Put a token in .env as "
        "HF_TOKEN=hf_..., or set MODEL_ID to an ungated mirror."
    )

print(f"Loading model and processor: {model_id}")
try:
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,  # Gemma 3 overflows in float16
    ).eval()
except (GatedRepoError, OSError) as e:
    if "gated" not in str(e).lower():
        raise  # unrelated I/O problem: let the real traceback through
    raise SystemExit(
        "\n".join([
            f"No access to the gated repo {model_id}.",
            f"  1. Accept the licence at https://huggingface.co/{model_id} with the",
            "     account that owns HF_TOKEN, then rerun; or",
            "  2. Run against the ungated mirror of the same weights:",
            '       $env:MODEL_ID = "unsloth/gemma-3-4b-it"',
            f"Hub said: {e}",
        ])
    ) from e
processor = AutoProcessor.from_pretrained(model_id)


# 2. Check the parameter budget (shared-task limit: 8B)
def count_params(*models) -> int:
    seen, total = set(), 0
    for m in models:
        for p in m.parameters():
            if p.data_ptr() not in seen:      # tied weights counted once
                seen.add(p.data_ptr())
                total += p.numel()
    return total


n_params = count_params(model)
print(f"Parameters: {n_params:,} ({n_params / 1e9:.2f}B)")
assert n_params <= 8_000_000_000, "Over the 8B parameter limit"

# Token ids of the answer numbers "1".."5"
choice_ids = [processor.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]


def build_prompt(item):
    """Zero-shot Sinhala prompt, same template as the official SinhalaMMLU code."""
    subject = item.get("metadata", {}).get("subject_original", item["subject"])
    labels = ", ".join(str(i) for i in range(1, len(item["choices"]) + 1))
    options = "\n".join(f"{i}. {choice.strip()}" for i, choice in enumerate(item["choices"], 1))
    return (
        f"මෙය {subject} විෂයයට අදාළ බහුවරණ ප්‍රශ්නයකි.\n"
        f"පහත ප්‍රශ්නයට {labels} යන පිළිතුරුවලින් නිවැරදි හෝ ඉතාමත් ගැළපෙන පිළිතුර තෝරාගන්න.\n\n"
        f"ප්‍රශ්නය: {item['question']}\n{options}\nපිළිතුර:"
    )


@torch.inference_mode()
def predict(item):
    """Pick the option number the model gives the highest probability (no text generation)."""
    messages = [{"role": "user", "content": [{"type": "text", "text": build_prompt(item)}]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    logits = model(**inputs).logits[0, -1].float()          # scores for the model's first answer token
    n_choices = len(item["choices"])
    probs = torch.softmax(logits[choice_ids[:n_choices]], dim=-1)
    return int(probs.argmax()) + 1, probs.tolist()


# 3. Evaluation loop over each difficulty folder
data_root = "data/SinhalaMMLU/TEST"   # contains easy/, medium/, hard/ with *.json files
levels = ["easy", "medium", "hard"]

accuracy = {}
predictions = []

for level in levels:
    json_files = sorted(glob.glob(os.path.join(data_root, level, "*.json")))
    if not json_files:
        print(f"No JSON files in {os.path.join(data_root, level)}, skipping.")
        continue

    correct, graded = 0, 0
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"\n[{level}] {os.path.basename(file_path)}: {len(data)} questions")

        for item in data:
            predicted, probs = predict(item)
            predictions.append({
                "q_no": str(item["q_no"]),
                "subject": item["subject"],
                "category": item.get("category", ""),
                "level": level,
                "answer": str(predicted),
            })

            gold = item.get("answer")
            if gold not in (None, ""):                   # the hidden test set has no answers
                gold = int(gold)                         # answers may be stored as "2" or 2
                correct += predicted == gold
                graded += 1
            print(f"  Q{item['q_no']}: gold={gold} pred={predicted} probs={[round(p, 2) for p in probs]}")

    if graded:
        accuracy[level] = correct / graded
        print(f"[{level}] accuracy: {accuracy[level]:.2%} ({correct}/{graded})")

# 4. Results and submission file
print("\n" + "=" * 40)
for level, acc in accuracy.items():
    print(f"{level:>6}: {acc:.2%}")
if accuracy:
    print(f" macro: {sum(accuracy.values()) / len(accuracy):.2%}   (official metric)")
print("=" * 40)

with open("predictions.jsonl", "w", encoding="utf-8") as f:
    for row in predictions:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
with zipfile.ZipFile("predictions.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write("predictions.jsonl")
print(f"Saved {len(predictions)} predictions to predictions.zip")