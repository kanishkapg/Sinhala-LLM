"""Zero-shot SinhalaMMLU baseline for Qwen3.5-4B (or any Hugging Face chat LM), no fine-tuning.

Method = the official SinhalaMMLU evaluation (github.com/naist-nlp/SinhalaMMLU, src/): same Sinhala
prompt, same system message, and the prediction is the option digit with the highest next-token
logit. Nothing is sampled, so runs are deterministic and every question gets an answer.

Setup:  pip install torch "transformers==5.19.0" accelerate
Run:    python eval_baseline.py --data "SinhalaMMLU/TEST/**/*.json"
"""
import argparse
import glob
import json
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Verbatim from the official src/util_prompt.py (language="sinhala", intro_type=True).
SYSTEM = "You must output the Answer"
INTRO = "මෙය {} විෂයයට අදාළ බහුවරණ ප්‍රශ්නයකි.\n"
INSTRUCTION = {
    "easy": "පහත ප්‍රශ්නයට 1, 2, 3, 4 යන පිළිතුරුවලින් නිවැරදි හෝ ඉතාමත් ගැළපෙන පිළිතුර තෝරන්න.\n\n",
    "medium": "පහත ප්‍රශ්නයට 1, 2, 3, 4  යන පිළිතුරුවලින් නිවැරදි හෝ ඉතාමත් ගැළපෙන පිළිතුර තෝරාගන්න.\n\n",
    "hard": "පහත ප්‍රශ්නයට 1, 2, 3, 4, 5 යන පිළිතුරුවලින් නිවැරදි හෝ ඉතාමත් ගැළපෙන පිළිතුර තෝරාගන්න.\n\n",
}


def load_items(pattern: str) -> list[dict]:
    """Read every .json (a list of questions) or .jsonl file that matches the glob."""
    items = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, encoding="utf-8") as f:
            items += [json.loads(line) for line in f if line.strip()] if path.endswith(".jsonl") else json.load(f)
    if not items:
        raise SystemExit(f"No questions found for {pattern!r}")
    return items


def level_of(item: dict) -> str:
    return (item.get("level") or item["metadata"]["difficulty"]).lower()


def build_prompt(item: dict) -> str:
    """Official zero-shot prompt: subject intro, level instruction, question, numbered options."""
    subject = item.get("metadata", {}).get("subject_original") or item["subject"]
    options = "\n".join(f"{i}. {choice.strip()}" for i, choice in enumerate(item["choices"], start=1))
    return f"{INTRO.format(subject)}{INSTRUCTION[level_of(item)]}ප්‍රශ්නය: {item['question']}\n{options}\nපිළිතුර:"


@torch.inference_mode()
def predict(model, tokenizer, item: dict, digit_ids: list[int]) -> int:
    """Return the 1-based option whose digit token has the highest next-token logit."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(item)}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,  # Qwen3.5 thinks by default; then the next token is reasoning, not a digit
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    logits = model(**inputs, logits_to_keep=1).logits[0, -1]
    return int(logits[digit_ids[: len(item["choices"])]].argmax()) + 1


def report(items: list[dict], preds: list[int]) -> None:
    """Print per-level accuracy and the ranking metric (macro average over levels)."""
    hits = defaultdict(list)
    for item, pred in zip(items, preds):
        if str(item.get("answer", "")).strip().isdigit():  # the hidden test set ships without answers
            hits[level_of(item)].append(int(item["answer"]) == pred)  # int(): gold is often the string "2"
    if not hits:
        return print("No gold answers in the data: predictions written, nothing to score.")
    acc = {level: sum(h) / len(h) for level, h in hits.items()}
    for level in ("easy", "medium", "hard"):
        if level in acc:
            print(f"{level:>6}: {acc[level]:.2%}  (n={len(hits[level])})")
    print(f" macro: {sum(acc.values()) / len(acc):.2%}  (over {len(acc)} of 3 levels)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot SinhalaMMLU baseline")
    parser.add_argument("--data", required=True, help='glob, e.g. "SinhalaMMLU/TEST/**/*.json"')
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out", default="predictions.jsonl")
    args = parser.parse_args()

    items = load_items(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # For Qwen3.5 this resolves to the text-only Qwen3_5ForCausalLM: the vision tower and MTP head are not loaded.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto").eval()
    print(f"{type(model).__name__}: {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B params, {len(items)} questions")

    digit_ids = [tokenizer.encode(str(d), add_special_tokens=False) for d in range(1, 6)]
    assert all(len(ids) == 1 for ids in digit_ids), "each option digit must be a single token"
    preds = [predict(model, tokenizer, item, [ids[0] for ids in digit_ids]) for item in tqdm(items)]

    with open(args.out, "w", encoding="utf-8") as f:
        for item, pred in zip(items, preds):
            row = {key: item.get(key) for key in ("q_no", "subject", "category")}
            f.write(json.dumps(row | {"level": level_of(item), "answer": str(pred)}, ensure_ascii=False) + "\n")
    report(items, preds)


if __name__ == "__main__":
    main()
