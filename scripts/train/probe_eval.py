"""MCQ probes + fixed sample outputs for CPT checkpoints (docs/prepare_cpt_training_pipeline.md, sections 8.3-8.4).

The same scoring code runs everywhere, so step 0, every training step and every saved model are comparable:
  - inside training, through ProbeCallback (scripts/train/cpt_callbacks.py);
  - from the command line, on a saved model: the reload test, the epoch-boundary comparison (section 5) and the
    Base vs CPT comparison (section 8.8).

    python scripts/train/probe_eval.py --model Qwen/Qwen3.5-4B-Base --out outputs/probes/base
    python scripts/train/probe_eval.py --model outputs/cpt-easy-v1/epochs/e1 --out outputs/probes/e1
    python scripts/train/probe_eval.py --model <hf-user>/<repo> --subfolder e2 --out outputs/probes/e2
    python scripts/train/probe_eval.py --compare outputs/probes/base outputs/probes/e1 outputs/probes/e2

Scoring: the official Sinhala prompt as plain text (no chat template: the Base model has none), ending with
"පිළිතුර: " (with the space); the prediction is the option digit with the highest next-token logit. One prompt at
a time and no padding, so a result never depends on the batch it was in (~30 ms per question on an L40S).
"""
import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))
from probe_common import ROOT, build_prompt, read_jsonl, stdout_utf8, write_json, write_jsonl  # noqa: E402
from common import sinhala_ratio  # noqa: E402  (scripts/cpt/common.py, put on sys.path by probe_common)

DIGITS = ["1", "2", "3", "4"]
EXPECTED_DIGIT_IDS = [16, 17, 18, 19]          # Qwen3.5 tokenizer


# ---------------------------------------------------------------- loading

def spread(items: list, limit: int | None) -> list:
    """`limit` evenly spaced items (the files are sorted by subject / split, so the first N would be one subject)."""
    if not limit or limit >= len(items):
        return items
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def load_probes(probe_dir: Path, limit: int | None = None) -> dict:
    """{'a_subset', 'a_full', 'b'} item lists. With `limit` (smoke / debug runs) a_full is the limited subset."""
    a_subset = list(read_jsonl(probe_dir / "probe_a_subset.jsonl"))
    a_full = list(read_jsonl(probe_dir / "probe_a_full.jsonl"))
    b = list(read_jsonl(probe_dir / "probe_b.jsonl"))
    if not {x["id"] for x in a_subset} <= {x["id"] for x in a_full}:
        raise SystemExit("probe_a_subset.jsonl has ids that are not in probe_a_full.jsonl: rebuild the probes")
    if limit:
        a_subset = spread(a_subset, limit)
        a_full = a_subset
        b = spread(b, limit)
    return {"a_subset": a_subset, "a_full": a_full, "b": b}


def load_sample_prompts(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["prompts"]


def digit_token_ids(tokenizer) -> list[int]:
    """Token ids of "1".."4", after checking that the prompt really ends where the bare digit comes next."""
    ids = [tokenizer.encode(d, add_special_tokens=False) for d in DIGITS]
    if any(len(i) != 1 for i in ids):
        raise SystemExit(f"each option digit must be one token, got {ids}")
    ids = [i[0] for i in ids]
    if ids != EXPECTED_DIGIT_IDS:
        print(f"WARNING: digit token ids are {ids}, expected {EXPECTED_DIGIT_IDS} (Qwen3.5 tokenizer)")
    probe = {"metadata": {"subject_original": "විද්‍යාව"}, "question": "?", "choices": ["a", "b", "c", "d"]}
    p = build_prompt(probe)
    if tokenizer.encode(p + "1", add_special_tokens=False) != tokenizer.encode(p, add_special_tokens=False) + [ids[0]]:
        raise SystemExit("prompt + '1' does not tokenize as prompt tokens + the digit token: check build_prompt")
    return ids


# ---------------------------------------------------------------- scoring

def _device(model):
    return next(model.parameters()).device


@torch.no_grad()
def score(model, tokenizer, items: list[dict], digit_ids: list[int]) -> list[dict]:
    """One row per item: prediction, correctness and the probability mass on the option digits (format check).
    The caller puts the model in eval mode."""
    dev = _device(model)
    rows = []
    for it in items:
        ids = tokenizer(build_prompt(it), add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        logits = model(input_ids=ids, logits_to_keep=1, use_cache=False).logits[0, -1].float()
        opts = digit_ids[: len(it["choices"])]
        pred = int(logits[opts].argmax()) + 1
        gold = int(it["answer"])
        rows.append({"id": it["id"], "subject": it["subject"], "split": it.get("split"), "gold": gold, "pred": pred,
                     "correct": pred == gold, "digit_mass": round(float(torch.softmax(logits, -1)[opts].sum()), 4)})
    return rows


def accuracy(rows: list[dict]) -> float:
    return sum(r["correct"] for r in rows) / len(rows) if rows else float("nan")


def std_error(p: float, n: int) -> float:
    """Standard error of an accuracy p measured on n questions (section 8.3: ~2 points at 500, ~1 at 1,851)."""
    return math.sqrt(max(p * (1 - p), 1e-9) / n) if n else float("nan")


def summarize_a(rows: list[dict], prefix: str) -> dict:
    preds = Counter(r["pred"] for r in rows)
    return {f"{prefix}_acc": accuracy(rows),
            f"{prefix}_n": len(rows),
            f"{prefix}_digit_mass": sum(r["digit_mass"] for r in rows) / max(len(rows), 1),
            # one digit taking most answers = answer-position habit, not knowledge
            f"{prefix}_pred_max_share": max(preds.values()) / len(rows) if rows else float("nan")}


def summarize_b(rows: list[dict]) -> dict:
    """Probe B per split, plus both splits on the subjects they share (the held-out split has no Art)."""
    tr = [r for r in rows if r["split"] == "train"]
    ho = [r for r in rows if r["split"] == "heldout"]
    shared = {r["subject"] for r in tr} & {r["subject"] for r in ho}
    tr_s = [r for r in tr if r["subject"] in shared]
    ho_s = [r for r in ho if r["subject"] in shared]
    return {"probe_b_train_acc": accuracy(tr), "probe_b_train_n": len(tr),
            "probe_b_heldout_acc": accuracy(ho), "probe_b_heldout_n": len(ho),
            "probe_b_train_acc_shared": accuracy(tr_s), "probe_b_heldout_acc_shared": accuracy(ho_s),
            "probe_b_gap_shared": accuracy(tr_s) - accuracy(ho_s),
            "probe_b_digit_mass": sum(r["digit_mass"] for r in rows) / max(len(rows), 1)}


def per_subject(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[(r["split"] + "/" if r["split"] else "") + r["subject"]].append(r)
    return {k: {"acc": round(accuracy(v), 4), "n": len(v)} for k, v in sorted(by.items())}


# ---------------------------------------------------------------- sample outputs

def distinct_ngrams(ids: list[int], n: int) -> float:
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 1.0


@torch.no_grad()
def generate_samples(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[dict]:
    """Greedy continuations of the fixed prompts: broken script, wrong language or loops show up at once."""
    from transformers import GenerationConfig
    gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                               eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    dev = _device(model)
    out = []
    for p in prompts:
        enc = tokenizer(p, add_special_tokens=False, return_tensors="pt").to(dev)
        new = model.generate(**enc, generation_config=gen_cfg)[0, enc.input_ids.shape[1]:].tolist()
        text = tokenizer.decode(new, skip_special_tokens=False)
        out.append({"prompt": p, "output": text, "n_tokens": len(new),
                    "distinct2": round(distinct_ngrams(new, 2), 3), "sinhala_ratio": round(sinhala_ratio(text), 3)})
    return out


# ---------------------------------------------------------------- one full probe pass

def run_probes(model, tokenizer, probes: dict, digit_ids: list[int], full: bool,
               prompts: list[str] | None = None, max_new_tokens: int = 100) -> tuple[dict, dict]:
    """Probe A (subset, or all usable questions if `full`), Probe B and the sample outputs.
    Returns (scalar metrics, details for the JSON record). Leaves the model's train/eval mode as it found it."""
    was_training = model.training
    model.eval()
    t0 = time.time()
    try:
        rows_a = score(model, tokenizer, probes["a_full"] if full else probes["a_subset"], digit_ids)
        subset_ids = {x["id"] for x in probes["a_subset"]}
        metrics = summarize_a([r for r in rows_a if r["id"] in subset_ids], "probe_a")   # same 504 at every step
        if full:
            metrics |= summarize_a(rows_a, "probe_a_full")
        rows_b = score(model, tokenizer, probes["b"], digit_ids)
        metrics |= summarize_b(rows_b)
        samples = generate_samples(model, tokenizer, prompts, max_new_tokens) if prompts else []
        if samples:
            metrics["samples_distinct2"] = sum(s["distinct2"] for s in samples) / len(samples)
    finally:
        model.train(was_training)
    metrics["probe_seconds"] = time.time() - t0
    details = {"metrics": metrics, "full_probe_a": full,
               "per_subject": {"probe_a": per_subject(rows_a), "probe_b": per_subject(rows_b)},
               "samples": samples, "rows_a": rows_a, "rows_b": rows_b}
    return metrics, details


def save_details(details: dict, out_dir: Path, name: str) -> None:
    """<out_dir>/<name>.json (metrics, per subject, samples) + <name>_probe_a.jsonl / _probe_b.jsonl (predictions)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"{name}.json", {k: v for k, v in details.items() if not k.startswith("rows_")})
    write_jsonl(out_dir / f"{name}_probe_a.jsonl", details["rows_a"])
    write_jsonl(out_dir / f"{name}_probe_b.jsonl", details["rows_b"])


# ---------------------------------------------------------------- command line

COMPARE_KEYS = ["probe_a_full_acc", "probe_a_acc", "probe_b_train_acc", "probe_b_heldout_acc",
                "probe_b_train_acc_shared", "probe_b_heldout_acc_shared", "probe_b_gap_shared",
                "probe_a_full_pred_max_share", "probe_a_full_digit_mass", "samples_distinct2"]


def compare(dirs: list[str]) -> None:
    """Side-by-side table of the summary.json of several probe_eval runs (first column = reference)."""
    runs = [(Path(d).name, json.loads((Path(d) / "summary.json").read_text(encoding="utf-8"))["metrics"]) for d in dirs]
    print(f"{'metric':<30}" + "".join(f"{n:>14}" for n, _ in runs))
    for k in COMPARE_KEYS:
        if any(k in m for _, m in runs):
            print(f"{k:<30}" + "".join(f"{m[k]:>14.4f}" if k in m else f"{'-':>14}" for _, m in runs))
    ref = runs[0][1]
    if "probe_a_full_acc" in ref:
        print(f"\nnoise (1 SE): Probe A full ±{100 * std_error(ref['probe_a_full_acc'], ref['probe_a_full_n']):.1f} pts, "
              f"Probe B train ±{100 * std_error(ref['probe_b_train_acc'], ref['probe_b_train_n']):.1f} pts; "
              "differences under ~2 SE mean nothing")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the CPT probes on a saved model, or compare probe runs")
    ap.add_argument("--model", help="local folder or Hugging Face repo id")
    ap.add_argument("--subfolder", default="", help="e.g. e1 for the epoch copies pushed by the training run")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--probe-dir", default="data/probes/easy")
    ap.add_argument("--sample-prompts", default="scripts/train/config/sample_prompts.json")
    ap.add_argument("--subset-only", action="store_true", help="Probe A subset (504) instead of all 1,826")
    ap.add_argument("--limit", type=int, default=None, help="score only N questions per probe (debugging)")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--out", help="output folder for summary.json and predictions")
    ap.add_argument("--compare", nargs="+", metavar="DIR", help="print a table of earlier --out folders and exit")
    args = ap.parse_args()
    stdout_utf8()
    if args.compare:
        return compare(args.compare)
    if not args.model or not args.out:
        ap.error("--model and --out are required (or use --compare)")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = {"subfolder": args.subfolder, "revision": args.revision}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **kw)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **kw)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    n_params = sum(p.numel() for p in model.parameters())
    tied = model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    print(f"{type(model).__name__}: {n_params / 1e9:.3f}B parameters, embeddings tied: {tied}")

    probes = load_probes(ROOT / args.probe_dir, args.limit)
    prompts = load_sample_prompts(ROOT / args.sample_prompts)
    metrics, details = run_probes(model, tokenizer, probes, digit_token_ids(tokenizer), not args.subset_only,
                                  prompts, args.max_new_tokens)
    details["model"] = {"name": args.model, "subfolder": args.subfolder, "revision": args.revision,
                        "class": type(model).__name__, "n_params": n_params, "tied_embeddings": tied}
    save_details(details, Path(args.out), "summary")
    for k in COMPARE_KEYS:
        if k in metrics:
            print(f"{k:<30}{metrics[k]:.4f}")
    for s in details["samples"]:
        print(f"\n> {s['prompt']!r}\n{s['output']}")
    print(f"\n-> {Path(args.out) / 'summary.json'}")


if __name__ == "__main__":
    main()
