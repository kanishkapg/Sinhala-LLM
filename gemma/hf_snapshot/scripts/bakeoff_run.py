"""Evaluate one base model on the locked evaluation sets with vLLM.

Two modes per model:
  direct  - no thinking; read the probability of options 1-4 right after "Answer: "
  think   - thinking on; parse "Answer: N" from the reply, otherwise (or if the thinking hit the
            token budget) close the thinking block and read the 1-4 probabilities ("forced" answer)

Usage (one model per process so GPU memory is fully released between models):
  python bakeoff_run.py --model gemma4_e4b --inputs inputs --out results --think-budget 4096
"""
import argparse
import json
import math
import random
import re
import time
import zlib
from pathlib import Path

MODELS = {
    "qwen35_4b": {
        "hf_id": "Qwen/Qwen3.5-4B",
        "think_kwargs": {"enable_thinking": True},
        "direct_kwargs": {"enable_thinking": False},
        "think_end": "</think>",
        "after_think": "\n\n",
        # Model-card settings for thinking mode, general tasks.
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
        "mm_limits": [{"image": 0, "video": 0}, {"image": 0}],
    },
    "gemma4_e4b": {
        "hf_id": "sinhala-mmlu/gemma-4-e4b-sinhala-mmlu",
        "think_kwargs": {"enable_thinking": True},
        "direct_kwargs": {"enable_thinking": False},
        "think_end": "<channel|>",
        "after_think": "",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "mm_limits": [{"image": 0, "audio": 0, "video": 0}, {"image": 0, "audio": 0}, {"image": 0}],
    },
    "gemma4_e2b": {
        "hf_id": "google/gemma-4-E2B-it",
        "think_kwargs": {"enable_thinking": True},
        "direct_kwargs": {"enable_thinking": False},
        "think_end": "<channel|>",
        "after_think": "",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "mm_limits": [{"image": 0, "audio": 0, "video": 0}, {"image": 0, "audio": 0}, {"image": 0}],
    },
    # Larger teachers (over the 8B limit: only for generating training data, never for test predictions).
    "gemma4_26b_a4b": {
        "hf_id": "google/gemma-4-26B-A4B-it",
        "tp": 2,
        "think_kwargs": {"enable_thinking": True},
        "direct_kwargs": {"enable_thinking": False},
        "think_end": "<channel|>",
        "after_think": "",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "mm_limits": [{"image": 0, "audio": 0, "video": 0}, {"image": 0, "audio": 0}, {"image": 0}],
    },
    "gemma4_31b": {
        "hf_id": "google/gemma-4-31B-it",
        "tp": 2,
        "think_kwargs": {"enable_thinking": True},
        "direct_kwargs": {"enable_thinking": False},
        "think_end": "<channel|>",
        "after_think": "",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "mm_limits": [{"image": 0, "audio": 0, "video": 0}, {"image": 0, "audio": 0}, {"image": 0}],
    },
    # Optional ensemble partners (not run by default).
    "qwen35_2b": {
        "hf_id": "Qwen/Qwen3.5-2B",
        "think_kwargs": {"enable_thinking": True},
        "direct_kwargs": {"enable_thinking": False},
        "think_end": "</think>",
        "after_think": "\n\n",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
        "mm_limits": [{"image": 0, "video": 0}, {"image": 0}],
    },
}

SYSTEM = (
    "You are an expert teacher of the Sri Lankan national school curriculum (grades 6-13), "
    "fluent in Sinhala. You will receive a multiple-choice question written in Sinhala with four "
    "numbered options. Exactly one option is correct. Read the Sinhala carefully, including "
    "negations such as 'නොවන' / 'නොවේ' (NOT). You may reason in English. "
    "Finish with the final answer in the form 'Answer: <number>'."
)

# Options that point at other options must keep their order when --shuffle permutes the options.
POSITIONAL_RE = re.compile(r"ඉහත|above|\b[1-4]\s*(?:හා|සහ|හෝ|,|and|or)\s*[1-4]\b", re.I)
ANSWER_RE = re.compile(r"(?:Answer|පිළිතුර)\s*[:：]?\s*\**\s*\(?([1-4])\)?")


def grade_of(row):
    grade = row.get("grade") or (row.get("meta") or {}).get("grade")
    return "AL" if str(grade) in ("12-13", "12", "13") else grade


def user_message(row, context=False, reference=False):
    """context=True adds the subject and grade (both given in the test file) above the question.
    reference=True (context distillation only, never at test time) shows the row's source passage first."""
    options = "\n".join(f"{i}. {c}" for i, c in enumerate(row["choices"], 1))
    head = ""
    if reference and row.get("reference"):
        head = ("පසුබිම් පාඨය (Reference text from the textbook):\n" + row["reference"].strip() + "\n\n"
                "(Use this material to answer, but reason as if you had already learned it yourself: "
                "do not mention or quote the reference text.)\n\n")
    if context:
        grade = grade_of(row)
        head += f"විෂය (Subject): {row['subject']}" + (f" | ශ්‍රේණිය (Grade): {grade}" if grade else "") + "\n\n"
    return head + (
        f"ප්‍රශ්නය (Question):\n{row['question']}\n\n"
        f"විකල්ප (Options):\n{options}\n\n"
        "Which option (1, 2, 3 or 4) is correct? End your reply with 'Answer: <number>'."
    )


def load_sets(inputs, names):
    sets = {}
    for name in names:
        path = Path(inputs) / f"{name}.jsonl"
        sets[name] = [json.loads(line) for line in open(path, encoding="utf-8")]
    return sets


def build_llm(cfg, args):
    from vllm import LLM

    last_error = None
    for limits in cfg["mm_limits"]:
        try:
            return LLM(
                model=cfg["hf_id"],
                dtype="bfloat16",
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_mem,
                limit_mm_per_prompt=limits,
                seed=args.seed,
                tensor_parallel_size=cfg.get("tp", 1),
            ), limits
        except (ValueError, TypeError) as e:  # unsupported modality key -> try a smaller dict
            last_error = e
            print(f"[warn] limit_mm_per_prompt={limits} rejected: {e}", flush=True)
    raise last_error


def digit_token_variants(tok):
    """Single-token encodings of '1'..'4' with and without a leading space."""
    variants = {}
    for d in "1234":
        ids = set()
        for text in (d, " " + d):
            enc = tok.encode(text, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        if not ids:
            raise RuntimeError(f"digit {d} has no single-token encoding")
        variants[int(d)] = sorted(ids)
    return variants


def option_probs(output, variants):
    """4-way distribution from the raw top-k logprobs of the first generated token."""
    top = output.outputs[0].logprobs[0]
    logp = {}
    for d, ids in variants.items():
        vals = [top[i].logprob for i in ids if i in top]
        logp[d] = max(vals) if vals else -30.0
    m = max(logp.values())
    z = sum(math.exp(v - m) for v in logp.values())
    return [math.exp(logp[d] - m) / z for d in (1, 2, 3, 4)]


def render(tok, row, kwargs, context=False, reference=False):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_message(row, context, reference)}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    # Avoid a double BOS: drop the template's BOS only if the tokenizer adds one itself
    # (vLLM tokenizes text prompts with add_special_tokens=True).
    if tok.bos_token and text.startswith(tok.bos_token) and tok("x")["input_ids"][:1] == [tok.bos_token_id]:
        text = text[len(tok.bos_token):]
    return text


def score_prompts(llm, prompts, variants, seed):
    from vllm import SamplingParams

    allowed = sorted({i for ids in variants.values() for i in ids})
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20, allowed_token_ids=allowed, seed=seed)
    outs = llm.generate(prompts, sp, use_tqdm=True)
    return [option_probs(o, variants) for o in outs]


def run_direct(llm, tok, cfg, rows, variants, args):
    prompts = [render(tok, r, cfg["direct_kwargs"], args.context) + "Answer: " for r in rows]
    t0 = time.time()
    probs = score_prompts(llm, prompts, variants, args.seed)
    secs = time.time() - t0
    preds = []
    for r, p in zip(rows, probs):
        pred = 1 + max(range(4), key=lambda i: p[i])
        preds.append({"id": r["id"], "subject": r["subject"], "gold": r.get("answer"), "pred": pred,
                      "probs": p, "source": "direct_logprob"})
    return preds, {"seconds": secs, "example_prompt": prompts[0]}


def permutations_for(row, n):
    """n option orders for one question: rotations of a per-question random order (then its reverse), so each
    option appears in each position equally often. perm[j] = original index shown at position j."""
    if POSITIONAL_RE.search(" ".join(row["choices"])):
        return [[0, 1, 2, 3]] * n
    base = list(range(4))
    random.Random(zlib.crc32(row["id"].encode())).shuffle(base)
    orders = [base[i:] + base[:i] for i in range(4)]
    orders += [list(reversed(o)) for o in orders]
    return [orders[k % 8] for k in range(n)]


def run_think(llm, tok, cfg, rows, variants, args):
    from vllm import SamplingParams

    def sampling(seed, n):
        return SamplingParams(max_tokens=args.think_budget, n=n, seed=seed, skip_special_tokens=False,
                              **cfg["sampling"])

    # One request per (question, sample) when shuffling, since each sample sees its own option order;
    # otherwise one request per question with n samples.
    jobs = []  # (row, k, perm, prompt)
    for r in rows:
        if args.shuffle:
            for k, perm in enumerate(permutations_for(r, args.samples)):
                shown = r | {"choices": [r["choices"][i] for i in perm]}
                jobs.append((r, k, perm, render(tok, shown, cfg["think_kwargs"], args.context, args.reference)))
        else:
            jobs.append((r, None, [0, 1, 2, 3], render(tok, r, cfg["think_kwargs"], args.context, args.reference)))
    prompts = [j[3] for j in jobs]
    if args.shuffle:
        sps = [sampling(args.seed * 1000 + k, 1) for _, k, _, _ in jobs]
    else:
        sps = sampling(args.seed, args.samples)
    t0 = time.time()
    outs = llm.generate(prompts, sps, use_tqdm=True)
    gen_secs = time.time() - t0

    end = cfg["think_end"]
    forced_prompts, records = [], []
    for (r, k0, perm, prompt), out in zip(jobs, outs):
        for k, sample in enumerate(out.outputs):
            text = sample.text
            closed = end in text
            thought = text.split(end, 1)[0] if closed else text
            reply = text.split(end, 1)[1] if closed else ""
            matches = ANSWER_RE.findall(reply)
            shown = int(matches[-1]) if matches else None
            forced_prompts.append(prompt + thought + ("" if thought.endswith("\n") else "\n") + end
                                  + cfg["after_think"] + "Answer: ")
            records.append({"row": r, "k": k if k0 is None else k0, "perm": perm,
                            "parsed": None if shown is None else perm[shown - 1] + 1, "closed": closed,
                            "tokens": len(sample.token_ids), "finish": sample.finish_reason,
                            "reply_tail": reply[-200:], "thought": thought if args.save_text else None})
    t1 = time.time()
    forced = score_prompts(llm, forced_prompts, variants, args.seed)
    score_secs = time.time() - t1

    by_row = {}
    for rec, p in zip(records, forced):
        # Map the shown-position probabilities back to the original option order.
        orig = [0.0] * 4
        for j, i in enumerate(rec["perm"]):
            orig[i] = p[j]
        rec["forced_probs"] = orig
        by_row.setdefault(rec["row"]["id"], []).append(rec)
    preds = []
    for r in rows:
        samples = by_row[r["id"]]
        votes = [0.0] * 4
        for s in samples:
            choice = s["parsed"] if s["parsed"] is not None else 1 + max(range(4), key=lambda i: s["forced_probs"][i])
            votes[choice - 1] += 1.0
        mean_p = [sum(s["forced_probs"][i] for s in samples) / len(samples) for i in range(4)]
        pred = 1 + max(range(4), key=lambda i: (votes[i], mean_p[i]))
        preds.append({
            "id": r["id"], "subject": r["subject"], "gold": r.get("answer"), "pred": pred, "probs": mean_p,
            "votes": votes,
            "samples": [{k: s[k] for k in ("k", "perm", "parsed", "closed", "tokens", "finish", "reply_tail")}
                        | {"forced_pred": 1 + max(range(4), key=lambda i: s["forced_probs"][i])}
                        | ({"thought": s["thought"]} if args.save_text else {})
                        for s in samples],
        })
    return preds, {"generation_seconds": gen_secs, "forced_scoring_seconds": score_secs,
                   "example_prompt": prompts[0], "example_output": outs[0].outputs[0].text[:3000]}


def summarise(preds, think):
    n = len(preds)
    graded = all(p["gold"] is not None for p in preds)  # the test set has no answer key
    out = {"n": n, "accuracy": sum(p["pred"] == p["gold"] for p in preds) / n if graded else None,
           "pred_positions": {d: sum(p["pred"] == d for p in preds) for d in (1, 2, 3, 4)}}
    if graded:
        subj = {}
        for p in preds:
            s = subj.setdefault(p["subject"], [0, 0])
            s[0] += p["pred"] == p["gold"]
            s[1] += 1
        out["by_subject"] = {k: {"correct": c, "n": t, "accuracy": c / t} for k, (c, t) in sorted(subj.items())}
    if think:
        samples = [s for p in preds for s in p["samples"]]
        out["parse_rate"] = sum(s["parsed"] is not None for s in samples) / len(samples)
        out["closed_thinking_rate"] = sum(s["closed"] for s in samples) / len(samples)
        out["mean_output_tokens"] = sum(s["tokens"] for s in samples) / len(samples)
        out["parsed_vs_forced_agreement"] = (
            sum(s["parsed"] == s["forced_pred"] for s in samples if s["parsed"] is not None)
            / max(1, sum(s["parsed"] is not None for s in samples)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--inputs", default="inputs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--modes", default="direct,think")
    ap.add_argument("--think-budget", type=int, default=4096)
    ap.add_argument("--samples", type=int, default=1, help="thinking samples per question (majority vote)")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=0, help="first N questions per set (smoke test)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--hf-id", default=None, help="override the model path, e.g. a merged fine-tuned checkpoint")
    ap.add_argument("--sets", default="v_easy600,v_hard_gmmlu1000")
    ap.add_argument("--tag", default=None, help="results folder name (default: the model key)")
    ap.add_argument("--greedy", action="store_true", help="one deterministic decoding policy (temperature 0)")
    ap.add_argument("--context", action="store_true", help="show subject and grade above each question")
    ap.add_argument("--save-text", action="store_true", help="keep each sample's full thinking text")
    ap.add_argument("--reference", action="store_true",
                    help="show each row's 'reference' passage (context distillation; never for test predictions)")
    ap.add_argument("--shuffle", action="store_true",
                    help="each thinking sample sees the options in a different order (mapped back before voting)")
    args = ap.parse_args()

    cfg = dict(MODELS[args.model])
    if args.greedy:
        cfg["sampling"] = {"temperature": 0.0}
    if args.hf_id:
        cfg["hf_id"] = args.hf_id
    sets = load_sets(args.inputs, args.sets.split(","))
    if args.limit:
        sets = {k: v[: args.limit] for k, v in sets.items()}
    out_dir = Path(args.out) / (args.tag or args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    import vllm

    llm, limits = build_llm(cfg, args)
    tok = llm.get_tokenizer()
    variants = digit_token_variants(tok)
    report = {"model": args.model, "hf_id": cfg["hf_id"], "vllm": vllm.__version__,
              "mm_limits": limits, "digit_token_ids": variants, "args": vars(args), "results": {}}

    for mode in args.modes.split(","):
        for set_name, rows in sets.items():
            fn = run_think if mode == "think" else run_direct
            print(f"=== {args.model} / {mode} / {set_name} ({len(rows)} questions)", flush=True)
            preds, info = fn(llm, tok, cfg, rows, variants, args)
            with open(out_dir / f"{mode}_{set_name}.jsonl", "w", encoding="utf-8") as f:
                for p in preds:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            summary = summarise(preds, mode == "think") | {k: v for k, v in info.items() if "example" not in k}
            report["results"][f"{mode}/{set_name}"] = summary
            (out_dir / f"example_{mode}_{set_name}.txt").write_text(
                "\n\n----- OUTPUT -----\n".join([info.get("example_prompt", ""), info.get("example_output", "")]),
                encoding="utf-8")
            acc = summary["accuracy"]
            print(f"    accuracy={'n/a' if acc is None else f'{acc:.4f}'}  "
                  + (f"parse={summary['parse_rate']:.3f} closed={summary['closed_thinking_rate']:.3f} "
                     f"tokens={summary['mean_output_tokens']:.0f}" if mode == "think" else ""), flush=True)
            (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
