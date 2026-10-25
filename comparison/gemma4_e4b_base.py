"""Thinking-mode accuracy of Google's untouched Gemma 4 E4B-it on the four comparison sets, on Modal with vLLM.

    modal run comparison/gemma4_e4b_base.py --limit 20             # smoke test, a few minutes
    modal run comparison/gemma4_e4b_base.py --limit 200            # the same 200 questions per set as results_table.txt
    modal run --detach comparison/gemma4_e4b_base.py               # full sets; keeps running if the client drops
    modal run comparison/gemma4_e4b_base.py --collect              # no GPU: rebuild the table from the volume

Writes comparison/results_<tag>_think[_limitN].txt and comparison/predictions/<tag>_think_<set>.jsonl.

The method is the think mode of the Gemma SFT card (scripts/bakeoff_run.py in sinhala-mmlu/gemma-4-e4b-sinhala-mmlu),
run the greedy way its published numbers were measured: its English system prompt with the subject/grade line, a
7,168-token thinking budget, "Answer: N" parsed from the reply after the thinking block, and, when the reply has
none (or the budget ran out), the thinking block closed and the best of the digits 1-4 read after "Answer: ".
It runs in the card's validated image (vLLM 0.32.0, Transformers 5.19.0), so it is like-for-like with the card.
`--hf-id sinhala-mmlu/gemma-4-e4b-sinhala-mmlu --revision v1.0.0 --tag gemma-sft` scores the fine-tuned model
the same way.

Scored sets (uploaded from this repo):
    mmlu_easy      data/SinhalaMMLU/TEST/easy/*.json                        1851 items (1850 scorable)
    shared_easy    data/SinhalaMMLU-SharedTask-Test/with_answers/easy.jsonl  642
    shared_medium  .../medium.jsonl                                         1111
    shared_hard    .../hard.jsonl                                            485
"""
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_ROOT", "/data"))  # DATA_ROOT and RESULTS_ROOT let the code run outside Modal
RESULTS = Path(os.environ.get("RESULTS_ROOT", "/cache/think_results"))
BASE_ID = "google/gemma-4-E4B-it"
THINK_END = "<channel|>"  # end of Gemma 4's thinking channel
ANSWER_RE = re.compile(r"(?:Answer|පිළිතුර)\s*[:：]?\s*\**\s*\(?([1-4])\)?")
SEED = 1234
SETS = ["mmlu_easy", "shared_easy", "shared_medium", "shared_hard"]

image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.32.0-cu129")  # the runtime the Gemma card validated
    .entrypoint([])  # the image would otherwise start a vLLM server
    .run_commands('ln -sf "$(command -v python3)" /usr/local/bin/python',  # make sure a plain `python` exists
                  "python3 -m pip uninstall -y torchcodec || true")  # card: unused, fails a native import
    .env({"HF_HOME": "/cache/hf"})
    .add_local_dir(str(ROOT / "data/SinhalaMMLU/TEST/easy"), "/data/mmlu_easy")
    .add_local_file(str(ROOT / "data/SinhalaMMLU-SharedTask-Test/with_answers/easy.jsonl"), "/data/shared/easy.jsonl")
    .add_local_file(str(ROOT / "data/SinhalaMMLU-SharedTask-Test/with_answers/medium.jsonl"), "/data/shared/medium.jsonl")
    .add_local_file(str(ROOT / "data/SinhalaMMLU-SharedTask-Test/with_answers/hard.jsonl"), "/data/shared/hard.jsonl")
)
cache = modal.Volume.from_name("sinhala-mmlu-hf-cache", create_if_missing=True)  # weights survive between runs
app = modal.App("sinhala-mmlu-gemma-base-think")


# ------------------------------------------------------------------ data and prompts (as in modal_comparison.py)

def load_sets() -> dict:
    sets = {"mmlu_easy": [q for p in sorted((DATA / "mmlu_easy").glob("*.json"))
                          for q in json.loads(p.read_text(encoding="utf-8"))]}
    for level in ("easy", "medium", "hard"):
        path = DATA / "shared" / f"{level}.jsonl"
        sets[f"shared_{level}"] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return sets


def spread(items: list, limit: int) -> list:
    """`limit` evenly spaced items: the files are grouped by subject, so the first N would be one subject."""
    return items if not limit or limit >= len(items) else items[:: len(items) // limit][:limit]


def options(q) -> str:
    return "\n".join(f"{i}. {c.strip()}" for i, c in enumerate(q["choices"], start=1))


SYSTEM = ("You are an expert teacher of the Sri Lankan national school curriculum (grades 6-13), fluent in "
          "Sinhala. You will receive a multiple-choice question written in Sinhala with four numbered options. "
          "Exactly one option is correct. Read the Sinhala carefully, including negations such as "
          "'නොවන' / 'නොවේ' (NOT). You may reason in English. "
          "Finish with the final answer in the form 'Answer: <number>'.")


def chat_prompt(tokenizer, q) -> str:
    """The Gemma card's MCQ prompt (its scripts/bakeoff_run.py) with the subject/grade line, thinking switched on."""
    user = (f"විෂය (Subject): {q['subject']} | ශ්‍රේණිය (Grade): {q['metadata'].get('grade')}\n\n"
            f"ප්‍රශ්නය (Question):\n{q['question']}\n\nවිකල්ප (Options):\n{options(q)}\n\n"
            "Which option (1, 2, 3 or 4) is correct? End your reply with 'Answer: <number>'.")
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)


def token_ids(tokenizer, text: str) -> dict:
    # The chat template already starts with BOS; vLLM would add a second one to a text prompt.
    return {"prompt_token_ids": tokenizer(text, add_special_tokens=False)["input_ids"]}


def digit_tokens(tokenizer) -> dict:
    """Token id -> digit for '1'..'4', with and without a leading space (single-token encodings only)."""
    found = {}
    for d in "1234":
        for text in (d, " " + d):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                found[ids[0]] = int(d)
    if set(found.values()) != {1, 2, 3, 4}:
        raise RuntimeError(f"no single-token encoding for every digit: {found}")
    return found


# ------------------------------------------------------------------ scoring

def think_scores(llm, tokenizer, items: list, budget: int) -> list:
    """One greedy reasoning trace per question; the parsed "Answer: N", else the forced digit after the trace."""
    from vllm import SamplingParams

    prompts = [chat_prompt(tokenizer, q) for q in items]
    outs = llm.generate([token_ids(tokenizer, p) for p in prompts],
                        SamplingParams(max_tokens=budget, temperature=0.0, seed=SEED, skip_special_tokens=False))
    records, forced = [], []
    for i, (prompt, out) in enumerate(zip(prompts, outs)):
        o = out.outputs[0]
        closed = THINK_END in o.text
        thought, reply = o.text.split(THINK_END, 1) if closed else (o.text, "")
        found = ANSWER_RE.findall(reply)
        records.append({"pred": int(found[-1]) if found else None, "source": "parsed" if found else "forced",
                        "closed": closed, "capped": o.finish_reason == "length", "tokens": len(o.token_ids)})
        if not found:  # close the thinking block where it stopped and read the answer digit
            forced.append((i, prompt + thought + ("" if thought.endswith("\n") else "\n") + THINK_END + "Answer: "))
    if forced:
        digits = digit_tokens(tokenizer)
        pick = SamplingParams(max_tokens=1, temperature=0.0, seed=SEED, allowed_token_ids=sorted(digits))
        for (i, _), out in zip(forced, llm.generate([token_ids(tokenizer, p) for _, p in forced], pick)):
            records[i]["pred"] = digits[out.outputs[0].token_ids[0]]
    if outs:
        print("example reply (first question):\n" + outs[0].outputs[0].text[-1500:], flush=True)
    return records


def summarise(items: list, records: list, seconds: float) -> dict:
    graded = [(q, r) for q, r in zip(items, records) if 1 <= q["answer"] <= len(q["choices"])]
    by_subject = defaultdict(lambda: [0, 0])
    for q, r in graded:
        by_subject[q["subject"]][0] += r["pred"] == q["answer"]
        by_subject[q["subject"]][1] += 1
    n = len(records)
    return {"accuracy": sum(r["pred"] == q["answer"] for q, r in graded) / len(graded), "n": len(graded),
            "asked": n, "parsed": sum(r["source"] == "parsed" for r in records) / n,
            "closed": sum(r["closed"] for r in records) / n, "capped": sum(r["capped"] for r in records) / n,
            "mean_tokens": sum(r["tokens"] for r in records) / n, "seconds": round(seconds, 1),
            "by_subject": {s: {"correct": c, "n": t} for s, (c, t) in sorted(by_subject.items())},
            "preds": [{"q_no": q.get("q_no"), "subject": q.get("subject"), "gold": q["answer"]} | r
                      for q, r in zip(items, records)]}


# ------------------------------------------------------------------ remote evaluation

@app.function(image=image, gpu="L40S", volumes={"/cache": cache}, timeout=6 * 3600,
              secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])])
def evaluate(hf_id: str, revision: str | None, tag: str, limit: int = 0, budget: int = 7168,
             max_model_len: int = 16384) -> dict:
    os.environ["HF_HOME"] = f"/cache/hf/{tag}"  # one cache subtree per model, as in modal_comparison.py

    from vllm import LLM

    llm, last_error = None, None
    # Like bakeoff_run.py: no images or audio per prompt; older vLLM builds reject some modality keys.
    for limits in ({"image": 0, "audio": 0, "video": 0}, {"image": 0, "audio": 0}, {"image": 0}):
        try:
            llm = LLM(model=hf_id, revision=revision, tokenizer_revision=revision, dtype="bfloat16",
                      max_model_len=max_model_len, gpu_memory_utilization=0.90, limit_mm_per_prompt=limits,
                      seed=SEED)
            break
        except (ValueError, TypeError) as e:
            last_error = e
            print(f"[warn] limit_mm_per_prompt={limits} rejected: {e}", flush=True)
    if llm is None:
        raise last_error
    if not modal.is_local():
        cache.commit()  # keep the downloaded weights for the next run
    tokenizer = llm.get_tokenizer()

    out_dir = RESULTS / tag / f"limit{limit}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for name, items in load_sets().items():
        subset, t0 = spread(items, limit), time.time()
        print(f"=== {tag} think {name}: {len(subset)} questions", flush=True)
        runs[name] = summarise(subset, think_scores(llm, tokenizer, subset, budget), time.time() - t0)
        r = runs[name]
        print(f"{tag} think {name}: {r['accuracy']:.2%} (n={r['n']}) parsed {r['parsed']:.1%} closed "
              f"{r['closed']:.1%} capped {r['capped']:.1%} mean {r['mean_tokens']:.0f} tok, {r['seconds']:.0f}s",
              flush=True)
        # Each finished set goes to the volume, so a dropped client never loses GPU work (see --collect).
        (out_dir / f"{name}.json").write_text(json.dumps(
            {"tag": tag, "hf_id": hf_id, "revision": revision, "limit": limit, "budget": budget, "set": name,
             "run": r}, ensure_ascii=False), encoding="utf-8")
        if not modal.is_local():
            cache.commit()
    return {"tag": tag, "hf_id": hf_id, "revision": revision, "limit": limit, "budget": budget, "sets": runs}


# ------------------------------------------------------------------ table

def render_table(result: dict) -> str:
    sets, limit = result["sets"], result["limit"]
    out = [f"Gemma 4 E4B thinking-mode accuracy - {result['hf_id']} @ {result['revision'] or 'main'} "
           f"(tag {result['tag']}) - {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
           "accuracy in %, " + ("full test sets" if not limit else
                                f"{limit} evenly spread questions per set (the same questions as "
                                f"modal_comparison.py --limit {limit})"), "",
           f"{'set':<16}{'accuracy':>9}{'scored':>11}{'parsed':>9}{'closed':>9}{'capped':>9}{'mean tok':>10}"
           f"{'minutes':>9}"]
    for name in SETS:
        r = sets.get(name)
        if r is None:
            out.append(f"{name:<16}{'-':>9}")
            continue
        out.append(f"{name:<16}{100 * r['accuracy']:>9.2f}{str(r['n']) + '/' + str(r['asked']):>11}"
                   f"{100 * r['parsed']:>8.1f}%{100 * r['closed']:>8.1f}%{100 * r['capped']:>8.1f}%"
                   f"{r['mean_tokens']:>10.0f}{r['seconds'] / 60:>9.1f}")
    shared = [sets[s]["accuracy"] for s in SETS[1:] if s in sets]
    if shared:
        out.append(f"{'shared_macro':<16}{100 * sum(shared) / len(shared):>9.2f}")
    out += ["", "accuracy by subject (correct/n):"]
    for name in SETS:
        if name in sets:
            out.append(f"  {name}: " + ", ".join(f"{s} {100 * v['correct'] / v['n']:.0f}% ({v['correct']}/{v['n']})"
                                                 for s, v in sets[name]["by_subject"].items()))
    out += ["", "notes:",
            f"  think = one greedy reasoning trace per question (temperature 0, {result['budget']}-token budget),",
            "  then 'Answer: N' parsed after the thinking block. parsed = share answered that way; the rest were",
            "  forced: thinking block closed where it stopped, best of digits 1-4 read after 'Answer: '.",
            "  closed = thinking block finished; capped = hit the token budget",
            "  mmlu_easy scores 1850 of 1851: one Geography item has a broken gold label (answer 24 of 4 options)",
            "  the three shared_* sets use the proxy answer key from scripts/answer_sharedtask_test.py (~85-90%",
            "  correct), so differences of a few points on those rows are inside the label noise"]
    return "\n".join(out) + "\n"


def write_outputs(result: dict, out: str) -> None:
    text = render_table(result)
    print("\n" + text)
    (ROOT / out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / out).write_text(text, encoding="utf-8")
    preds_dir = ROOT / "comparison" / "predictions"
    preds_dir.mkdir(exist_ok=True)
    for name, run in result["sets"].items():
        (preds_dir / f"{result['tag']}_think_{name}.jsonl").write_text(
            "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in run["preds"]), encoding="utf-8")
    print(f"wrote {out} and {len(result['sets'])} prediction files")


@app.local_entrypoint()
def main(limit: int = 0, hf_id: str = BASE_ID, revision: str = "", tag: str = "gemma-base", budget: int = 7168,
         gpu: str = "L40S", collect: bool = False, out: str = ""):
    """limit=0 scores the full sets; --hf-id/--revision/--tag score another Gemma 4 checkpoint the same way;
    --gpu A100-80GB if no L40S is free; --collect runs no GPU and rebuilds the table from the sets the volume
    already holds for this tag and limit (dropped client, detached run)."""
    out = out or f"comparison/results_{tag}_think{f'_limit{limit}' if limit else ''}.txt"
    if collect:
        folder = f"think_results/{tag}/limit{limit}"
        saved = [json.loads(b"".join(cache.read_file(e.path)).decode()) for e in cache.listdir(folder)]
        if not saved:
            raise SystemExit(f"nothing in the volume under {folder}")
        first = saved[0]
        result = {k: first[k] for k in ("tag", "hf_id", "revision", "limit", "budget")}
        result["sets"] = {s["set"]: s["run"] for s in saved}
        print("found", ", ".join(sorted(result["sets"])))
    else:
        result = evaluate.with_options(gpu=gpu).remote(hf_id, revision or None, tag, limit, budget)
    write_outputs(result, out)
