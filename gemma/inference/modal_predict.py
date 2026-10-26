"""Shared-task test predictions from sinhala-mmlu/gemma-4-e4b-sinhala-mmlu (thinking mode), on Modal with vLLM.

    modal run gemma/inference/modal_predict.py --limit 24        # smoke test: one batch of 24, a few minutes
    modal run --detach gemma/inference/modal_predict.py          # all 2238 questions -> predictions.zip
    modal run gemma/inference/modal_predict.py --collect         # no GPU: rebuild predictions.zip from the volume
    modal run --detach gemma/inference/modal_predict.py --run 2  # predict again, into outputs/run2 (consistency)

Writes gemma/inference/outputs/predictions.jsonl, predictions.zip (holding predictions.jsonl at its root) and
predictions_detail.jsonl (source, token counts and the model's reply for every question). A --limit run writes
to outputs/limit<N>/ instead, so a smoke test never looks like a submission.

Inputs are data/SinhalaMMLU-SharedTask-Test/{easy,medium,hard}/*.json (642 + 1111 + 485 = 2238 questions). Each
output line echoes q_no, subject and category from the test file, the level from its folder, and the answer as
a string "1".."4".

The method is the model card's validated think-mode recipe (scripts/bakeoff_run.py on the repo, run with
--modes think --samples 1 --greedy --context --think-budget 7168 --max-model-len 16384) in the card's image
(vLLM 0.32.0): its English system prompt with the subject/grade line, one greedy reasoning trace per question,
"Answer: N" parsed from the reply after the thinking block, and, when the reply has none (or the budget ran out),
the thinking block closed and the best of the digits 1-4 read after "Answer: ". The GPU runs at most 24
sequences at a time (vLLM max_num_seqs); questions are sent in chunks of --chunk (default 240 = 10 batches of
24), and every finished chunk is saved to the volume, so a dropped or restarted run resumes where it stopped.
--chunk 24 gives strict batches of 24 at the cost of waiting on the longest trace in each batch.

Because finished chunks are reused, a plain rerun predicts nothing and returns the first run's answers: a
repeat run needs --run <label>, and compare_runs.py diffs two runs.
"""
import json
import os
import re
import time
import zipfile
from collections import Counter
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # in the container the file sits at /root/
DATA = Path(os.environ.get("DATA_ROOT", "/data"))  # DATA_ROOT and RESULTS_ROOT let the code run outside Modal
RESULTS = Path(os.environ.get("RESULTS_ROOT", "/cache/submission"))
TEST = ROOT / "data/SinhalaMMLU-SharedTask-Test"
HF_ID = "sinhala-mmlu/gemma-4-e4b-sinhala-mmlu"
REVISION = "v1.0.0"
LEVELS = {"easy": 642, "medium": 1111, "hard": 485}
BATCH = 24  # sequences on the GPU at once
THINK_END = "<channel|>"  # end of Gemma 4's thinking channel
ANSWER_RE = re.compile(r"(?:Answer|පිළිතුර)\s*[:：]?\s*\**\s*\(?([1-4])\)?")
SEED = 1234

image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.32.0-cu129")  # the runtime the Gemma card validated
    .entrypoint([])  # the image would otherwise start a vLLM server
    .run_commands('ln -sf "$(command -v python3)" /usr/local/bin/python',  # make sure a plain `python` exists
                  "python3 -m pip uninstall -y torchcodec || true")  # card: unused, fails a native import
    .env({"HF_HOME": "/cache/hf"})
    .add_local_dir(str(TEST / "easy"), "/data/easy")
    .add_local_dir(str(TEST / "medium"), "/data/medium")
    .add_local_dir(str(TEST / "hard"), "/data/hard")
)
cache = modal.Volume.from_name("sinhala-mmlu-hf-cache", create_if_missing=True)  # weights survive between runs
# The model is public; a local HF_TOKEN, if set, only lifts the anonymous download rate limit.
secrets = [modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})] if os.environ.get("HF_TOKEN") else []
app = modal.App("sinhala-mmlu-gemma-sft-predict")


# ------------------------------------------------------------------ data and prompts

def load_questions(data: Path = DATA) -> list:
    """Every test question in a fixed order (level, file name, file order), tagged with its level and file."""
    items = []
    for level, expected in LEVELS.items():
        paths = sorted((data / level).glob("*.json"))
        found = [q | {"level": level, "file": p.name} for p in paths
                 for q in json.loads(p.read_text(encoding="utf-8"))]
        if len(found) != expected:
            raise RuntimeError(f"{level}: expected {expected} questions, found {len(found)} in {len(paths)} files")
        items += found
    keys = Counter(key(q) for q in items)
    if len(keys) != len(items):
        raise RuntimeError(f"duplicate (level, subject, q_no): {[k for k, n in keys.items() if n > 1][:5]}")
    return items


def key(q) -> tuple:
    return q["level"], q["subject"], q["q_no"]


def spread(items: list, limit: int) -> list:
    """`limit` evenly spaced items: the files are grouped by subject, so the first N would be one subject."""
    return items if not limit or limit >= len(items) else items[:: len(items) // limit][:limit]


def options(q) -> str:
    return "\n".join(f"{i}. {c.strip()}" for i, c in enumerate(q["choices"], start=1))


def grade_of(q):
    grade = q.get("metadata", {}).get("grade")
    return "AL" if str(grade) in ("12-13", "12", "13") else grade  # as bakeoff_run.py


SYSTEM = ("You are an expert teacher of the Sri Lankan national school curriculum (grades 6-13), fluent in "
          "Sinhala. You will receive a multiple-choice question written in Sinhala with four numbered options. "
          "Exactly one option is correct. Read the Sinhala carefully, including negations such as "
          "'නොවන' / 'නොවේ' (NOT). You may reason in English. "
          "Finish with the final answer in the form 'Answer: <number>'.")


def chat_prompt(tokenizer, q) -> str:
    """The Gemma card's MCQ prompt (its scripts/bakeoff_run.py --context) with thinking switched on."""
    grade = grade_of(q)
    head = f"විෂය (Subject): {q['subject']}" + (f" | ශ්‍රේණිය (Grade): {grade}" if grade else "") + "\n\n"
    user = (f"{head}ප්‍රශ්නය (Question):\n{q['question']}\n\nවිකල්ප (Options):\n{options(q)}\n\n"
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


# ------------------------------------------------------------------ inference

def think_answers(llm, tokenizer, items: list, budget: int) -> list:
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
                        "closed": closed, "capped": o.finish_reason == "length", "tokens": len(o.token_ids),
                        "reply": reply.strip()[-2000:], "thought_tail": thought[-1000:]})
        if not found:  # close the thinking block where it stopped and read the answer digit
            forced.append((i, prompt + thought + ("" if thought.endswith("\n") else "\n") + THINK_END + "Answer: "))
    if forced:
        digits = digit_tokens(tokenizer)
        pick = SamplingParams(max_tokens=1, temperature=0.0, seed=SEED, allowed_token_ids=sorted(digits))
        for (i, _), out in zip(forced, llm.generate([token_ids(tokenizer, p) for _, p in forced], pick)):
            records[i]["pred"] = digits[out.outputs[0].token_ids[0]]
    return records


def run_folder(limit: int, chunk: int, budget: int, run: str = "") -> str:
    """One folder per configuration. `run` labels a repeat of the same configuration, so that a consistency
    check predicts again instead of being handed the chunks the first run saved."""
    return f"{REVISION}/budget{budget}_chunk{chunk}/limit{limit}" + (f"/run{run}" if run else "")


@app.function(image=image, gpu="L40S", volumes={"/cache": cache}, timeout=12 * 3600, secrets=secrets)
def predict(limit: int = 0, chunk: int = 10 * BATCH, budget: int = 7168, run: str = "",
            max_model_len: int = 16384) -> list:
    from vllm import LLM

    items = spread(load_questions(), limit)
    out_dir = RESULTS / run_folder(limit, chunk, budget, run)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    todo = []
    for n, part in enumerate(chunks):
        path = out_dir / f"chunk{n:04d}.json"
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if not saved or [list(key(r)) for r in saved] != [list(key(q)) for q in part]:
            todo.append(n)
    print(f"{len(items)} questions in {len(chunks)} chunks of {chunk}; {len(chunks) - len(todo)} already saved "
          f"in {out_dir}", flush=True)

    if todo:
        llm, last_error = None, None
        # Like bakeoff_run.py: no images or audio per prompt; older vLLM builds reject some modality keys.
        for limits in ({"image": 0, "audio": 0, "video": 0}, {"image": 0, "audio": 0}, {"image": 0}):
            try:
                llm = LLM(model=HF_ID, revision=REVISION, tokenizer_revision=REVISION, dtype="bfloat16",
                          max_model_len=max_model_len, max_num_seqs=BATCH, gpu_memory_utilization=0.90,
                          limit_mm_per_prompt=limits, seed=SEED)
                break
            except (ValueError, TypeError) as e:
                last_error = e
                print(f"[warn] limit_mm_per_prompt={limits} rejected: {e}", flush=True)
        if llm is None:
            raise last_error
        if not modal.is_local():
            cache.commit()  # keep the downloaded weights for the next run
        tokenizer = llm.get_tokenizer()

        started, done = time.time(), 0
        for n in todo:
            part = chunks[n]
            records = think_answers(llm, tokenizer, part, budget)
            rows = [{k: q[k] for k in ("q_no", "subject", "category", "level", "file")} | r
                    for q, r in zip(part, records)]
            (out_dir / f"chunk{n:04d}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            if not modal.is_local():
                cache.commit()  # a dropped client or a restart never loses a finished chunk
            done += len(part)
            rate = done / (time.time() - started)
            left = sum(len(chunks[m]) for m in todo) - done
            print(f"chunk {n + 1}/{len(chunks)}: parsed {sum(r['source'] == 'parsed' for r in records)}/{len(part)}"
                  f", capped {sum(r['capped'] for r in records)}, mean {sum(r['tokens'] for r in records) / len(part):.0f}"
                  f" tok | {done} done this run, {rate * 60:.1f} q/min, ~{left / rate / 60:.0f} min left", flush=True)

    return [row for n in range(len(chunks))
            for row in json.loads((out_dir / f"chunk{n:04d}.json").read_text(encoding="utf-8"))]


# ------------------------------------------------------------------ outputs

def proxy_scores(rows: list) -> list:
    """Accuracy against the proxy answer key (scripts/answer_sharedtask_test.py, ~85-90% correct), if present."""
    lines = []
    for level in LEVELS:
        path = TEST / "with_answers" / f"{level}.jsonl"
        if not path.exists():
            continue
        gold = {(level, q["subject"], q["q_no"]): q["answer"] for q in
                map(json.loads, path.read_text(encoding="utf-8").splitlines()) if q}
        graded = [r["pred"] == gold[key(r)] for r in rows if r["level"] == level and key(r) in gold]
        if graded:
            lines.append((level, sum(graded) / len(graded), len(graded)))
    return lines


def write_outputs(rows: list, limit: int, run: str = "") -> None:
    out = HERE / "outputs"
    for part in (f"limit{limit}" if limit else "", f"run{run}" if run else ""):
        out = out / part
    out.mkdir(parents=True, exist_ok=True)
    bad = [r for r in rows if r["pred"] not in (1, 2, 3, 4)]
    if bad:
        raise SystemExit(f"{len(bad)} rows without an answer 1-4, e.g. {bad[0]['level']}/{bad[0]['file']}")
    if not limit and len(rows) != sum(LEVELS.values()):
        raise SystemExit(f"expected {sum(LEVELS.values())} predictions, got {len(rows)}")

    jsonl = "".join(json.dumps({"q_no": r["q_no"], "subject": r["subject"], "category": r["category"],
                                "level": r["level"], "answer": str(r["pred"])}, ensure_ascii=False) + "\n"
                    for r in rows)
    (out / "predictions.jsonl").write_text(jsonl, encoding="utf-8")
    with zipfile.ZipFile(out / "predictions.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("predictions.jsonl", jsonl)
    (out / "predictions_detail.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    n = len(rows)
    print(f"\n{n} predictions: " + ", ".join(f"{lv} {sum(r['level'] == lv for r in rows)}" for lv in LEVELS))
    print(f"parsed {sum(r['source'] == 'parsed' for r in rows) / n:.1%}, closed {sum(r['closed'] for r in rows) / n:.1%}"
          f", capped {sum(r['capped'] for r in rows) / n:.1%}, mean {sum(r['tokens'] for r in rows) / n:.0f} tokens")
    print("answer spread: " + ", ".join(f"{d}: {c}" for d, c in sorted(Counter(r["pred"] for r in rows).items())))
    scores = proxy_scores(rows)
    if scores:
        print("proxy-key accuracy (noisy key, not the official score): "
              + ", ".join(f"{lv} {acc:.1%} (n={m})" for lv, acc, m in scores)
              + (f", macro {sum(a for _, a, _ in scores) / len(scores):.1%}" if len(scores) == len(LEVELS) else ""))
    print(f"wrote {out / 'predictions.zip'}")


@app.local_entrypoint()
def main(limit: int = 0, chunk: int = 10 * BATCH, budget: int = 7168, run: str = "", collect: bool = False):
    """limit=0 predicts all 2238 questions; --limit 24 is a one-batch smoke test. --collect runs no GPU and
    rebuilds the outputs from the chunks the volume holds (after a --detach run or a dropped client).
    --run 2 predicts the same questions again into their own folders, for a consistency check: without it a
    rerun finds every chunk already saved and returns the first run's answers untouched."""
    if collect:
        folder = f"submission/{run_folder(limit, chunk, budget, run)}"
        files = sorted(e.path for e in cache.listdir(folder) if e.path.endswith(".json"))
        rows = [row for path in files for row in json.loads(b"".join(cache.read_file(path)).decode("utf-8"))]
        expected = len(spread(load_questions(TEST), limit))
        if len(rows) != expected:
            raise SystemExit(f"{folder} holds {len(rows)} of {expected} predictions; rerun without --collect to "
                             "finish the missing chunks (saved chunks are skipped)")
    else:
        rows = predict.remote(limit, chunk, budget, run)
    write_outputs(rows, limit, run)
