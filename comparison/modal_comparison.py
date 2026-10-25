"""Accuracy of the two SinhalaMMLU checkpoints on Modal: load from the Hub, score, write a comparison table.

    pip install -U modal
    modal setup
    modal secret create huggingface-secret HF_TOKEN=hf_...     # read access to the private CPT repo
    modal run comparison/modal_comparison.py --limit 200       # smoke test first
    modal run comparison/modal_comparison.py                   # full sets -> comparison/results_table.txt

Nothing is trained: each checkpoint is loaded, scored on the four test sets, and released.

Scored sets (uploaded from this repo, no Hub dataset needed):
    mmlu_easy      data/SinhalaMMLU/TEST/easy/*.json                        1851 items (1850 scorable)
    shared_easy    data/SinhalaMMLU-SharedTask-Test/with_answers/easy.jsonl  642
    shared_medium  .../medium.jsonl                                         1111
    shared_hard    .../hard.jsonl                                            485

Modes: "direct" is one forward pass per question, the answer being the option digit with the highest logit
(the official SinhalaMMLU scoring, and the only sensible mode for the CPT Base model). "think" is the Gemma
checkpoint's intended inference: English reasoning, then "Answer: N" parsed out, falling back to the direct
digit when the reply has none. Its model card measures ~71% think vs ~51% direct, so the Gemma row to quote
is the think one.
"""
import json
import os
import re
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_ROOT", "/data"))  # DATA_ROOT lets the same code run outside Modal
THINK_END = "<channel|>"  # end of Gemma 4's thinking channel, from the card's bakeoff_run.py
ANSWER_RE = re.compile(r"(?:Answer|පිළිතුර)\s*[:：]?\s*\**\s*\(?([1-4])\)?")

# L40S: 48 GB, so the 7.94B Gemma checkpoint in bf16 (~16 GB) leaves room for a large reasoning batch. Decoding
# is memory-bandwidth-bound and the L40S has less of it than an A100 (~864 GB/s vs ~1.56 TB/s), so the batch is
# what buys back the wall-clock time: --batch 24 generates 24 reasoning traces per forward pass instead of 8.
GPU = "L40S"
MODELS = {
    "qwen-cpt": dict(hf_id="sinhala-mmlu/qwen3.5-4b-si-cpt-easy", subfolder="e3",  # e1/e2/e3 = epoch copies
                     style="plain", gpu=GPU),
    "gemma-sft": dict(hf_id="sinhala-mmlu/gemma-4-e4b-sinhala-mmlu", revision="v1.0.0",
                      style="chat", gpu=GPU),
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers==5.19.0", "accelerate", "sentencepiece", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf"})
    .add_local_dir(str(ROOT / "data/SinhalaMMLU/TEST/easy"), "/data/mmlu_easy")
    .add_local_file(str(ROOT / "data/SinhalaMMLU-SharedTask-Test/with_answers/easy.jsonl"), "/data/shared/easy.jsonl")
    .add_local_file(str(ROOT / "data/SinhalaMMLU-SharedTask-Test/with_answers/medium.jsonl"), "/data/shared/medium.jsonl")
    .add_local_file(str(ROOT / "data/SinhalaMMLU-SharedTask-Test/with_answers/hard.jsonl"), "/data/shared/hard.jsonl")
)
cache = modal.Volume.from_name("sinhala-mmlu-hf-cache", create_if_missing=True)  # weights survive between runs
app = modal.App("sinhala-mmlu-comparison")


# ------------------------------------------------------------------ data and prompts

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


def plain_prompt(q) -> str:
    """The official SinhalaMMLU prompt (scripts/probes/probe_common.py), ending in "පිළිතුර: " with the space,
    so the next token is a bare digit. The CPT model is a Base model and has no chat template."""
    intro = "මෙය {} විෂයයට අදාළ බහුවරණ ප්‍රශ්නයකි.\n".format(q["metadata"].get("subject_original") or q["subject"])
    instruction = "පහත ප්‍රශ්නයට 1, 2, 3, 4 යන පිළිතුරුවලින් නිවැරදි හෝ ඉතාමත් ගැළපෙන පිළිතුර තෝරන්න.\n\n"
    return f"{intro}{instruction}ප්‍රශ්නය: {q['question']}\n{options(q)}\nපිළිතුර: "


SYSTEM = ("You are an expert teacher of the Sri Lankan national school curriculum (grades 6-13), fluent in "
          "Sinhala. You will receive a multiple-choice question written in Sinhala with four numbered options. "
          "Exactly one option is correct. Read the Sinhala carefully, including negations such as "
          "'නොවන' / 'නොවේ' (NOT). You may reason in English. "
          "Finish with the final answer in the form 'Answer: <number>'.")


def chat_prompt(tokenizer, q, think=False) -> str:
    """The Gemma card's own MCQ prompt (its scripts/bakeoff_run.py), with the subject/grade context it used."""
    user = (f"විෂය (Subject): {q['subject']} | ශ්‍රේණිය (Grade): {q['metadata'].get('grade')}\n\n"
            f"ප්‍රශ්නය (Question):\n{q['question']}\n\nවිකල්ප (Options):\n{options(q)}\n\n"
            "Which option (1, 2, 3 or 4) is correct? End your reply with 'Answer: <number>'.")
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=think)
    return text if think else text + "Answer: "  # prime the answer, then read the digit


# ------------------------------------------------------------------ remote evaluation

@app.function(image=image, gpu=GPU, volumes={"/cache": cache}, timeout=8 * 3600,
              secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])])
def evaluate(key: str, modes: str, limit: int = 0, batch: int = 24, max_new: int = 2048) -> dict:
    import inspect

    os.environ["HF_HOME"] = f"/cache/hf/{key}"  # one cache subtree per model: the two jobs run side by side

    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    cfg = MODELS[key]
    kw = {k: cfg[k] for k in ("revision", "subfolder") if k in cfg}
    loader = AutoModelForImageTextToText if cfg["style"] == "chat" else AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"], **kw)
    model = loader.from_pretrained(cfg["hf_id"], dtype=torch.bfloat16, device_map="cuda", **kw).eval()
    digit_ids = [tokenizer.encode(d, add_special_tokens=False)[-1] for d in "1234"]
    keep = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(model.forward).parameters else {}
    print(f"{key} loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params, digit ids {digit_ids}",
          flush=True)
    cache.commit()  # keep the downloaded weights for the next run

    @torch.inference_mode()
    def digit_pred(prompt: str, n_options: int = 4) -> int:
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        logits = model(**ids, **keep).logits[0, -1]
        return int(logits[digit_ids[:n_options]].argmax()) + 1

    @torch.inference_mode()
    def think_pred(chunk: list) -> list:
        ids = tokenizer([chat_prompt(tokenizer, q, think=True) for q in chunk], add_special_tokens=False,
                        padding=True, return_tensors="pt").to(model.device)
        t0 = time.time()
        gen = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
        dt, new = time.time() - t0, gen.shape[-1] - ids["input_ids"].shape[-1]
        print(f"    {len(chunk)} prompts of {ids['input_ids'].shape[-1]} tok -> {new} new tok in {dt:.0f}s "
              f"({len(chunk) * new / dt:.0f} tok/s)", flush=True)
        preds = []
        for q, text in zip(chunk, tokenizer.batch_decode(gen[:, ids["input_ids"].shape[-1]:],
                                                         skip_special_tokens=False)):
            reply = text.split(THINK_END, 1)[-1]  # the answer after the thinking block, else the whole reply
            m = ANSWER_RE.findall(reply) or ANSWER_RE.findall(text)
            preds.append(int(m[-1]) if m else digit_pred(chat_prompt(tokenizer, q), len(q["choices"])))
        return preds

    build = plain_prompt if cfg["style"] == "plain" else (lambda q: chat_prompt(tokenizer, q))
    sets, runs = load_sets(), {}
    for mode in [m.strip() for m in modes.split(",") if m.strip()]:
        if mode == "think":
            tokenizer.padding_side = "left"
        runs[mode] = {}
        for name, items in sets.items():
            subset, t0 = spread(items, limit), time.time()
            if mode == "think":
                # Longest prompts first, so each batch holds prompts of similar length and little of the
                # forward pass is spent on padding; the predictions go back into the original order after.
                order = sorted(range(len(subset)), key=lambda i: -len(chat_prompt(tokenizer, subset[i], think=True)))
                done = {}
                for start in range(0, len(order), batch):
                    idx = order[start:start + batch]
                    done.update(zip(idx, think_pred([subset[i] for i in idx])))
                    print(f"  {key}/{mode} {name}: {len(done)}/{len(subset)} ({time.time() - t0:.0f}s)", flush=True)
                preds = [done[i] for i in range(len(subset))]
            else:
                preds = []
                while len(preds) < len(subset):
                    preds += [digit_pred(build(q), len(q["choices"])) for q in subset[len(preds):len(preds) + 100]]
                    print(f"  {key}/{mode} {name}: {len(preds)}/{len(subset)} ({time.time() - t0:.0f}s)", flush=True)
            hits = [p == q["answer"] for q, p in zip(subset, preds) if 1 <= q["answer"] <= len(q["choices"])]
            runs[mode][name] = {"accuracy": sum(hits) / len(hits), "n": len(hits), "asked": len(subset),
                                "seconds": round(time.time() - t0, 1),
                                "preds": [{"q_no": q.get("q_no"), "subject": q.get("subject"),
                                           "gold": q["answer"], "pred": p} for q, p in zip(subset, preds)]}
            print(f"{key}/{mode} {name}: {runs[mode][name]['accuracy']:.2%} (n={len(hits)})", flush=True)
        # Keep each finished mode in the volume, so a dropped client never loses hours of GPU work:
        # `modal run comparison/modal_comparison.py::collect` rebuilds the table from these.
        Path("/cache/results").mkdir(parents=True, exist_ok=True)
        Path(f"/cache/results/{key}_{mode}_limit{limit}.json").write_text(
            json.dumps({"key": key, "mode": mode, "limit": limit, "sets": runs[mode]}, ensure_ascii=False))
        cache.commit()
    return {"key": key, "runs": runs}


# ------------------------------------------------------------------ table

def render_table(results: list, limit: int) -> str:
    """Fixed-width accuracy table: one column per model/mode, one row per test set."""
    cols = [(r["key"], mode) for r in results for mode in r["runs"]]
    acc = {(r["key"], mode): r["runs"][mode] for r in results for mode in r["runs"]}
    rows = ["mmlu_easy", "shared_easy", "shared_medium", "shared_hard"]
    width = max(18, *(len(f"{k}/{m}") + 2 for k, m in cols))
    out = [f"SinhalaMMLU baseline comparison - {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
           f"accuracy in %, {'full test sets' if not limit else f'{limit} evenly spread questions per set'}", "",
           "set".ljust(16) + "".join(f"{k + '/' + m:>{width}}" for k, m in cols)]
    for row in rows + ["shared_macro"]:
        cells = []
        for col in cols:
            if row == "shared_macro":
                vals = [acc[col][s]["accuracy"] for s in rows[1:] if s in acc[col]]
                cells.append(f"{100 * sum(vals) / len(vals):>{width}.2f}" if vals else f"{'-':>{width}}")
            else:
                r = acc[col].get(row)
                cells.append(f"{100 * r['accuracy']:>{width}.2f}" if r else f"{'-':>{width}}")
        out.append(row.ljust(16) + "".join(cells))
    out += ["", "questions scored (gold labels used / questions asked):",
            "set".ljust(16) + "".join(f"{k + '/' + m:>{width}}" for k, m in cols)]
    for row in rows:
        out.append(row.ljust(16) + "".join(
            (f"{str(acc[col][row]['n']) + '/' + str(acc[col][row]['asked']):>{width}}" if row in acc[col]
             else f"{'-':>{width}}") for col in cols))
    out += ["", "notes:",
            "  direct = highest-logit option digit (official SinhalaMMLU scoring; the only mode for the Base CPT model)",
            "  think  = Gemma's intended inference: English reasoning, then 'Answer: N' (its card: ~71% vs ~51% direct)",
            "  mmlu_easy scores 1850 of 1851: one Geography item has a broken gold label (answer 24 of 4 options)",
            "  the three shared_* sets use the proxy answer key from scripts/answer_sharedtask_test.py (~85-90%",
            "  correct), so differences of a few points on those rows are inside the label noise",
            f"  qwen-cpt = {MODELS['qwen-cpt']['hf_id']} (subfolder {MODELS['qwen-cpt']['subfolder']}), "
            f"gemma-sft = {MODELS['gemma-sft']['hf_id']} ({MODELS['gemma-sft']['revision']})"]
    return "\n".join(out) + "\n"


def write_outputs(results: list, limit: int, out: str) -> None:
    text = render_table(results, limit)
    print("\n" + text)
    (ROOT / out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / out).write_text(text, encoding="utf-8")
    preds_dir = ROOT / "comparison" / "predictions"
    preds_dir.mkdir(exist_ok=True)
    for r in results:
        for mode, sets in r["runs"].items():
            for name, run in sets.items():
                (preds_dir / f"{r['key']}_{mode}_{name}.jsonl").write_text(
                    "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in run["preds"]), encoding="utf-8")
    print(f"wrote {out} and {sum(len(s) for r in results for s in r['runs'].values())} prediction files")


@app.local_entrypoint()
def main(limit: int = 0, modes_qwen: str = "direct", modes_gemma: str = "direct,think",
         batch: int = 24, max_new: int = 2048, out: str = "comparison/results_table.txt", collect: bool = False):
    """limit=0 scores the full sets; --modes-gemma direct skips the slow reasoning pass; --batch sets how many
    reasoning traces are generated per forward pass (24 fits an L40S with room to spare, try 32-48); --collect
    runs no GPU and rebuilds the table from what the volume already holds (dropped client, detached run)."""
    if collect:
        runs = {}
        for entry in cache.listdir("results"):
            saved = json.loads(b"".join(cache.read_file(entry.path)).decode())
            if saved["limit"] == limit:
                runs.setdefault(saved["key"], {})[saved["mode"]] = saved["sets"]
                print("found", entry.path)
        if not runs:
            raise SystemExit(f"nothing in the volume for limit={limit}")
        return write_outputs([{"key": key, "runs": m} for key, m in runs.items()], limit, out)

    jobs = [(key, modes) for key, modes in (("qwen-cpt", modes_qwen), ("gemma-sft", modes_gemma)) if modes.strip()]
    calls = [evaluate.with_options(gpu=MODELS[key]["gpu"]).spawn(key, modes, limit, batch, max_new)
             for key, modes in jobs]
    write_outputs([call.get() for call in calls], limit, out)  # both models run in parallel, one container each
