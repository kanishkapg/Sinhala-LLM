"""LoRA SFT on Sinhala MCQs with short English reasoning, then merge into the base weights.

Usage (inside the vLLM image, after `pip install peft`):
  python sft_lora.py --model gemma4_e4b --train train.jsonl --out /tmp/sft [--epochs 1 --lr 1e-4 --rank 64]
Train rows: {"question", "choices": [4], "answer": 1-4, "reasoning_en"} (+ "subject", "grade" for --context).

Each example is rendered with the model's own chat template in thinking mode, using exactly the prompt that
bakeoff_run.py sends at evaluation (same system prompt and user message). The assistant turn is
`reasoning_content = reasoning_en`, `content = "Answer: N"`, so the model learns: short reasoning, close the
thinking block, answer. Loss is on the assistant part only. LoRA touches only language-model linear layers;
vision/audio towers are untouched. The adapter is merged at the end, so the parameter count does not change.
Writes <out>/merged (full model + tokenizer/processor files, loadable by vLLM) and <out>/train_log.json.

Multi-GPU: launch with `torchrun --nproc_per_node=N sft_lora.py ...`. Every rank builds the same micro-batch
stream (same seed) and takes every N-th micro-batch; LoRA gradients and target-token counts are summed across
ranks before each optimizer step, so results match the single-GPU recipe. Only rank 0 merges and saves.

--ckpt-dir DIR saves LoRA weights + optimizer + scheduler every --ckpt-every steps; a restarted job (Spot
preemption) resumes from there, fast-forwarding the deterministic micro-batch stream.

--docs adds plain-text passages (JSONL rows with "text", e.g. textbook chunks) trained with ordinary
language-model loss on every token, mixed with the MCQ examples: knowledge injection from the curriculum.
"""
import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
from bakeoff_run import SYSTEM, user_message


def build_examples(tok, rows, max_len, context=False):
    examples, too_long = [], 0
    for r in rows:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(r, context)}]
        asst = {"role": "assistant", "reasoning_content": r["reasoning_en"].strip(), "content": f"Answer: {r['answer']}"}
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        full = tok.apply_chat_template(msgs + [asst], tokenize=False, enable_thinking=True)
        assert full.startswith(prompt), "chat template does not extend the generation prompt"
        # The templates already contain BOS where needed; do not let the tokenizer add another.
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
        ids = p_ids + c_ids
        if len(ids) > max_len:
            too_long += 1
            continue
        examples.append({"input_ids": ids, "labels": [-100] * len(p_ids) + c_ids})
    return examples, too_long


def build_doc_examples(tok, texts, doc_len):
    """Each passage as one LM example (BOS + text), split into doc_len-token pieces if longer."""
    examples = []
    for text in texts:
        ids = tok(text, add_special_tokens=True)["input_ids"]
        for i in range(0, len(ids), doc_len):
            piece = ids[i:i + doc_len]
            if len(piece) >= 64:
                examples.append({"input_ids": piece, "labels": list(piece), "doc": True})
    return examples


def lora_targets(model):
    """Regex for every nn.Linear inside the language model (not lm_head, not vision/audio towers)."""
    leaves = set()
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "language_model" in name and "lm_head" not in name:
            leaves.add(name.rsplit(".", 1)[-1])
    if not leaves:
        raise RuntimeError("no language_model linear layers found")
    return r".*language_model.*\.(" + "|".join(sorted(leaves)) + r")$", sorted(leaves)


def load_model(hf_id):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    kw = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
    try:  # Gemma 4 / Qwen3.5 checkpoints are multimodal ...ForConditionalGeneration
        return AutoModelForImageTextToText.from_pretrained(hf_id, **kw)
    except (ValueError, KeyError):
        return AutoModelForCausalLM.from_pretrained(hf_id, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sinhala-mmlu/gemma-4-e4b-sinhala-mmlu", help="Hub model ID or local checkpoint directory")
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batch-tokens", type=int, default=16384, help="tokens per optimizer step (approx.)")
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--docs", default=None, help="JSONL of {'text': ...} passages trained with plain LM loss")
    ap.add_argument("--doc-len", type=int, default=2048)
    ap.add_argument("--qa-repeat", type=int, default=1, help="repeat the MCQ examples N times per epoch")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--ckpt-every", type=int, default=60)
    ap.add_argument("--context", action="store_true",
                    help="subject/grade line in the prompt (use with bakeoff_run.py --context at evaluation)")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, AutoTokenizer, get_cosine_schedule_with_warmup

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        import torch.distributed as dist
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group("nccl")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    hf_id = args.model
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(hf_id)
    rows = [json.loads(l) for l in open(args.train, encoding="utf-8")]
    random.shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    examples, too_long = build_examples(tok, rows, args.max_len, args.context)
    print(f"{len(examples)} MCQ examples ({too_long} too long)", flush=True)
    print("example completion:", repr(tok.decode([t for t in examples[0]["labels"] if t != -100])), flush=True)
    examples = examples * args.qa_repeat
    n_doc_tokens = 0
    if args.docs:
        texts = [json.loads(l)["text"] for l in open(args.docs, encoding="utf-8")]
        docs = build_doc_examples(tok, texts, args.doc_len)
        n_doc_tokens = sum(len(e["input_ids"]) for e in docs)
        print(f"{len(docs)} doc examples from {len(texts)} passages, {n_doc_tokens} tokens", flush=True)
        examples += docs
    mean_len = sum(len(e["input_ids"]) for e in examples) / len(examples)
    print(f"{len(examples)} training examples per epoch, mean {mean_len:.0f} tokens", flush=True)

    model = load_model(hf_id)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    pattern, leaves = lora_targets(model)
    print("LoRA on", leaves, flush=True)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
                                             target_modules=pattern, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.cuda()

    accum = max(1, round(args.batch_tokens / (mean_len * args.micro_batch)))
    accum = math.ceil(accum / world) * world  # every rank gets the same number of micro-batches per step
    steps_per_epoch = math.ceil(len(examples) / (args.micro_batch * accum))
    total_steps = max(1, round(steps_per_epoch * args.epochs))
    params = [p for p in model.parameters() if p.requires_grad]
    if world > 1:  # identical start on every rank (the seed already makes it so; this guarantees it)
        for p in params:
            dist.broadcast(p.data, 0)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * total_steps)), total_steps)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"micro_batch {args.micro_batch} x accum {accum} ({world} GPUs); {total_steps} optimizer steps", flush=True)

    def batches():
        while True:
            order = sorted(range(len(examples)), key=lambda i: random.random())
            # Group similar lengths to cut padding; shuffle the groups.
            order.sort(key=lambda i: len(examples[i]["input_ids"]) // 64 + random.random())
            groups = [order[i:i + args.micro_batch] for i in range(0, len(order), args.micro_batch)]
            random.shuffle(groups)
            yield from groups

    log, it, t0 = [], batches(), time.time()
    start = 1
    ckpt = Path(args.ckpt_dir) / "ckpt.pt" if args.ckpt_dir else None
    if ckpt and ckpt.exists():
        state = torch.load(ckpt, map_location="cuda", weights_only=False)
        named = dict(model.named_parameters())
        for name, value in state["lora"].items():
            named[name].data.copy_(value)
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        log, start = state["log"], state["step"] + 1
        for _ in range((start - 1) * accum):  # same stream as the interrupted run
            next(it)
        print(f"resumed from {ckpt} at step {start}", flush=True)
    model.train()
    for step in range(start, total_steps + 1):
        loss_sum, tok_sum = 0.0, 0
        for k in range(accum):
            idx = next(it)
            if k % world != rank:
                continue
            L = max(len(examples[i]["input_ids"]) for i in idx)
            ids = torch.full((len(idx), L), pad)
            lab = torch.full((len(idx), L), -100)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for j, i in enumerate(idx):
                e = examples[i]
                ids[j, :len(e["input_ids"])] = torch.tensor(e["input_ids"])
                lab[j, :len(e["labels"])] = torch.tensor(e["labels"])
                att[j, :len(e["input_ids"])] = 1
            n_tok = int((lab != -100).sum())
            outp = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda())
            # HF averages over the micro-batch's target tokens; re-weight so every token counts equally.
            (outp.loss * n_tok).backward()
            loss_sum += float(outp.loss) * n_tok
            tok_sum += n_tok
        if world > 1:
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)  # in place, one tensor at a time (no large extra buffer)
            totals = torch.tensor([loss_sum, tok_sum], device=params[0].device, dtype=torch.float64)
            dist.all_reduce(totals)
            loss_sum, tok_sum = float(totals[0]), int(totals[1])
        for p in params:
            if p.grad is not None:
                p.grad.div_(max(1, tok_sum))
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        log.append({"step": step, "loss": loss_sum / max(1, tok_sum), "lr": sched.get_last_lr()[0],
                    "target_tokens": tok_sum, "elapsed_s": round(time.time() - t0)})
        if (step % 10 == 0 or step == total_steps) and rank == 0:
            print(json.dumps(log[-1]), flush=True)
        if ckpt and rank == 0 and (step % args.ckpt_every == 0 or step == total_steps):
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            tmp = ckpt.with_suffix(".tmp")
            torch.save({"step": step, "log": log, "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "lora": {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}}, tmp)
            tmp.replace(ckpt)

    if world > 1:
        dist.barrier()
        if rank != 0:
            dist.destroy_process_group()
            return

    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "examples": len(examples),
                                                    "too_long": too_long, "doc_tokens": n_doc_tokens,
                                                    "lora_leaves": leaves,
                                                    "log": log}, indent=1))
    model = model.merge_and_unload()
    model.config.use_cache = True
    merged = out / "merged"
    model.save_pretrained(merged, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(merged)
    try:
        AutoProcessor.from_pretrained(hf_id).save_pretrained(merged)
    except Exception as e:  # noqa: BLE001 - text-only use does not need the processor
        print("[warn] processor not saved:", e, flush=True)
    # Keep the original chat template / generation config files byte-identical.
    from huggingface_hub import hf_hub_download
    for name in ("chat_template.jinja", "generation_config.json", "preprocessor_config.json", "LICENSE", "NOTICE"):
        try:
            local_file = Path(hf_id) / name
            source_file = local_file if local_file.is_file() else hf_hub_download(hf_id, name)
            shutil.copy(source_file, merged / name)
        except Exception:  # noqa: BLE001 - not every repo has every file
            pass
    parent_id = hf_id
    if Path(hf_id).is_dir():
        provenance_file = Path(hf_id) / "provenance.json"
        parent_id = json.loads(provenance_file.read_text()).get("model_id") if provenance_file.is_file() else None
    parent_header = f"base_model: {json.dumps(parent_id)}\nbase_model_relation: finetune\n" if parent_id else ""
    parent_label = parent_id or "a local checkpoint"
    (merged / "README.md").write_text(
        "---\nlicense: apache-2.0\nlibrary_name: transformers\n"
        + parent_header + "---\n\n"
        f"# Continued fine-tune of {parent_label}\n\n"
        "This checkpoint was produced by a new LoRA training run and merge. "
        "The parent model's published benchmark scores do not apply to these new weights. "
        "Evaluate the model on your own held-out data before reporting scores.\n"
    )
    (merged / "MODIFICATIONS.md").write_text(
        f"# Additional modifications\n\nNew language-model LoRA fine-tuning and merge from {parent_label}.\n\n"
        f"Training settings: epochs={args.epochs}, learning_rate={args.lr}, rank={args.rank}, alpha={args.alpha}.\n"
    )
    n_params = sum(p.numel() for p in model.parameters())
    (merged / "param_count.json").write_text(json.dumps({"total_parameters": n_params}))
    print(f"merged model saved to {merged} ({n_params / 1e9:.3f} B parameters)", flush=True)


if __name__ == "__main__":
    main()
