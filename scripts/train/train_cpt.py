"""Full-parameter continued pre-training (CPT) of Qwen/Qwen3.5-4B-Base on the packed Sinhala dataset.

The plan and the reasons behind every setting: docs/prepare_cpt_training_pipeline.md. Run guide: scripts/train/README.md.
Run from the repo root, inside tmux (never from a notebook):

    python scripts/train/train_cpt.py --config scripts/train/config/cpt_easy.json --smoke            # setup check, ~10 min
    python scripts/train/train_cpt.py --config scripts/train/config/cpt_easy.json --stop-at-step 25  # resume test, part 1
    python scripts/train/train_cpt.py --config scripts/train/config/cpt_easy.json --resume           # part 2 = the full run

Override config values with --set KEY=VALUE (JSON values), e.g. --set per_device_train_batch_size=4 learning_rate=1e-5
"""
import argparse
import hashlib
import importlib
import json
import logging
import math
import os
import platform
import subprocess
import sys
import uuid
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less fragmentation near 48 GB
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from cpt_callbacks import (CPTTrainer, EpochCopyCallback, GateCallback, ProbeCallback,  # noqa: E402
                           StopAtStepCallback, ThroughputCallback)
from cpt_data import boundary_steps, build_val_sets, check_dataset, collate, load_train  # noqa: E402
from probe_eval import ROOT, digit_token_ids, load_probes, load_sample_prompts, stdout_utf8, write_json  # noqa: E402

try:    # HF_TOKEN / WANDB_API_KEY from <repo>/.env (git-ignored); variables already set in the shell win
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

LN_VOCAB = math.log(248_320)       # ~12.42: the loss of uniform guessing over the vocabulary

# The DeltaNet kernels transformers looks for (section 4.4); without them it silently uses a much slower torch path.
KERNELS = {"flash-linear-attention": ("fla.ops.gated_delta_rule", "chunk_gated_delta_rule"),
           "causal-conv1d": ("causal_conv1d", "causal_conv1d_fn")}


# ---------------------------------------------------------------- config

def load_config(path: Path, overrides: list[str]) -> dict:
    """JSON config; "extends": "<file>" merges over another config in the same folder; --set KEY=VALUE on top."""
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if "extends" in cfg:
        cfg = load_config(path.parent / cfg.pop("extends"), []) | cfg
    for o in overrides:
        key, sep, value = o.partition("=")
        if not sep or key not in cfg:
            raise SystemExit(f"--set {o!r}: expected KEY=VALUE with a key of {path.name}")
        try:
            cfg[key] = json.loads(value)
        except json.JSONDecodeError:
            cfg[key] = value                      # bare strings: --set run_name=lr1e-5
    return cfg


# ---------------------------------------------------------------- kernels

def kernel_status() -> dict[str, bool]:
    status = {}
    for dist, (module, fn) in KERNELS.items():
        try:
            status[dist] = getattr(importlib.import_module(module), fn, None) is not None
        except Exception:
            status[dist] = False
    return status


class FallbackWarnings(logging.Handler):
    """Catches transformers' "`chunk_gated_delta_rule` is falling back to its reference PyTorch implementation"."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        msg = record.getMessage()
        if "falling back" in msg:
            self.messages.append(msg)


# ---------------------------------------------------------------- reproducibility record (section 7)

def _sha1(path: Path) -> str | None:
    return hashlib.sha1(path.read_bytes()).hexdigest() if path.exists() else None


def _git(*args) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return None


def git_commit() -> str | None:
    """HEAD of the repo; CPT_GIT_COMMIT when the code runs without its .git folder (Modal: modal_cpt.py)."""
    return _git("rev-parse", "HEAD") or os.environ.get("CPT_GIT_COMMIT") or None


def repro_record(cfg: dict, derived: dict, kernels: dict) -> dict:
    import accelerate
    import datasets
    import transformers
    probe_dir = ROOT / cfg["probe_dir"]
    return {
        "config": cfg,
        "derived": derived,
        "argv": sys.argv,
        "git_commit": git_commit(),
        "git_dirty": bool(_git("status", "--porcelain")) or os.environ.get("CPT_GIT_DIRTY") == "1",
        "dataset_manifest_sha1": _sha1(ROOT / cfg["data_dir"] / "manifest.json"),
        "probe_sha1": {f: _sha1(probe_dir / f) for f in ("probe_a_subset.jsonl", "probe_a_full.jsonl", "probe_b.jsonl")},
        "probe_b_author": "claude-opus-5-5 (closed model: disclose in the system report)",
        "seed": cfg["seed"],
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                     "transformers": transformers.__version__, "datasets": datasets.__version__,
                     "accelerate": accelerate.__version__},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "kernels": kernels,
    }


def pip_freeze() -> str:
    try:
        return subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120).stdout
    except Exception as e:
        return f"pip freeze failed: {e}\n"


def init_wandb(cfg: dict, run_name: str, out: Path, record: dict) -> None:
    """Start the W&B run ourselves (the Trainer's callback then reuses it): the run id is kept in
    <output_dir>/wandb_run_id.txt, so a resumed run continues the same charts."""
    import wandb
    id_file = out / "wandb_run_id.txt"
    run_id = id_file.read_text().strip() if id_file.exists() else uuid.uuid4().hex[:8]   # wandb.util.generate_id is gone in 0.31
    id_file.write_text(run_id)
    wandb.init(project=cfg["wandb_project"], name=run_name, id=run_id, resume="allow", dir=str(out), config=record)
    for f in ("repro.json", "pip_freeze.txt"):
        wandb.save(str(out / f), base_path=str(out), policy="now")


# ---------------------------------------------------------------- smoke-test report (section 8.6)

def smoke_report(trainer, throughput, catcher, kernels, cfg, derived, probes) -> None:
    hist = trainer.state.log_history
    val0 = {k: v for h in hist if h.get("step") == 0 for k, v in h.items()
            if k.startswith("eval_") and k.endswith("_loss") and not k.startswith("eval_probe")}
    eval_seconds = sum(v for h in hist if h.get("step") == 0 for k, v in h.items() if k.endswith("_runtime"))
    probe_seconds = baseline_value(hist, "eval_probe_seconds")
    train = [h for h in hist if "loss" in h]
    first_loss = train[0]["loss"] if train else float("nan")
    step_s = throughput.step_seconds()
    max_steps = derived["max_steps"]
    rows = []

    mean_val0 = sum(val0.values()) / len(val0) if val0 else float("nan")
    ok = first_loss < LN_VOCAB - 2 and abs(first_loss - mean_val0) < 0.5 * mean_val0
    rows.append(("first training loss vs step-0 val loss", ok,
                 f"{first_loss:.3f} vs {mean_val0:.3f} (mean of {', '.join(f'{k[5:-5]} {v:.2f}' for k, v in val0.items())});"
                 f" ~{LN_VOCAB:.1f} would mean a data/label bug"))
    ok = all(kernels.values()) and not catcher.messages
    rows.append(("DeltaNet kernels, no fallback warning", ok,
                 f"{kernels}; fallback warnings: {len(catcher.messages)}" + (f" ({catcher.messages[0][:90]}...)" if catcher.messages else "")))
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        peak = torch.cuda.max_memory_reserved() / 1e9
        rows.append(("GPU memory headroom >= 2 GB", total - peak >= 2,
                     f"peak reserved {peak:.1f} of {total:.1f} GB (micro-batch {cfg['per_device_train_batch_size']})"))
    if step_s:
        n_evals = max_steps // cfg["eval_steps"] + 1
        n_probes = max_steps // cfg["probe_steps"] + len(derived["epoch_boundaries"]) + 1
        n_q = len(probes["a_subset"]) + len(probes["b"])
        per_q = (probe_seconds or 0) / max(n_q, 1)
        probe_total = per_q * (n_probes * (504 + 405) + len(derived["epoch_boundaries"]) * 1826)
        train_h = max_steps * step_s / 3600
        extra_h = (n_evals * eval_seconds + probe_total) / 3600
        rows.append(("throughput / ETA", True,
                     f"{cfg['global_batch'] * cfg['seq_len'] / step_s:,.0f} tokens/s, {step_s:.1f} s/step -> training "
                     f"{train_h:.2f} h + evaluation ~{extra_h:.2f} h for {max_steps} steps "
                     f"(eval {eval_seconds:.0f} s x {n_evals}, probes ~{per_q * 1000:.0f} ms/question)"))
    print("\n" + "=" * 100 + "\nSMOKE TEST (section 8.6)\n" + "=" * 100)
    for name, ok, detail in rows:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}")
    print("=" * 100 + "\nNext: the resume test (README step 3) if every line passed.\n")


def baseline_value(hist, key):
    return next((h[key] for h in hist if h.get("step") == 0 and key in h), None)


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="Full-parameter CPT of Qwen3.5-4B-Base (docs/prepare_cpt_training_pipeline.md)")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--smoke", action="store_true",
                    help="setup check: step-0 eval + small probes, `smoke_steps` updates, no checkpoints, into <output_dir>-smoke")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint in output_dir")
    ap.add_argument("--stop-at-step", type=int, help="stop (and save a checkpoint) after N steps; the LR schedule is unchanged")
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", help="override config values")
    args = ap.parse_args()
    stdout_utf8()
    cfg = load_config(args.config, args.set)
    cuda = torch.cuda.is_available()

    out = ROOT / cfg["output_dir"]
    run_name = cfg["run_name"]
    if args.smoke:
        out, run_name = out.with_name(out.name + "-smoke"), run_name + "-smoke"
    os.environ.setdefault("WANDB_PROJECT", cfg["wandb_project"])
    os.environ.setdefault("WANDB_LOG_MODEL", "false")      # never upload checkpoints to W&B
    os.environ.setdefault("WANDB_WATCH", "false")          # gradient histograms of 4B parameters: too slow

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    last = get_last_checkpoint(str(out)) if out.is_dir() else None
    if args.resume and not last:
        raise SystemExit(f"--resume: no checkpoint-* folder in {out}")
    if last and not args.resume and not args.smoke:
        raise SystemExit(f"{last} exists: add --resume to continue that run, or choose another output_dir")
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])

    # ---- kernels (section 4.4)
    kernels = kernel_status()
    catcher = FallbackWarnings()
    logging.getLogger("transformers.integrations.hub_kernels").addHandler(catcher)
    print(f"DeltaNet kernels: {kernels}")
    if cuda and not all(kernels.values()) and not cfg["allow_slow_kernels"]:
        raise SystemExit("flash-linear-attention and causal-conv1d must be installed on the GPU machine (README, setup); "
                         "--set allow_slow_kernels=true to run anyway (>10x slower DeltaNet layers)")

    # ---- data (sections 2, 4.1, 8.2)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    data_dir = ROOT / cfg["data_dir"]
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    train_ds, segment_ends = load_train(data_dir, cfg["seq_len"], cfg["limit_blocks_per_epoch"])
    check_dataset(manifest, train_ds, tokenizer, full_data=not cfg["limit_blocks_per_epoch"])
    val_sets = build_val_sets(data_dir / "mix" / "val.jsonl", tokenizer, cfg["seq_len"], manifest["eos_id"],
                              cfg["val_limit_blocks"])

    # ---- steps and schedule (section 4)
    micro = cfg["per_device_train_batch_size"]
    world = int(os.environ.get("WORLD_SIZE", 1))
    if cfg["global_batch"] % (micro * world):
        raise SystemExit(f"global_batch {cfg['global_batch']} is not a multiple of micro-batch {micro} x {world} GPU(s)")
    grad_accum = cfg["global_batch"] // (micro * world)
    max_steps = math.ceil(len(train_ds) / cfg["global_batch"])
    decay_steps = max(1, round(cfg["decay_ratio"] * max_steps))
    boundaries = boundary_steps(segment_ends, cfg["global_batch"])
    if max(boundaries) != max_steps:
        raise SystemExit(f"last epoch boundary {max(boundaries)} != {max_steps} steps")
    derived = {"max_steps": max_steps, "grad_accum": grad_accum, "warmup_steps": math.ceil(cfg["warmup_ratio"] * max_steps),
               "decay_steps": decay_steps, "epoch_boundaries": boundaries,
               "check1_step": math.ceil(cfg["check1_fraction"] * max_steps),
               "tokens_per_step": cfg["global_batch"] * cfg["seq_len"], "train_tokens": len(train_ds) * cfg["seq_len"]}
    print(f"plan: {max_steps} steps of {cfg['global_batch']} x {cfg['seq_len']} tokens ({micro} x {grad_accum} accumulation), "
          f"warm-up {derived['warmup_steps']}, decay {decay_steps} to {cfg['min_lr_ratio']:.0%} of {cfg['learning_rate']}, "
          f"epoch copies at {boundaries}, Check 1 at the first eval >= step {derived['check1_step']}")

    # ---- model (section 3)
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=getattr(torch, cfg.get("model_dtype", "bfloat16")))
    if "ConditionalGeneration" in type(model).__name__:
        raise SystemExit(f"loaded {type(model).__name__}; the text-only Qwen3_5ForCausalLM is expected")
    model.config.use_cache = False
    n_params = sum(p.numel() for p in model.parameters())
    tied = model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    derived |= {"model_class": type(model).__name__, "n_params": n_params, "tied_embeddings": tied}
    print(f"{type(model).__name__}: {n_params / 1e9:.3f}B parameters, embeddings tied: {tied}")
    if not tied:
        print("WARNING: input and output embeddings are not tied (the Qwen3.5-4B-Base config ties them)")

    # ---- probes (section 8.3)
    probes = load_probes(ROOT / cfg["probe_dir"], cfg["smoke_probe_limit"] if args.smoke else cfg["probe_limit"])
    digit_ids = digit_token_ids(tokenizer)
    prompts = load_sample_prompts(ROOT / cfg["sample_prompts"])

    targs = TrainingArguments(
        output_dir=str(out),
        run_name=run_name,
        num_train_epochs=1,                                   # the 3 textbook epochs are already in the data
        per_device_train_batch_size=micro,
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg["warmup_ratio"],                     # a float < 1 is a ratio in transformers v5
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs={"num_decay_steps": decay_steps, "decay_type": "linear", "min_lr_ratio": cfg["min_lr_ratio"]},
        optim=cfg["optim"],                                   # adamw_torch_8bit = torchao AdamW8bit
        optim_args=cfg["optim_args"],                         # bf16_stochastic_round=True (section 3.4)
        adam_beta1=cfg["adam_beta1"],
        adam_beta2=cfg["adam_beta2"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        bf16=cfg["bf16"] and cuda,
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        train_sampling_strategy="sequential",                 # keep the epoch segments in order (section 4.2)
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        remove_unused_columns=False,
        logging_steps=1,
        include_num_input_tokens_seen=True,
        report_to=cfg["report_to"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        eval_on_start=not args.resume,                        # step 0 = the base model (section 8.1)
        prediction_loss_only=True,                            # never gather 248k-wide logits during eval
        save_strategy="no" if args.smoke else "steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
    )

    throughput = ThroughputCallback()
    stop_step = cfg["smoke_steps"] if args.smoke else args.stop_at_step
    tracked = {} if args.smoke else boundaries
    epoch_copies = EpochCopyCallback(tracked, out, None if args.smoke else cfg["hub_repo"],
                                     {"run_name": run_name, "base_model": cfg["model"], "git_commit": git_commit(),
                                      "dataset_manifest_sha1": _sha1(data_dir / "manifest.json")},
                                     derived["tokens_per_step"])
    gates = GateCallback(derived["check1_step"], cfg["replay_loss_tolerance"], cfg["gate_action"])
    probe_cb = ProbeCallback(tokenizer, probes, digit_ids, prompts, cfg["sample_max_new_tokens"], cfg["probe_steps"],
                             tracked, out / "probes", cfg["probe_at_start"], cfg["probe_full_at_start"] and not args.smoke,
                             gates)
    # order matters: time the step first, save the epoch copy before probing it
    callbacks = [throughput, epoch_copies, probe_cb, gates]
    if stop_step:
        callbacks.append(StopAtStepCallback(stop_step, save=not args.smoke, evaluate=True))
    trainer = CPTTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_sets, data_collator=collate,
                         processing_class=tokenizer, callbacks=callbacks, throughput=throughput,
                         tokens_per_step=derived["tokens_per_step"])
    for cb in (epoch_copies, probe_cb, gates):
        cb.trainer = trainer

    record = repro_record(cfg, derived, kernels)
    write_json(out / "repro.json", record)
    (out / "pip_freeze.txt").write_text(pip_freeze(), encoding="utf-8")
    if "wandb" in (cfg["report_to"] if isinstance(cfg["report_to"], list) else [cfg["report_to"]]):
        init_wandb(cfg, run_name, out, record)

    trainer.train(resume_from_checkpoint=True if args.resume else None)

    if args.smoke:
        smoke_report(trainer, throughput, catcher, kernels, cfg, derived, probes)
    else:
        if catcher.messages:
            print(f"WARNING: kernel fallback during training: {catcher.messages}")
        done = trainer.state.global_step >= max_steps
        print(f"\n{'finished' if done else 'stopped'} at step {trainer.state.global_step}/{max_steps}; output: {out}")
        copies = sorted(p.name for p in (out / "epochs").glob("e*") if (p / "cpt_info.json").exists())
        print(f"epoch copies: {copies or 'none yet'}; probe records: {out / 'probes'}")
        if not done:
            print("continue with: --resume")


if __name__ == "__main__":
    main()
