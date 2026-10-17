"""Trainer pieces for CPT (docs/prepare_cpt_training_pipeline.md, sections 6-8).

- CPTTrainer          adds tokens/s, step time, ETA and peak GPU memory to every training log line (section 7)
- ThroughputCallback  times the optimizer steps alone (evaluation, probes and saving run between steps)
- EpochCopyCallback   weights-only copy at each epoch boundary -> <output_dir>/epochs/eN + private HF repo (section 6)
- ProbeCallback       Probe A / Probe B / fixed samples at step 0, every `probe_steps` and at the boundaries (8.3-8.5)
- GateCallback        stop points: Check 1 at ~20 % (validation losses), Check 2 at the end of epoch 1 (probes) (8.6)
- StopAtStepCallback  ends a run early WITHOUT changing the LR schedule: smoke test and resume test

Probe and gate metrics go through trainer.log_extra() with an "eval_" prefix, so they land in W&B under eval/ and in
trainer_state.json (log_history), which a resumed run reads back for the step-0 baselines.
"""
import statistics
import time
from collections import deque
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback

from probe_eval import run_probes, save_details, std_error, write_json


def _wandb():
    """The wandb module if a run is active, else None."""
    try:
        import wandb
    except ImportError:
        return None
    return wandb if wandb.run is not None else None


def alert(title: str, text: str, level: str = "WARN") -> None:
    """Print a banner and, with W&B, send an alert (e-mail / app notification, see the W&B alert settings)."""
    bar = "!" * 100
    print(f"\n{bar}\n{title}\n{text}\n{bar}\n", flush=True)
    wb = _wandb()
    if wb:
        try:
            wb.alert(title=title, text=text, level=getattr(wb.AlertLevel, level))
        except Exception as e:  # an alert must never kill the run
            print(f"(W&B alert failed: {e})")


def baseline(state, key: str):
    """Value of `key` logged at step 0 (before the first update), from the Trainer's log history."""
    for h in state.log_history:
        if h.get("step") == 0 and key in h:
            return h[key]
    return None


class CPTTrainer(Trainer):
    def __init__(self, *args, throughput=None, tokens_per_step: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.throughput = throughput
        self.tokens_per_step = tokens_per_step

    def log(self, logs: dict, start_time: float | None = None) -> None:
        if "loss" in logs:                              # a training log line, not eval / probes
            dt = self.throughput.step_seconds() if self.throughput else None
            if dt:
                logs["step_seconds"] = dt
                logs["tokens_per_second"] = self.tokens_per_step / dt
                logs["eta_train_minutes"] = (self.state.max_steps - self.state.global_step) * dt / 60
            if torch.cuda.is_available():
                logs["gpu_mem_peak_gb"] = torch.cuda.max_memory_reserved() / 1e9
        super().log(logs, start_time)

    def log_extra(self, logs: dict) -> None:
        """log() from a callback. on_log clears control.should_log, so a call from on_step_end would silently drop
        the training-loss line of that step (every probe step and every epoch boundary): keep the flag."""
        should_log = self.control.should_log
        self.log(logs)
        self.control.should_log = should_log


class ThroughputCallback(TrainerCallback):
    def __init__(self):
        self.times = deque(maxlen=20)
        self._t0 = None

    def on_step_begin(self, args, state, control, **kwargs):
        self._t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self._t0 is not None:
            self.times.append(time.perf_counter() - self._t0)
            self._t0 = None

    def step_seconds(self) -> float | None:
        """Median of the last 20 optimizer steps (the first step includes kernel warm-up)."""
        return statistics.median(self.times) if self.times else None


class EpochCopyCallback(TrainerCallback):
    """At each epoch-boundary step: check the tied embeddings, save_pretrained (bf16 safetensors + tokenizer) into
    <output_dir>/epochs/<name>, write cpt_info.json, and upload the folder to `hub_repo` (private) in the background.
    Only these weights-only copies leave the machine, never the 17 GB optimizer checkpoints."""

    def __init__(self, boundaries: dict[int, str], out_dir: Path, hub_repo: str | None, info: dict, tokens_per_step: int):
        self.boundaries = boundaries
        self.out_dir = Path(out_dir)
        self.hub_repo = hub_repo
        self.info = info
        self.tokens_per_step = tokens_per_step
        self.trainer = None                 # set by train_cpt.py once the Trainer exists
        self.n_params = None
        self.futures = []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.n_params = sum(p.numel() for p in model.parameters())
        if self.hub_repo:
            from huggingface_hub import HfApi
            HfApi().create_repo(self.hub_repo, private=True, exist_ok=True)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        name = self.boundaries.get(state.global_step)
        if name:
            self.save(model, state, name)

    def save(self, model, state, name: str) -> None:
        d = self.out_dir / "epochs" / name
        if (d / "cpt_info.json").exists():
            print(f"epoch copy {name} already exists ({d}): skipped")
            return
        n = sum(p.numel() for p in model.parameters())
        tied = model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
        if not tied or n != self.n_params:
            alert("Tied embeddings check failed",
                  f"{name}: tied={tied}, parameters {n:,} vs {self.n_params:,} at the start. lm_head and embed_tokens "
                  "must stay one tensor, or the saved model grows by ~0.64B (section 9).", "ERROR")
        use_cache = model.config.use_cache
        model.config.use_cache = True                    # training runs with False; inference wants True
        try:
            self.trainer.save_model(str(d))
        finally:
            model.config.use_cache = use_cache
        write_json(d / "cpt_info.json", {"epoch_copy": name, "step": state.global_step, "max_steps": state.max_steps,
                                         "tokens_trained": state.global_step * self.tokens_per_step,
                                         "n_params": n, "tied_embeddings": tied, **self.info})
        print(f"epoch copy {name} (step {state.global_step}) -> {d}")
        if self.hub_repo:
            from huggingface_hub import HfApi
            fut = HfApi().upload_folder(repo_id=self.hub_repo, folder_path=str(d), path_in_repo=name,
                                        commit_message=f"{name}: step {state.global_step}", run_as_future=True)
            self.futures.append((name, d, fut))
            print(f"uploading {name} to {self.hub_repo}/{name} in the background")

    def on_train_end(self, args, state, control, **kwargs):
        for name, d, fut in self.futures:
            print(f"waiting for the upload of {name} ...", flush=True)
            try:
                fut.result()
                print(f"uploaded {name}")
            except Exception as e:
                alert("Hub upload failed", f"{name}: {e}\nUpload by hand: hf upload {self.hub_repo} {d} {name}", "ERROR")
        self.futures = []


class ProbeCallback(TrainerCallback):
    def __init__(self, tokenizer, probes: dict, digit_ids: list[int], prompts: list[str], max_new_tokens: int,
                 probe_steps: int, boundaries: dict[int, str], out_dir: Path, at_start: bool, full_at_start: bool,
                 gates=None):
        self.tokenizer = tokenizer
        self.probes = probes
        self.digit_ids = digit_ids
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens
        self.probe_steps = probe_steps
        self.boundaries = boundaries
        self.out_dir = Path(out_dir)
        self.at_start = at_start
        self.full_at_start = full_at_start
        self.gates = gates
        self.trainer = None
        self.sample_rows = []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # Step 0 = the base model: the reference line of every probe chart (8.1). Not repeated on resume.
        if self.at_start and state.global_step == 0:
            self.run(model, state, full=self.full_at_start)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        s = state.global_step
        boundary = s in self.boundaries or s == state.max_steps
        if boundary or s % self.probe_steps == 0:
            metrics = self.run(model, state, full=boundary)       # all 1,826 Probe A questions at the boundaries
            if self.gates and self.boundaries.get(s) == "e1":
                self.gates.check2(state, control, metrics)

    def run(self, model, state, full: bool) -> dict:
        s = state.global_step
        metrics, details = run_probes(model, self.tokenizer, self.probes, self.digit_ids, full,
                                      self.prompts, self.max_new_tokens)
        details["step"] = s
        save_details(details, self.out_dir, f"step_{s:04d}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()                               # generation caches
        line = (f"[probes step {s}] A(subset) {metrics['probe_a_acc']:.3f}"
                + (f"  A(full) {metrics['probe_a_full_acc']:.3f}" if full else "")
                + f"  B train {metrics['probe_b_train_acc']:.3f}  B held-out {metrics['probe_b_heldout_acc']:.3f}"
                + f"  ({metrics['probe_seconds']:.0f}s)")
        print(line, flush=True)
        self.trainer.log_extra({f"eval_{k}": v for k, v in metrics.items()})
        wb = _wandb()
        if wb and details["samples"]:
            self.sample_rows += [[s, x["prompt"], x["output"], x["distinct2"], x["sinhala_ratio"]]
                                 for x in details["samples"]]
            wb.log({"samples": wb.Table(columns=["step", "prompt", "output", "distinct2", "sinhala_ratio"],
                                        data=self.sample_rows), "train/global_step": s})
        return metrics


class GateCallback(TrainerCallback):
    """Section 8.6. A failed check sends an alert; with gate_action="stop" it also ends the run (the last full
    checkpoint and any epoch copy stay on disk)."""

    LOSSES = ["eval_textbook_loss", "eval_wiki_loss", "eval_fineweb2_loss", "eval_english_loss"]

    def __init__(self, check1_step: int, replay_tolerance: float, action: str):
        if action not in ("alert", "stop"):
            raise SystemExit(f"gate_action must be 'alert' or 'stop', not {action!r}")
        self.check1_step = check1_step
        self.replay_tolerance = replay_tolerance
        self.action = action
        self.trainer = None
        self.check1_done = False
        self._evals = {}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # The 4 eval sets are evaluated one after the other, each with its own on_evaluate call: collect them.
        if self.check1_done or state.global_step < self.check1_step or not metrics:
            return
        if any("eval_gate_check1" in h for h in state.log_history):      # already ran before a resume
            self.check1_done = True
            return
        cur = self._evals.setdefault(state.global_step, {})
        cur.update(metrics)
        base = {k: baseline(state, k) for k in self.LOSSES if baseline(state, k) is not None}
        if not base:
            print("Check 1 skipped: no step-0 validation losses in the log history (eval_on_start was off)")
            self.check1_done = True
            return
        if not all(k in cur for k in base):
            return
        self.check1_done = True
        lines, ok = [], True
        for k in base:
            change = cur[k] / base[k] - 1
            if k == "eval_textbook_loss":
                good = cur[k] < base[k]
                rule = "must go down"
            else:
                good = change <= self.replay_tolerance
                rule = f"must stay within +{self.replay_tolerance:.0%}"
            ok &= good
            lines.append(f"  {k:<22} step 0 {base[k]:.4f} -> step {state.global_step} {cur[k]:.4f} "
                         f"({change:+.1%}, {rule}) {'ok' if good else 'FAIL'}")
        self.report("Check 1", ok, lines, state, control,
                    "If it failed: stop and change the LR (replay up = forgetting -> lower LR; textbook flat -> bug or LR too low).")

    def check2(self, state, control, metrics: dict) -> None:
        """End of epoch 1: Probe B 'in training' up by more than the noise; Probe A not down by more than the noise."""
        if any("eval_gate_check2" in h for h in state.log_history):
            return
        b0, nb = baseline(state, "eval_probe_b_train_acc"), metrics["probe_b_train_n"]
        h0 = baseline(state, "eval_probe_b_heldout_acc")
        a_key = "probe_a_full_acc" if baseline(state, "eval_probe_a_full_acc") is not None else "probe_a_acc"
        a0 = baseline(state, f"eval_{a_key}")
        if b0 is None or a0 is None:
            print("Check 2 skipped: no step-0 probe results in the log history (probe_at_start was off)")
            return
        na = metrics.get(f"{a_key.replace('_acc', '')}_n", metrics["probe_a_n"])
        db, noise_b = metrics["probe_b_train_acc"] - b0, 2 * std_error(b0, nb)
        da, noise_a = metrics[a_key] - a0, 2 * std_error(a0, na)
        dh = metrics["probe_b_heldout_acc"] - h0 if h0 is not None else float("nan")
        ok_b, ok_a = db > noise_b, da > -noise_a
        lines = [f"  Probe B in training  {b0:.3f} -> {metrics['probe_b_train_acc']:.3f} ({100 * db:+.1f} pts, "
                 f"must be > +{100 * noise_b:.1f}) {'ok' if ok_b else 'FAIL'}",
                 f"  Probe B held-out     {h0 if h0 is not None else float('nan'):.3f} -> "
                 f"{metrics['probe_b_heldout_acc']:.3f} ({100 * dh:+.1f} pts; only the training-side gain is knowledge)",
                 f"  {a_key:<20} {a0:.3f} -> {metrics[a_key]:.3f} ({100 * da:+.1f} pts, "
                 f"must be > -{100 * noise_a:.1f}) {'ok' if ok_a else 'FAIL'}"]
        self.report("Check 2", ok_b and ok_a, lines, state, control,
                    "If it failed: stop and rethink (LR, epochs, reworded textbooks) - section 8.3.")

    def report(self, name: str, ok: bool, lines: list[str], state, control, advice: str) -> None:
        text = "\n".join(lines)
        if self.trainer is not None:
            self.trainer.log_extra({f"eval_gate_{name.lower().replace(' ', '')}": float(ok)})
        if ok:
            print(f"\n{name} at step {state.global_step}: PASS\n{text}\n", flush=True)
            return
        stop = self.action == "stop"
        alert(f"{name} FAILED at step {state.global_step}" + (" - stopping the run" if stop else ""),
              f"{text}\n{advice}", "ERROR")
        if stop:
            control.should_training_stop = True


class StopAtStepCallback(TrainerCallback):
    """Stop after `step` optimizer steps, keeping the full-run LR schedule. With save=True a full checkpoint is
    written at that step (the resume test resumes from it); evaluate=True adds a validation pass there."""

    def __init__(self, step: int, save: bool, evaluate: bool):
        self.step = step
        self.save = save
        self.evaluate = evaluate

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.step:
            control.should_training_stop = True
            if self.save:
                control.should_save = True
            if self.evaluate:
                control.should_evaluate = True
