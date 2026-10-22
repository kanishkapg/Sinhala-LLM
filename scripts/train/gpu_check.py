"""GPU checks before spending GPU hours (docs/prepare_cpt_training_pipeline.md, sections 3.4 and 4.4).

Run after setup_machine.sh on the training machine, or on a free Colab / Kaggle GPU (colab_gpu_check.ipynb):

    python scripts/train/gpu_check.py

1. GPU, torch, CUDA, bf16 support.
2. DeltaNet kernels: flash-linear-attention's chunk_gated_delta_rule and causal-conv1d's causal_conv1d_fn against
   transformers' own PyTorch reference, on the 4B model's head sizes, outputs AND gradients, per dtype.
3. 8-bit AdamW with bf16 stochastic rounding (torchao): updates far below bf16 precision must still move bf16
   weights, while plain AdamW on bf16 weights leaves them unchanged (the rounding trap of section 3.4).

WARN = a little outside the expected precision (not a broken kernel; a wrong one gives errors near 1).
Exit code 1 if a check needed for training fails. A T4 (Colab / Kaggle) has no native bf16: there a bf16 failure is
a T4 limit, and the float16 / float32 rows tell whether the kernels themselves work.
"""
import argparse
import importlib
import inspect
import sys
import traceback

import torch

# PASS / WARN thresholds: (dtype, max relative error of the output, of the gradients) vs the fp32 PyTorch reference
TOLERANCE = {torch.bfloat16: (0.03, 0.08), torch.float16: (0.01, 0.05), torch.float32: (0.005, 0.02)}
results = []          # (check, status, detail); status PASS / FAIL / SKIP / INFO


def report(check: str, status: str, detail: str) -> None:
    results.append((check, status, detail))
    print(f"[{status}] {check}: {detail}", flush=True)


def grade(e_out: float, e_grad: float, dtype) -> str:
    """PASS within the dtype's precision; WARN if a bit above (precision, not a broken kernel); FAIL if far off
    (a wrong kernel gives relative errors near 1)."""
    tol_out, tol_grad = TOLERANCE[dtype]
    if e_out < tol_out and e_grad < tol_grad:
        return "PASS"
    return "WARN" if max(e_out, e_grad) < 0.3 else "FAIL"


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def load(module: str, name: str):
    try:
        return getattr(importlib.import_module(module), name)
    except Exception as e:
        return e


# ---------------------------------------------------------------- 1. environment

def check_environment(dev: str) -> None:
    import transformers
    report("versions", "INFO", f"python {sys.version.split()[0]}, torch {torch.__version__}, CUDA {torch.version.cuda}, "
                               f"transformers {transformers.__version__}")
    if dev == "cpu":
        report("GPU", "FAIL", "no CUDA GPU visible (Colab: Runtime -> Change runtime type -> T4 GPU)")
        return
    p = torch.cuda.get_device_properties(0)
    native_bf16 = p.major >= 8
    report("GPU", "INFO", f"{p.name}, {p.total_memory / 1e9:.0f} GB, compute capability {p.major}.{p.minor}, "
                          f"{torch.cuda.device_count()} visible")
    report("bf16", "INFO", "native (Ampere or newer: L40S, A100, L4)" if native_bf16 else
           "NOT native on this GPU (T4 / P100): bf16 results below may fail or be emulated; the L40S has it")
    for mod, name in [("fla.ops.gated_delta_rule", "chunk_gated_delta_rule"), ("causal_conv1d", "causal_conv1d_fn"),
                      ("torchao.optim", "AdamW8bit")]:
        f = load(mod, name)
        ok = not isinstance(f, Exception)
        version = ""
        if ok:
            dist = {"fla": "flash-linear-attention", "causal_conv1d": "causal-conv1d", "torchao": "torchao"}[mod.split(".")[0]]
            try:
                from importlib.metadata import version as v
                version = f" {dist} {v(dist)}"
            except Exception:
                pass
        report(f"import {mod}.{name}", "PASS" if ok else "FAIL", ("ok" + version) if ok else f"{type(f).__name__}: {f}")
    # transformers has its own torchao version gate (5.19: >= 0.15.0), stricter than the import above
    check = "transformers selects torchao AdamW8bit"
    try:
        import tempfile
        from transformers import Trainer, TrainingArguments
        targs = TrainingArguments(output_dir=tempfile.mkdtemp(), optim="adamw_torch_8bit",
                                  optim_args="bf16_stochastic_round=True", report_to="none")
        cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(targs)
        ok = cls.__name__ == "AdamW8bit" and bool(kwargs.get("bf16_stochastic_round"))   # transformers passes 1
        report(check, "PASS" if ok else "FAIL",
               f"{cls.__module__}.{cls.__name__}, bf16_stochastic_round={kwargs.get('bf16_stochastic_round')}")
    except Exception as e:
        report(check, "FAIL", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- 2. kernels vs reference

def gated_delta_inputs(dtype, dev, B=2, T=1000, H=4, Dk=128, Dv=128, seed=0):
    """Shapes and value ranges as in Qwen3_5GatedDeltaNet.forward (T not a multiple of the 64-token chunk on purpose)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, T, H, Dk, generator=g)
    k = torch.randn(B, T, H, Dk, generator=g)
    v = torch.randn(B, T, H, Dv, generator=g)
    a = torch.randn(B, T, H, generator=g)
    decay = -torch.rand(H, generator=g).add(0.5) * torch.nn.functional.softplus(a)   # g <= 0, float32 as in the model
    beta = torch.randn(B, T, H, generator=g).sigmoid()
    w = torch.randn(B, T, H, Dv, generator=g)                                         # fixed projection for a scalar loss
    tensors = [x.to(dev, dtype) for x in (q, k, v, beta)] + [decay.to(dev), w.to(dev)]
    return tensors


def run_gdr(fn, q, k, v, beta, decay, w):
    q, k, v, beta, decay = [x.detach().clone().requires_grad_(True) for x in (q, k, v, beta, decay)]
    out, _ = fn(q, k, v, g=decay, beta=beta, use_qk_l2norm_in_kernel=True)
    (out.float() * w).sum().backward()
    return out.detach(), [x.grad for x in (q, k, v, beta, decay)]


def check_gated_delta(dev: str, dtypes) -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    reference = inspect.unwrap(mq.torch_chunk_gated_delta_rule)        # the plain PyTorch path, computes in fp32
    fast = load("fla.ops.gated_delta_rule", "chunk_gated_delta_rule")
    if isinstance(fast, Exception) or dev == "cpu":
        report("chunk_gated_delta_rule", "SKIP" if dev == "cpu" else "FAIL",
               "needs a CUDA GPU" if dev == "cpu" else "flash-linear-attention is not importable")
        return
    inputs32 = gated_delta_inputs(torch.float32, dev)
    ref_out, ref_grads = run_gdr(reference, *inputs32)
    for dtype in dtypes:
        name = f"chunk_gated_delta_rule {str(dtype).split('.')[-1]}"
        try:
            out, grads = run_gdr(fast, *gated_delta_inputs(dtype, dev))
            torch.cuda.synchronize()
        except Exception as e:
            report(name, "FAIL", f"kernel error: {type(e).__name__}: {str(e).splitlines()[0][:150]}")
            continue
        tol_out, tol_grad = TOLERANCE[dtype]
        e_out = rel_err(out, ref_out)
        e_grad = max(rel_err(a, b) for a, b in zip(grads, ref_grads))
        report(name, grade(e_out, e_grad, dtype) if torch.isfinite(out).all() else "FAIL",
               f"relative error vs PyTorch reference: output {e_out:.4f} (< {tol_out}), worst gradient {e_grad:.4f} (< {tol_grad})")


def check_conv(dev: str, dtypes) -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    reference = inspect.unwrap(mq.causal_conv1d_fn)
    fast = load("causal_conv1d", "causal_conv1d_fn")
    if isinstance(fast, Exception) or dev == "cpu":
        report("causal_conv1d_fn", "SKIP" if dev == "cpu" else "FAIL",
               "needs a CUDA GPU" if dev == "cpu" else "causal-conv1d is not importable")
        return
    g = torch.Generator(device="cpu").manual_seed(1)
    B, C, T, W = 2, 1024, 1000, 4                        # conv channels: 2 x 2 key heads x 128 + 4 value heads x 128
    x0, w0, b0, p0 = torch.randn(B, T, C, generator=g), torch.randn(C, W, generator=g), None, torch.randn(B, C, T, generator=g)

    def run(fn, dtype):
        x = x0.to(dev, dtype).transpose(1, 2).detach().requires_grad_(True)      # channels-last, as in the model
        w = w0.to(dev, dtype).detach().requires_grad_(True)
        out = fn(x, w, b0, activation="silu")
        (out.float() * p0.to(dev)).sum().backward()
        return out.detach(), [x.grad, w.grad]

    ref_out, ref_grads = run(reference, torch.float32)
    for dtype in dtypes:
        name = f"causal_conv1d_fn {str(dtype).split('.')[-1]}"
        try:
            out, grads = run(fast, dtype)
            torch.cuda.synchronize()
        except Exception as e:
            report(name, "FAIL", f"kernel error: {type(e).__name__}: {str(e).splitlines()[0][:150]}")
            continue
        tol_out, tol_grad = TOLERANCE[dtype]
        e_out, e_grad = rel_err(out, ref_out), max(rel_err(a, b) for a, b in zip(grads, ref_grads))
        report(name, grade(e_out, e_grad, dtype),
               f"relative error vs PyTorch reference: output {e_out:.4f} (< {tol_out}), worst gradient {e_grad:.4f} (< {tol_grad})")


# ---------------------------------------------------------------- 3. optimizer rounding

def check_optimizer(dev: str) -> None:
    """Weights at 1.0 in bf16 (spacing 2^-8 just below 1.0). AdamW moves each weight by ~lr per step whatever the
    gradient, so 200 steps of lr 1e-5 should move them by ~2e-3 in total, while each single step is ~400x smaller
    than the bf16 spacing: round-to-nearest drops every step, stochastic rounding keeps them on average."""
    AdamW8bit = load("torchao.optim", "AdamW8bit")
    if isinstance(AdamW8bit, Exception):
        report("8-bit AdamW + bf16 stochastic rounding", "FAIL", f"torchao.optim.AdamW8bit not importable: {AdamW8bit}")
        return
    steps, lr = 200, 1e-5
    expected = steps * lr

    def train(make_opt):
        p = torch.nn.Parameter(torch.ones(1024, 1024, device=dev, dtype=torch.bfloat16))
        opt = make_opt([p])
        for _ in range(steps):
            p.grad = torch.full_like(p, 0.01)
            opt.step()
        return float(1.0 - p.detach().float().mean())

    plain = train(lambda ps: torch.optim.AdamW(ps, lr=lr, weight_decay=0.0))
    report("plain AdamW on bf16 weights (the trap)", "INFO",
           f"mean weight change {plain:.2e} (expected {expected:.0e} if updates were kept; ~0 = updates rounded away)")
    try:
        sr = train(lambda ps: AdamW8bit(ps, lr=lr, weight_decay=0.0, bf16_stochastic_round=True))
    except Exception as e:
        report("8-bit AdamW + bf16 stochastic rounding", "FAIL", f"{type(e).__name__}: {str(e).splitlines()[0][:150]}")
        return
    ok = 0.6 * expected < sr < 1.4 * expected
    report("8-bit AdamW + bf16 stochastic rounding", "PASS" if ok else "FAIL",
           f"mean weight change {sr:.2e}, expected ~{expected:.0e}")


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dtypes", nargs="+", default=["bfloat16", "float16", "float32"])
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtypes = [getattr(torch, d) for d in args.dtypes]
    for name, fn in [("environment", lambda: check_environment(dev)), ("gated delta rule", lambda: check_gated_delta(dev, dtypes)),
                     ("causal conv1d", lambda: check_conv(dev, dtypes)), ("optimizer", lambda: check_optimizer(dev))]:
        print(f"\n---- {name}", flush=True)
        try:
            fn()
        except Exception as e:
            traceback.print_exc()
            report(name, "FAIL", f"crashed: {type(e).__name__}: {e}")

    print("\n" + "=" * 100 + "\nGPU CHECK SUMMARY\n" + "=" * 100)
    for check, status, detail in results:
        if status != "INFO":
            print(f"[{status}] {check}" + (f"  ({detail})" if status == "WARN" else ""))
    native_bf16 = dev == "cuda" and torch.cuda.get_device_properties(0).major >= 8
    # On a T4 a bf16-only failure is the GPU's limit, not the setup's: judge the kernels on fp16 / fp32 there.
    needed = [r for r in results if r[1] == "FAIL" and (native_bf16 or "bfloat16" not in r[0])]
    print("=" * 100)
    print("RESULT: " + ("all needed checks passed" if not needed else f"{len(needed)} needed check(s) failed"))
    sys.exit(1 if needed else 0)


if __name__ == "__main__":
    main()
