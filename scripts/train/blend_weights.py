"""Weighted average of models with the same architecture: the "free fix" of docs/prepare_cpt_training_pipeline.md,
section 9 (average the last checkpoints, or blend CPT with the base model if general skills dropped). No training.

    python scripts/train/blend_weights.py --models outputs/cpt-easy-v1/epochs/e3 Qwen/Qwen3.5-4B-Base \
        --weights 0.8 0.2 --out outputs/blend-e3-0.8
    python scripts/train/probe_eval.py --model outputs/blend-e3-0.8 --out outputs/probes/blend-e3-0.8

Choose the weights by validation loss / probes. The first model's config, tokenizer and tied embeddings are kept.
Needs ~8.4 GB of RAM per model (bf16), plus one fp32 tensor at a time.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_eval import ROOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="local folders or HF repo ids (same architecture)")
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if len(args.models) != len(args.weights) or abs(sum(args.weights) - 1) > 1e-6:
        raise SystemExit("give one weight per model; the weights must sum to 1")

    models = [AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16) for m in args.models]
    params = [dict(m.named_parameters()) for m in models]       # tied weights appear once
    names = set(params[0])
    for m, p in zip(args.models[1:], params[1:]):
        if set(p) != names:
            raise SystemExit(f"{m} has different parameter names than {args.models[0]}")
    with torch.no_grad():
        for name, target in params[0].items():
            acc = sum(w * p[name].float() for w, p in zip(args.weights, params))
            target.copy_(acc.to(target.dtype))

    out = ROOT / args.out
    models[0].config.use_cache = True
    models[0].save_pretrained(out)
    AutoTokenizer.from_pretrained(args.models[0]).save_pretrained(out)
    (out / "blend_info.json").write_text(json.dumps({"models": args.models, "weights": args.weights}, indent=2) + "\n",
                                         encoding="utf-8")
    print(f"blend of {list(zip(args.models, args.weights))} -> {out}")


if __name__ == "__main__":
    main()
