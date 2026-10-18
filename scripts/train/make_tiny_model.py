"""Tiny random Qwen3.5 for dry runs of train_cpt.py off the GPU meter (docs/prepare_cpt_training_pipeline.md, section 9).

Same tokenizer, same layer mix (3 Gated DeltaNet + 1 full attention), same tied embeddings as Qwen3.5-4B-Base, but
hidden size 64 and 4 layers (~16M parameters, almost all of them the 248k-token embedding). Downloads only
config.json and the tokenizer.

    python scripts/train/make_tiny_model.py                     # -> outputs/tiny-qwen3.5
    python scripts/train/train_cpt.py --config scripts/train/config/debug_tiny.json

--real-heads keeps the 4B model's head sizes (DeltaNet key/value 128, attention 256, same rotary split), so the GPU
kernels see real shapes (~66M parameters; used by colab_gpu_check.ipynb):

    python scripts/train/make_tiny_model.py --real-heads --out outputs/tiny-qwen3.5-gpu
"""
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_eval import ROOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--out", default="outputs/tiny-qwen3.5")
    ap.add_argument("--real-heads", action="store_true", help="keep the real head sizes (GPU kernel test)")
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.base)
    cfg = getattr(cfg, "text_config", cfg)              # Qwen3.5 checkpoints wrap the text config
    cfg.num_hidden_layers = 4
    cfg.layer_types = ["linear_attention"] * 3 + ["full_attention"]
    cfg.num_attention_heads = 2
    cfg.num_key_value_heads = 1
    cfg.linear_num_key_heads = 2
    cfg.linear_num_value_heads = 4
    if args.real_heads:                                  # head_dim, linear_*_head_dim and mrope_section stay
        cfg.hidden_size = 256
        cfg.intermediate_size = 512
    else:
        cfg.hidden_size = 64
        cfg.intermediate_size = 128
        cfg.head_dim = 64                                # partial rotary 0.25 -> 16 dims -> mrope sections sum to 8
        cfg.rope_parameters = {**cfg.rope_parameters, "mrope_section": [3, 3, 2]}
        cfg.linear_key_head_dim = 16
        cfg.linear_value_head_dim = 16
    cfg.mtp_num_hidden_layers = 0
    cfg.tie_word_embeddings = True

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    out = ROOT / args.out
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(out)
    n = sum(p.numel() for p in model.parameters())
    tied = model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    print(f"{type(model).__name__}: {n / 1e6:.1f}M parameters, tied: {tied} -> {out}")


if __name__ == "__main__":
    main()
