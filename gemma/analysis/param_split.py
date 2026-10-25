"""Count parameters per component (language model / vision / audio) from the safetensors headers only."""
import sys
from collections import defaultdict
from math import prod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_weights import BASE, FT, Repo  # noqa: E402


def split(repo):
    r = Repo(repo)
    groups = defaultdict(int)
    for shard in sorted(set(r.map.values())):
        for name, meta in r.header(shard)[1].items():
            if name == "__metadata__":
                continue
            if "embed_tokens_per_layer" in name:
                g = "LM per-layer embeddings (lookup table)"
            elif "embed_tokens" in name:
                g = "LM token embeddings (tied with output head)"
            elif "language_model" in name:
                g = "LM transformer layers + projections"
            elif "vision" in name:
                g = "vision tower + projector"
            elif "audio" in name:
                g = "audio tower + projector"
            else:
                g = "other: " + name
            groups[g] += prod(meta["shape"])
    return groups


for label, repo in (("fine-tuned", FT), ("base", BASE)):
    g = split(repo)
    total = sum(g.values())
    print(f"== {label}: {total:,} ({total / 1e9:.3f} B)")
    for k, v in sorted(g.items(), key=lambda kv: -kv[1]):
        print(f"   {v / 1e9:7.3f} B  {k}")
    text = total - sum(v for k, v in g.items() if k.startswith(("vision", "audio")))
    print(f"   text-only (no vision/audio): {text / 1e9:.3f} B -> headroom under 8B: {(8e9 - text) / 1e9:.3f} B")
