"""Run verify_weights.compare on whatever tensor pairs are already in tensor_cache (no network)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_weights import CACHE, compare  # noqa: E402

def load(p, cols):
    bits = np.frombuffer(p.read_bytes(), dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32).reshape(-1, cols)


ft_dir = CACHE / "sinhala-mmlu__gemma-4-e4b-sinhala-mmlu"
base_dir = CACHE / "google__gemma-4-E4B-it"
for p in sorted(ft_dir.glob("*.bin")):
    q = base_dir / p.name
    if not q.exists() or q.stat().st_size != p.stat().st_size:
        continue
    rows = int(p.name.rsplit(".", 2)[1])
    cols = p.stat().st_size // 2 // rows
    name = p.name.rsplit(".", 2)[0]
    r = compare(name, load(p, cols), load(q, cols), None)
    r.pop("sv", None)
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}))
