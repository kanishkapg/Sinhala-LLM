"""Check the model card's training claims against the real weights, without downloading 16 GB.

Fetches single tensors (or a row slice of one) from sinhala-mmlu/gemma-4-e4b-sinhala-mmlu and from its base
google/gemma-4-E4B-it with HTTP range requests, then compares them:
  - LoRA-trained layers should differ from the base, with a low-rank difference (rank <= 64 if LoRA r=64);
  - embeddings, norms and the vision/audio towers should be byte-identical (LoRA never touches them).
Usage: python -I verify_weights.py   (writes weight_check.json next to this script)
"""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

FT = ("sinhala-mmlu/gemma-4-e4b-sinhala-mmlu", "v1.0.0")
BASE = ("google/gemma-4-E4B-it", "ee0ef6023621cff504d758262d4e04895a5af4a2")
CACHE = Path(__file__).resolve().parent.parent / "tensor_cache"
LM = "model.language_model."

# (tensor name, max rows to fetch or None for all)
TENSORS = [
    (LM + "layers.0.self_attn.q_proj.weight", None),
    (LM + "layers.10.self_attn.q_proj.weight", None),
    (LM + "layers.10.self_attn.k_proj.weight", None),
    (LM + "layers.10.self_attn.v_proj.weight", None),
    (LM + "layers.10.self_attn.o_proj.weight", None),
    (LM + "layers.10.mlp.gate_proj.weight", 1024),
    (LM + "layers.10.mlp.down_proj.weight", 256),
    (LM + "layers.10.per_layer_input_gate.weight", None),
    (LM + "layers.10.per_layer_projection.weight", None),
    (LM + "layers.35.self_attn.q_proj.weight", 512),
    (LM + "layers.41.self_attn.o_proj.weight", 512),
    (LM + "per_layer_model_projection.weight", 256),
    (LM + "layers.10.input_layernorm.weight", None),
    (LM + "layers.10.layer_scalar", None),
    (LM + "norm.weight", None),
    (LM + "embed_tokens.weight", 256),
    (LM + "embed_tokens_per_layer.weight", 64),
    ("model.vision_tower.patch_embedder.input_proj.weight", None),
    ("model.audio_tower.output_proj.weight", None),
]


def get(url, rng=None, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "weight-check/1.0"})
            if rng:
                req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - flaky connection: retry
            if k == tries - 1:
                raise
            print(f"  retry {k + 1} ({e})", flush=True)
            time.sleep(3 * (k + 1))


def url(repo, name):
    return f"https://huggingface.co/{repo[0]}/resolve/{repo[1]}/{name}"


class Repo:
    def __init__(self, repo):
        self.repo = repo
        self.headers = {}
        try:
            self.map = json.loads(get(url(repo, "model.safetensors.index.json"), tries=1))["weight_map"]
        except urllib.error.HTTPError:  # single-file checkpoint: the header lists every tensor
            self.map = {k: "model.safetensors" for k in self.header("model.safetensors")[1] if k != "__metadata__"}

    def header(self, shard):
        if shard not in self.headers:
            n = int.from_bytes(get(url(self.repo, shard), (0, 7)), "little")
            self.headers[shard] = (8 + n, json.loads(get(url(self.repo, shard), (8, 8 + n - 1))))
        return self.headers[shard]

    def tensor(self, name, max_rows=None):
        if name not in self.map:
            return None, None
        shard = self.map[name]
        start, hdr = self.header(shard)
        meta = hdr[name]
        assert meta["dtype"] == "BF16", meta["dtype"]
        shape = meta["shape"]
        a, b = meta["data_offsets"]
        if max_rows and len(shape) == 2 and shape[0] > max_rows:
            b = a + max_rows * shape[1] * 2
            shape = [max_rows, shape[1]]
        cache = CACHE / self.repo[0].replace("/", "__") / f"{name}.{shape[0]}.bin"
        if cache.exists():
            raw = cache.read_bytes()
        else:
            raw = get(url(self.repo, shard), (start + a, start + b - 1))
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(raw)
        bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32).reshape(shape), meta["shape"]


def compare(name, ft, base, full_shape):
    out = {"tensor": name.replace(LM, "LM."), "full_shape": full_shape, "compared_shape": list(ft.shape)}
    d = ft.astype(np.float64) - base.astype(np.float64)
    out["byte_identical"] = bool(np.array_equal(ft.view(np.uint32), base.view(np.uint32)))
    out["frac_elements_changed"] = float((d != 0).mean())
    out["rel_change_fro"] = float(np.linalg.norm(d) / max(np.linalg.norm(base), 1e-30))
    if ft.ndim == 2 and min(ft.shape) > 128 and not out["byte_identical"]:
        s = np.linalg.svd(d, compute_uv=False)
        energy = np.cumsum(s ** 2) / np.sum(s ** 2)
        out["rank_for_99pct_energy"] = int(np.searchsorted(energy, 0.99) + 1)
        out["energy_in_top64"] = float(energy[63])
        out["sv_ratio_64_to_65"] = float(s[63] / s[64])
        out["sv_ratio_63_to_64"] = float(s[62] / s[63])
        out["sv_ratio_65_to_66"] = float(s[64] / s[65])
        out["sv"] = {i: float(s[i]) for i in (0, 31, 62, 63, 64, 65, 127)}
        # bf16 rounding of the merged weights adds full-rank noise. Treat directions 65+ as noise only and
        # subtract the same per-direction noise energy from the top 64: what is left is the low-rank signal.
        e = s ** 2
        noise_per_dir = e[64:].mean()
        signal = max(0.0, float(e[:64].sum() - 64 * noise_per_dir))
        out["signal_energy_frac"] = signal / float(e.sum())
        out["signal_rel_change"] = float(np.sqrt(signal) / max(np.linalg.norm(base), 1e-30))
        out["n_sv_above_noise_edge"] = int((s > 1.05 * s[64]).sum())
    return out


def main():
    ft_repo, base_repo = Repo(FT), Repo(BASE)
    only_ft = sorted(set(ft_repo.map) - set(base_repo.map))
    only_base = sorted(set(base_repo.map) - set(ft_repo.map))
    report = {"ft_tensors": len(ft_repo.map), "base_tensors": len(base_repo.map),
              "only_in_ft": only_ft[:20], "only_in_base_count": len(only_base), "only_in_base_sample": only_base[:20],
              "results": []}
    for name, rows in TENSORS:
        print("fetching", name, flush=True)
        ft, shape = ft_repo.tensor(name, rows)
        base, _ = base_repo.tensor(name, rows)
        if ft is None or base is None:
            report["results"].append({"tensor": name, "missing_in": "ft" if ft is None else "base"})
            continue
        r = compare(name, ft, base, shape)
        print("  ", {k: v for k, v in r.items() if k != "sv"}, flush=True)
        report["results"].append(r)
    path = Path(__file__).with_name("weight_check.json")
    path.write_text(json.dumps(report, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
