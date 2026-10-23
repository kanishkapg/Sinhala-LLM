"""Step 8 of docs/prepare_sft_dataset.md: near-duplicates inside the accepted set, and decontamination against the
shared-task test (all levels), Probe A (full + excluded), Probe B and the style exemplars.

    python scripts/sft/s06_dedup.py --level easy

Uses the same matching as the probes (probe_common.near_duplicates: similar question + identical options) plus a
MinHash-LSH pass for near-identical question text. Writes 05_final/items.jsonl, dedup_dropped.jsonl and
dedup_report.json (counts for the system report + flagged pairs to look at).
"""
import argparse
import difflib
from collections import Counter

from datasketch import MinHash, MinHashLSH

from sft_common import (CONFIG_DIR, SHARED_TEST_GLOB, Paths, ensure_dirs, level_arg, load_config,
                        load_json_items, near_duplicates, norm, read_json, read_jsonl, stdout_utf8,
                        update_manifest, write_json, write_jsonl)


def shingles(text: str, k: int) -> set[str]:
    t = norm(text)
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def minhash(text: str, k: int) -> MinHash:
    m = MinHash(num_perm=64, seed=42)
    for s in shingles(text, k):
        m.update(s.encode("utf-8"))
    return m


def ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def stem_matches(items: list[dict], refs: list[dict], d: dict, self_match: bool = False):
    """(i, j, stem ratio) for pairs whose question texts are at least `flag_stem_ratio` similar."""
    lsh = MinHashLSH(threshold=d["lsh_threshold"], num_perm=64)
    for j, r in enumerate(refs):
        lsh.insert(str(j), minhash(r["question"], d["shingle"]))
    out = []
    for i, it in enumerate(items):
        for key in lsh.query(minhash(it["question"], d["shingle"])):
            j = int(key)
            if self_match and j <= i:
                continue
            q = ratio(it["question"], refs[j]["question"])
            if q >= d["flag_stem_ratio"]:
                out.append((i, j, round(q, 3)))
    return out


def main():
    ap = argparse.ArgumentParser(description="Deduplicate and decontaminate the accepted items")
    level_arg(ap)
    args = ap.parse_args()
    stdout_utf8()
    paths, cfg = Paths(args.level), load_config(args.level)
    ensure_dirs(paths)
    d = cfg["dedup"]
    items = list(read_jsonl(paths.validated / "accepted.jsonl"))
    ex = read_json(CONFIG_DIR / f"exemplars_{args.level}.json")["exemplars"]
    refs = {
        "shared_test": load_json_items(SHARED_TEST_GLOB),
        "probe_a": [r for f in ["probe_a_full.jsonl", "probe_a_excluded.jsonl"]
                    for r in read_jsonl(paths.probes / f) if r.get("question") and r.get("choices")],
        "probe_b": list(read_jsonl(paths.probes / "probe_b.jsonl")),
        "exemplars": [{"question": e["question"], "choices": [e["correct"], *e["distractors"]] if e["correct"] else []}
                      for e in ex],
    }
    drop, flags, counts = {}, [], Counter()

    # 1. inside the set (keep the first by id)
    for i, j, q, n in near_duplicates(items, items, d["drop_stem_ratio_with_options"], d["drop_min_shared_options"]):
        if i < j:
            drop.setdefault(items[j]["id"], f"duplicate_of:{items[i]['id']}")
    for i, j, q in stem_matches(items, items, d, self_match=True):
        if q >= d["drop_stem_ratio"]:
            drop.setdefault(items[j]["id"], f"duplicate_of:{items[i]['id']}")
    counts["self_duplicates"] = len(drop)

    # 2. against the test, the probes and the exemplars
    for name, ref in refs.items():
        before = len(drop)
        for i, j, q, n in near_duplicates(items, ref, d["drop_stem_ratio_with_options"], d["drop_min_shared_options"]):
            drop.setdefault(items[i]["id"], f"near_copy_of:{name}")
        for i, j, q in stem_matches(items, ref, d):
            if q >= d["drop_stem_ratio"]:
                drop.setdefault(items[i]["id"], f"near_copy_of:{name}")
            elif items[i]["id"] not in drop:
                flags.append({"item_id": items[i]["id"], "against": name, "stem_ratio": q,
                              "item": items[i]["question"], "ref": ref[j]["question"]})
        for i, j, q, n in near_duplicates(items, ref, 0.5, d["flag_min_shared_options"]):
            if items[i]["id"] not in drop and items[i]["metadata"]["qtype"] != "combination":
                flags.append({"item_id": items[i]["id"], "against": name, "stem_ratio": q, "shared_options": n,
                              "item": items[i]["question"], "ref": ref[j]["question"]})
        counts[f"dropped_vs_{name}"] = len(drop) - before

    # 3. Probe B evidence reused by items from Probe B source sections (flag only; the pilot leaves them out)
    pb_ev = [norm(r["metadata"].get("evidence", "")) for r in refs["probe_b"] if r["metadata"].get("evidence")]
    for it in items:
        if it["metadata"]["probe_b_source"] and it["id"] not in drop:
            ev = norm(it["metadata"].get("evidence", ""))
            if ev and any(ev in p or p in ev or difflib.SequenceMatcher(None, ev, p).ratio() >= 0.8 for p in pb_ev):
                flags.append({"item_id": it["id"], "against": "probe_b_evidence", "item": it["question"]})

    kept = [it for it in items if it["id"] not in drop]
    dropped = [{**it, "dedup_reason": drop[it["id"]]} for it in items if it["id"] in drop]
    write_jsonl(paths.final / "items.jsonl", kept)
    write_jsonl(paths.final / "dedup_dropped.jsonl", dropped)
    seen = set()
    flags = [f for f in flags if not ((f["item_id"], f["against"]) in seen or seen.add((f["item_id"], f["against"])))]
    report = {"accepted_in": len(items), "kept": len(kept), "dropped": len(dropped), "counts": dict(counts),
              "reference_sizes": {k: len(v) for k, v in refs.items()}, "flagged": flags}
    write_json(paths.final / "dedup_report.json", report)
    update_manifest(paths, "dedup", {k: v for k, v in report.items() if k != "flagged"} | {"flagged": len(flags)})
    print(f"accepted {len(items)} -> kept {len(kept)}, dropped {len(dropped)}: {dict(counts)}")
    print(f"flagged for a look: {len(flags)} (05_final/dedup_report.json -> 'flagged'); "
          f"by reference: {dict(Counter(f['against'] for f in flags))}")
    print(f"-> {paths.final}")


if __name__ == "__main__":
    main()
