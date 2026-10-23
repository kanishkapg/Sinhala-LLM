"""Step 11 of docs/prepare_sft_dataset.md: split, re-balance positions, render the official prompt, tokenise.

    python scripts/sft/s09_render.py --level easy --variant pilot     # wave 1, without Probe B source sections
    python scripts/sft/s09_render.py --level easy --variant final     # every wave

Writes 06_rendered/<variant>/{train,val}.jsonl (prompt + digit, readable) and {train,val}.parquet
(input_ids, labels: -100 on the prompt, the digit token as the only target). Refuses to run unless the audit
(s07) passed for the waves used (override: --no-gate).
"""
import argparse
import math
from collections import Counter, defaultdict

from sft_common import (SEED, Paths, build_prompt, ensure_dirs, file_sha1, level_arg, load_config, read_json,
                        read_jsonl, read_jsonl_tolerant, seeded_rng, stable_int, stdout_utf8,
                        subject_string_variants, update_manifest, write_jsonl)

VARIANTS = {"pilot": {"waves": {1}, "probe_b_source": False}, "final": {"waves": {0, 1, 2, 3}, "probe_b_source": True}}


def place(choices: list[str], answer: int, target: int, rng=None) -> tuple[list[str], int]:
    key = choices[answer - 1]
    ds = [c for i, c in enumerate(choices) if i != answer - 1]
    if rng:
        rng.shuffle(ds)
    return ds[: target - 1] + [key] + ds[target - 1:], target


def main():
    ap = argparse.ArgumentParser(description="Render the train-ready SFT files")
    level_arg(ap)
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--no-gate", action="store_true", help="render even if the audit did not pass")
    args = ap.parse_args()
    stdout_utf8()
    paths, cfg = Paths(args.level), load_config(args.level)
    ensure_dirs(paths)
    var = VARIANTS[args.variant]
    rc = cfg["render"]

    audit = read_json(paths.final / "audit.json")
    if not args.no_gate:
        if not audit or not audit.get("passed"):
            raise SystemExit("the audit (s07_audit.py) has not passed: fix it first, or use --no-gate knowingly")
        if args.variant == "pilot" and 1 not in audit["waves"]:
            raise SystemExit("the last audit did not include wave 1: run s07_audit.py --waves 1")

    drops = {r["id"] for f in [paths.final / "audit_drops.jsonl", paths.review / "review_drops.jsonl"]
             for r in read_jsonl_tolerant(f)}
    items = [it for it in read_jsonl(paths.final / "items.jsonl")
             if it["id"] not in drops and it["metadata"]["wave"] in var["waves"]
             and (var["probe_b_source"] or not it["metadata"]["probe_b_source"])]
    if not items:
        raise SystemExit("no items for this variant")

    # validation split: 5% of sections per subject, fixed by a seeded hash (identical for pilot and final)
    by_subj = defaultdict(list)
    for s in read_jsonl(paths.plan / "sections.jsonl"):
        by_subj[s["subject"]].append(s["id"])
    val_sections = set()
    for subj, ids in by_subj.items():
        ids.sort(key=lambda i: stable_int(SEED, "val", i))
        val_sections |= set(ids[: math.ceil(rc["val_share"] * len(ids))])
    for it in items:
        it["_split"] = "val" if it["metadata"]["source_id"] in val_sections else "train"

    # exact position balance per (split, subject, qtype); combination items keep their natural order
    cells = defaultdict(list)
    for it in items:
        if it["metadata"]["qtype"] != "combination":
            cells[(it["_split"], it["subject"], it["metadata"]["qtype"])].append(it)
    for cell, its in cells.items():
        its.sort(key=lambda x: stable_int(SEED, "render-pos", x["id"]))
        off = stable_int(SEED, "render-off", *cell) % 4
        for k, it in enumerate(its):
            it["choices"], it["answer"] = place(it["choices"], it["answer"], (off + k) % 4 + 1)

    variants = subject_string_variants()
    from common import tokenizer
    tok = tokenizer()
    digit = {d: tok.encode(str(d), add_special_tokens=False).ids for d in range(1, 5)}
    assert all(len(v) == 1 for v in digit.values()), digit
    digit = {d: v[0] for d, v in digit.items()}
    assert list(digit.values()) == [16, 17, 18, 19], f"unexpected digit ids {digit}"

    rows = {"train": [], "val": []}
    for it in items:
        subj = it["subject"]
        vs = variants.get(subj) or [it["metadata"]["subject_original"]]
        so = vs[stable_int(SEED, "so", it["id"]) % len(vs)]
        views = [(0, it["choices"], it["answer"])]
        if it["_split"] == "train" and rc["views"] >= 2 and it["metadata"]["qtype"] != "combination":
            ch, a = place(it["choices"], it["answer"], (it["answer"] + 1) % 4 + 1, seeded_rng(SEED, "view1", it["id"]))
            views.append((1, ch, a))
        for v, ch, a in views:
            prompt = build_prompt({"metadata": {"subject_original": so}, "question": it["question"], "choices": ch})
            rows[it["_split"]].append({"id": f"{it['id']}-v{v}", "item_id": it["id"], "view": v, "split": it["_split"],
                                       "subject": subj, "qtype": it["metadata"]["qtype"],
                                       "probe_b_source": it["metadata"]["probe_b_source"], "subject_original": so,
                                       "prompt": prompt, "completion": str(a), "answer": a})

    out_dir = paths.rendered / args.variant
    too_long = 0
    import pyarrow as pa
    import pyarrow.parquet as pq
    summary = {}
    for split, rs in rows.items():
        enc = tok.encode_batch([r["prompt"] for r in rs], add_special_tokens=False)
        chk = tok.encode_batch([r["prompt"] + r["completion"] for r in rs], add_special_tokens=False)
        kept = []
        for r, e, c in zip(rs, enc, chk):
            d = digit[r["answer"]]
            if c.ids != e.ids + [d]:
                raise SystemExit(f"{r['id']}: prompt + digit does not tokenize as prompt tokens + the digit token")
            if len(e.ids) + 1 > rc["max_tokens"]:
                too_long += 1
                continue
            r["input_ids"] = e.ids + [d]
            r["labels"] = [-100] * len(e.ids) + [d]
            r["n_tokens"] = len(r["input_ids"])
            kept.append(r)
        kept.sort(key=lambda r: stable_int(SEED, "order", split, r["id"]))      # shuffled, reproducible
        write_jsonl(out_dir / f"{split}.jsonl", [{k: v for k, v in r.items() if k not in ("input_ids", "labels")}
                                                 for r in kept])
        cols = {k: [r[k] for r in kept] for k in kept[0]} if kept else {}
        table = pa.table({**{k: v for k, v in cols.items() if k not in ("input_ids", "labels")},
                          "input_ids": pa.array(cols.get("input_ids", []), type=pa.list_(pa.int32())),
                          "labels": pa.array(cols.get("labels", []), type=pa.list_(pa.int32()))})
        pq.write_table(table, out_dir / f"{split}.parquet")
        lab = Counter(r["answer"] for r in kept)
        ntok = sorted(r["n_tokens"] for r in kept)
        summary[split] = {"examples": len(kept), "items": len({r["item_id"] for r in kept}),
                          "labels": dict(sorted(lab.items())),
                          "tokens_total": sum(ntok), "tokens_median": ntok[len(ntok) // 2] if ntok else 0,
                          "tokens_max": ntok[-1] if ntok else 0,
                          "jsonl_sha1": file_sha1(out_dir / f"{split}.jsonl"),
                          "parquet_sha1": file_sha1(out_dir / f"{split}.parquet")}
        print(f"{split}: {len(kept)} examples from {summary[split]['items']} items; labels {summary[split]['labels']}; "
              f"tokens median {summary[split]['tokens_median']}, max {summary[split]['tokens_max']}, "
              f"total {summary[split]['tokens_total']}")
        per_subj = defaultdict(Counter)
        for r in kept:
            per_subj[r["subject"]][r["answer"]] += 1
        worst = max((max(c.values()) - min(c.get(d, 0) for d in range(1, 5)) for c in per_subj.values()), default=0)
        print(f"  worst per-subject label imbalance (max - min count): {worst}")
    if too_long:
        print(f"dropped {too_long} examples longer than {rc['max_tokens']} tokens")
    so_use = Counter((r["subject"], r["subject_original"]) for r in rows["train"] if r["subject"] in ("Art", "Health"))
    print("subject-string variants used (Art/Health):", dict(so_use))
    update_manifest(paths, f"render_{args.variant}", {"waves": sorted(var["waves"]),
                                                      "probe_b_source_included": var["probe_b_source"],
                                                      "views": rc["views"], "val_share": rc["val_share"],
                                                      "dropped_too_long": too_long, "excluded_drops": len(drops),
                                                      **summary})
    ex = rows["train"][0]
    print("\nexample prompt (the target is the digit after it):\n" +
          "\n".join("  | " + line for line in ex["prompt"].split("\n")) + f"\n  -> {ex['completion']}")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
