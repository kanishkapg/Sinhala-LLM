"""Step 9 of docs/prepare_sft_dataset.md: dataset-level bias audit with gates.

    python scripts/sft/s07_audit.py --level easy [--waves 1]            # metrics + gates -> 05_final/audit.json
    python scripts/sft/s07_audit.py --level easy --fix-length           # drop worst "key much longer" items
    python scripts/sft/s07_audit.py --level easy --export-choices-only 1000
    python scripts/sft/s07_audit.py --level easy --choices-only-summary outputs/probes/choices_only/summary.json

Exit code 1 when a gate fails (s09 refuses to render until the audit passes, unless --no-gate there).
Answer positions are re-balanced exactly at render time (s09), so here they are only reported.
"""
import argparse
import difflib
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sft_common import (QTYPES, Paths, append_jsonl, ensure_dirs, level_arg, load_config, norm, read_json, read_jsonl,
                        read_jsonl_tolerant, seeded_rng, stdout_utf8, update_manifest, words, write_json,
                        write_jsonl)


def unique_arg(vals: list[float], best=max) -> int | None:
    b = best(vals)
    return vals.index(b) if vals.count(b) == 1 else None


def cue_flags(it: dict) -> dict:
    opts, k = it["choices"], it["answer"] - 1
    lens = [len(o) for o in opts]
    sw = words(it["question"])
    overlap = [len(words(o) & sw) for o in opts]
    conv = [sum(difflib.SequenceMatcher(None, norm(o), norm(p)).ratio() for p in opts if p is not o) / 3 for o in opts]
    return {"longest": unique_arg(lens) == k, "shortest": unique_arg(lens, min) == k,
            "has_unique_longest": unique_arg(lens) is not None, "has_unique_shortest": unique_arg(lens, min) is not None,
            "overlap": unique_arg(overlap) == k and max(overlap) > 0, "convergence": unique_arg(conv) == k}


def rate(rows: list[dict], flag: str, denom: str | None = None) -> float:
    d = [r for r in rows if r[denom]] if denom else rows
    return sum(r[flag] for r in d) / len(d) if d else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Dataset-level bias audit")
    level_arg(ap)
    ap.add_argument("--waves", type=int, nargs="*", default=None, help="audit only these waves (default: all)")
    ap.add_argument("--fix-length", action="store_true", help="drop the worst long-key items until the gate passes")
    ap.add_argument("--export-choices-only", type=int, default=0, metavar="N")
    ap.add_argument("--choices-only-summary", default=None, help="summary.json of probe_eval on the export")
    args = ap.parse_args()
    stdout_utf8()
    paths, cfg = Paths(args.level), load_config(args.level)
    ensure_dirs(paths)
    g = cfg["gates"]
    drops_file = paths.final / "audit_drops.jsonl"
    dropped_ids = {r["id"] for r in read_jsonl_tolerant(drops_file)}
    items = [it for it in read_jsonl(paths.final / "items.jsonl") if it["id"] not in dropped_ids
             and (args.waves is None or it["metadata"]["wave"] in args.waves)]
    if not items:
        raise SystemExit("no items to audit")

    if args.export_choices_only:
        out = paths.final / "choices_only"
        rng = seeded_rng(42, "choices_only")
        pool = [it for it in items if it["metadata"]["qtype"] != "combination"]
        sample = rng.sample(pool, min(args.export_choices_only, len(pool)))
        b = [{"id": it["id"], "subject": it["subject"], "split": "train", "question": "", "choices": it["choices"],
              "answer": it["answer"], "metadata": {"subject_original": it["metadata"]["subject_original"]}} for it in sample]
        a = [{**r, "question": ""} for r in read_jsonl(paths.probes / "probe_a_full.jsonl")]
        write_jsonl(out / "probe_b.jsonl", b)
        write_jsonl(out / "probe_a_full.jsonl", a)
        write_jsonl(out / "probe_a_subset.jsonl", a)
        print(f"choices-only probe files -> {out} (SFT sample {len(b)} as 'probe_b', Probe A {len(a)} as 'probe_a')")
        print("score them with scripts/train/probe_eval.py (see scripts/sft/README.md, step 9b)")
        return

    plain = [it for it in items if it["metadata"]["qtype"] != "combination"]
    flags = {it["id"]: cue_flags(it) for it in plain}

    if args.fix_length:
        new_drops = []
        for subj in sorted({it["subject"] for it in plain}):
            sp = [it for it in plain if it["subject"] == subj and it["id"] not in dropped_ids]

            def ratio_next(it):
                ls = sorted((len(o) for o in it["choices"]), reverse=True)
                return ls[0] / max(ls[1], 1)
            cands = sorted([it for it in sp if flags[it["id"]]["longest"] and ratio_next(it) >= g["length_fix_ratio"]],
                           key=ratio_next, reverse=True)
            while cands and rate([flags[i["id"]] for i in sp], "longest", "has_unique_longest") > g["key_longest_max"]:
                it = cands.pop(0)
                sp.remove(it)
                new_drops.append({"id": it["id"], "reason": "key_longest_fix"})
        for r in new_drops:
            append_jsonl(drops_file, r)
        print(f"--fix-length: dropped {len(new_drops)} items (-> 05_final/audit_drops.jsonl); re-run the audit")
        return

    quotas = read_json(paths.plan / "quotas.json")
    jobs = list(read_jsonl(paths.plan / "jobs.jsonl"))
    res, gates = {}, {}
    F = list(flags.values())
    res["n_items"] = len(items)
    res["positions"] = dict(sorted(Counter(it["answer"] for it in items).items()))
    res["key_longest"] = rate(F, "longest", "has_unique_longest")
    res["key_shortest"] = rate(F, "shortest", "has_unique_shortest")
    res["stem_overlap"] = rate(F, "overlap")
    res["convergence"] = rate(F, "convergence")
    per_subj = defaultdict(list)
    for it in plain:
        per_subj[it["subject"]].append(flags[it["id"]])
    res["key_longest_per_subject"] = {s: round(rate(v, "longest", "has_unique_longest"), 3) for s, v in per_subj.items()}
    gates["key_longest"] = res["key_longest"] <= g["key_longest_max"] and all(
        v <= g["key_longest_max"] for s, v in res["key_longest_per_subject"].items() if len(per_subj[s]) >= 30)
    gates["key_shortest"] = res["key_shortest"] >= g["key_shortest_min"]
    gates["stem_overlap"] = res["stem_overlap"] <= g["stem_overlap_max"]
    gates["convergence"] = res["convergence"] <= g["convergence_max"]

    qt = Counter(it["metadata"]["qtype"] for it in items)
    quota = cfg["plan"]["qtype_quota"]
    res["qtype_share"] = {t: round(qt[t] / len(items), 3) for t in QTYPES}
    gates["qtype_shares"] = all(abs(qt[t] / len(items) - quota[t]) <= g["qtype_share_tolerance"] for t in QTYPES)
    waves = set(args.waves) if args.waves is not None else {j["wave"] for j in jobs}
    planned = Counter()
    for j in jobs:
        if j["wave"] in waves:
            planned[j["subject"]] += len(j["slots"])
    sc = Counter(it["subject"] for it in items)
    tp = sum(planned.values())
    res["subject_share_vs_plan"] = {s: round((sc[s] / len(items)) / (planned[s] / tp), 3) for s in sorted(planned)}
    gates["subject_shares"] = all(abs(v - 1) <= g["subject_share_rel_tolerance"] for v in res["subject_share_vs_plan"].values())
    chapters = {j["chapter_id"] for j in jobs if j["wave"] in waves}
    covered = {it["metadata"]["chapter_id"] for it in items}
    res["chapters_covered"] = f"{len(chapters & covered)}/{len(chapters)}"
    res["chapters_missing"] = sorted(chapters - covered)
    gates["chapter_coverage"] = not (chapters - covered)
    m = read_json(paths.manifest, {}) or {}
    acc = (m.get("validation") or {}).get("acceptance")
    res["validator_acceptance"] = acc
    gates["validator_acceptance"] = acc is not None and acc >= g["validator_accept_min"]
    dd = read_json(paths.final / "dedup_report.json", {}) or {}
    res["dedup"] = dd.get("counts")
    if args.choices_only_summary:
        s = read_json(Path(args.choices_only_summary))["metrics"]
        res["choices_only_sft"], res["choices_only_probe_a"] = s["probe_b_train_acc"], s["probe_a_full_acc"]
        gates["choices_only"] = (s["probe_b_train_acc"] <= s["probe_a_full_acc"] + g["choices_only_margin"]
                                 and s["probe_b_train_acc"] <= g["choices_only_max"])
    else:
        res["choices_only_sft"] = None

    audit = {"waves": sorted(waves), "metrics": res, "gates": gates, "passed": all(gates.values())}
    write_json(paths.final / "audit.json", audit)
    update_manifest(paths, "audit", {"waves": sorted(waves), "passed": audit["passed"], "gates": gates})
    print(f"items {len(items)} (waves {sorted(waves)}); positions {res['positions']} (re-balanced exactly in s09)")
    print(f"key uniquely longest {100 * res['key_longest']:.1f}% (gate <= {100 * g['key_longest_max']:.0f}%), "
          f"shortest {100 * res['key_shortest']:.1f}% (>= {100 * g['key_shortest_min']:.0f}%)")
    print(f"stem-word overlap cue {100 * res['stem_overlap']:.1f}%, convergence cue {100 * res['convergence']:.1f}% "
          f"(gates <= {100 * g['stem_overlap_max']:.0f}%)")
    print("longest per subject:", res["key_longest_per_subject"])
    print("type shares:", res["qtype_share"])
    print("subject share / plan:", res["subject_share_vs_plan"])
    print(f"chapters covered {res['chapters_covered']}; validator acceptance {acc}; dedup {res['dedup']}")
    if res["choices_only_sft"] is not None:
        print(f"choices-only: SFT {res['choices_only_sft']:.3f} vs Probe A {res['choices_only_probe_a']:.3f}")
    else:
        print("choices-only probe: not run yet (optional, needs a GPU: README step 9b)")
    for k, v in gates.items():
        print(f"  gate {k:<22}{'PASS' if v else 'FAIL'}")
    print(f"AUDIT {'PASSED' if audit['passed'] else 'FAILED'} -> {paths.final / 'audit.json'}")
    sys.exit(0 if audit["passed"] else 1)


if __name__ == "__main__":
    main()
