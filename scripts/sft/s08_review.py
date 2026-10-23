"""Step 10 of docs/prepare_sft_dataset.md: the Sinhala-reader review.

    python scripts/sft/s08_review.py sample --level easy --round wave0 --waves 0 --all
    python scripts/sft/s08_review.py sample --level easy --round pilot --waves 1 --n 100 --rejected 30
    (fill in the verdict columns in Excel, save as CSV UTF-8)
    python scripts/sft/s08_review.py report --level easy --round pilot

The CSV opens in Excel (UTF-8 with BOM). Verdict columns: key_correct, single_best, natural_sinhala,
fair_distractors (Y/N) and cue (Y if the answer can be guessed from a surface cue), plus a free comment.
`report` prints the gate and appends accepted items the reviewer marked bad to review/review_drops.jsonl, which
s09 leaves out of the training files.
"""
import argparse
import csv
from collections import Counter, defaultdict

from sft_common import (Paths, append_jsonl, ensure_dirs, level_arg, read_jsonl, read_jsonl_tolerant, stable_int,
                        stdout_utf8, update_manifest)

VERDICTS = ["key_correct", "single_best", "natural_sinhala", "fair_distractors", "cue", "comment"]


def stratified(items: list[dict], n: int, tag: str) -> list[dict]:
    cells = defaultdict(list)
    for it in sorted(items, key=lambda x: stable_int(tag, x["id"])):
        cells[(it["subject"], it["metadata"]["qtype"])].append(it)
    out, keys = [], sorted(cells, key=lambda k: stable_int(tag, *k))
    while len(out) < n and any(cells.values()):
        for k in keys:
            if cells[k] and len(out) < n:
                out.append(cells[k].pop(0))
    return out


def row(it: dict, status: str) -> dict:
    m = it["metadata"]
    r = {"item_id": it["id"], "status": status, "subject": it["subject"], "qtype": m["qtype"],
         "question": it["question"]}
    for k in range(4):
        r[f"option_{k + 1}"] = it["choices"][k]
    r["key"] = it["answer"]
    r["evidence"] = m.get("evidence") or " | ".join(f"{s['label']}: {s['evidence']}" for s in m.get("statements", []))
    r["explanation_en"] = m.get("explanation_en", "")
    r["reject_reasons"] = ", ".join(it.get("reject_reasons", []))
    v = m.get("validation") or {}
    r["validator_answer"] = v.get("answer", "")
    r["validator_issues"] = "; ".join(v.get("issues", []))
    for c in VERDICTS:
        r[c] = ""
    return r


def sample(paths: Paths, args) -> None:
    drops = {r["id"] for r in read_jsonl_tolerant(paths.final / "audit_drops.jsonl")}
    acc = [it for it in read_jsonl(paths.final / "items.jsonl") if it["id"] not in drops]
    rej = list(read_jsonl_tolerant(paths.checked / "rejected.jsonl")) + \
        list(read_jsonl_tolerant(paths.validated / "rejected.jsonl"))
    if args.waves is not None:
        acc = [it for it in acc if it["metadata"]["wave"] in args.waves]
        rej = [it for it in rej if it["metadata"]["wave"] in args.waves]
    if not args.all:
        acc = stratified(acc, args.n, args.round + "a")
        rej = stratified(rej, args.rejected, args.round + "r")
    rows = [row(it, "accepted") for it in acc] + [row(it, "rejected") for it in rej]
    out = paths.review / f"review_{args.round}.csv"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists (it may hold verdicts): use --force to overwrite")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(acc)} accepted + {len(rej)} rejected items -> {out}")
    print("Fill in key_correct / single_best / natural_sinhala / fair_distractors (Y/N), cue (Y/N), comment.")


def yes(v: str) -> bool:
    return str(v).strip().upper() in ("Y", "YES", "1", "TRUE", "ඔව්")


def report(paths: Paths, args) -> None:
    path = paths.review / f"review_{args.round}.csv"
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    done = [r for r in rows if r["key_correct"].strip()]
    if not done:
        raise SystemExit(f"no verdicts in {path} yet")

    def good(r):
        return (yes(r["key_correct"]) and yes(r["single_best"]) and yes(r["natural_sinhala"])
                and yes(r["fair_distractors"]) and not yes(r["cue"]))
    acc = [r for r in done if r["status"] == "accepted"]
    rej = [r for r in done if r["status"] == "rejected"]
    g_acc = sum(good(r) for r in acc)
    print(f"reviewed {len(done)}/{len(rows)}: accepted items fully good {g_acc}/{len(acc)} "
          f"({100 * g_acc / max(len(acc), 1):.0f}%; gate >= 90%)")
    for c in ["key_correct", "single_best", "natural_sinhala", "fair_distractors"]:
        bad = sum(not yes(r[c]) for r in acc)
        print(f"  {c:<18} problems: {bad}")
    print(f"  cue found: {sum(yes(r['cue']) for r in acc)}")
    print("  problems by type:", dict(Counter(r["qtype"] for r in acc if not good(r))))
    if rej:
        wrong = sum(good(r) for r in rej)
        print(f"rejected items the reviewer finds good: {wrong}/{len(rej)} ({100 * wrong / len(rej):.0f}%; "
              "above 50% = the validator is too strict)")
    existing = {r["id"] for r in read_jsonl_tolerant(paths.review / "review_drops.jsonl")}
    n_new = 0
    for r in acc:
        if not good(r) and r["item_id"] not in existing:
            append_jsonl(paths.review / "review_drops.jsonl", {"id": r["item_id"], "round": args.round,
                                                              "comment": r.get("comment", "")})
            n_new += 1
    print(f"{n_new} reviewer-rejected accepted items -> review/review_drops.jsonl (left out by s09)")
    update_manifest(paths, f"review_{args.round}", {"reviewed": len(done), "accepted_good": g_acc,
                                                    "accepted_reviewed": len(acc), "rejected_reviewed": len(rej)})
    if acc and g_acc / len(acc) < 0.9:
        print("GATE FAILED: fewer than 90% of accepted items are fully good. Read the comments, fix the prompts, "
              "and do not generate the next wave yet.")


def main():
    ap = argparse.ArgumentParser(description="Sinhala-reader review sample and report")
    ap.add_argument("action", choices=["sample", "report"])
    level_arg(ap)
    ap.add_argument("--round", required=True, help="name of this review round, e.g. wave0 or pilot")
    ap.add_argument("--waves", type=int, nargs="*", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--rejected", type=int, default=30)
    ap.add_argument("--all", action="store_true", help="every item of the waves (wave 0)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    stdout_utf8()
    paths = Paths(args.level)
    ensure_dirs(paths)
    (sample if args.action == "sample" else report)(paths, args)


if __name__ == "__main__":
    main()
