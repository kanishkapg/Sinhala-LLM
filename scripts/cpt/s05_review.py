"""Step 6 — quality review: automatic checks, review samples for a Sinhala reader, and the verdict report.

Sub-commands (run from the repo root):

  python scripts/cpt/s05_review.py wordlist            # once: Sinhala word list from Wikipedia (54 MB download)
  python scripts/cpt/s05_review.py auto   --level easy # automatic checks on every paragraph of 02_clean/
  python scripts/cpt/s05_review.py sample --level easy --round 1
        -> 03_review/round1/tasks_labelstudio.json  (import into Label Studio)
           03_review/round1/review_sheet.csv        (or fill this in Google Sheets / Excel)
           03_review/pages/<book_id>/pNNNN.jpg      page images shown next to the text
  ... a Sinhala reader marks every sample OK / Minor errors / Broken (+ error type, + wrong -> right notes) ...
  python scripts/cpt/s05_review.py report --level easy --round 1 --labels <export.json | filled.csv>
        -> 03_review/review_status.json   pass / fail per book (s06 only uses books that passed)

Acceptance per book: >= 95 % of samples "OK" and no wrong form repeated in 2+ samples.
"""
import argparse
import bisect
import collections
import csv
import json
import random
import re

from common import (LEVEL_GRADES, ROOT, SEED, level_dir, read_json, read_jsonl, sinhala_ratio,
                    stdout_utf8, write_json)

COMMON = ROOT / "data" / "processed" / "cpt" / "common"
MALFORMED = re.compile(r"(?:^|[\s(\"'“‘])[්ා-ෟෲෳංඃ]"   # sign at word start
                       r"|[ා-ෟෲෳ]{2,}"                               # two vowel signs
                       r"|්[්ා-ෟ]")                                  # hal + sign
OK_CHARS = set(" \u200D0123456789.,:;?!()-/%'\"=+*×÷–—‘’“”[]•…_<>#\n" "√Ωµ°″²³→←↑↓»{}^")   # 2nd part: real symbols seen in the books
WORD = re.compile(r"[\u0D80-\u0DFF\u200D]+")


def paragraphs(level):
    base = level_dir(level) / "02_clean"
    for path in sorted(base.glob("*.jsonl")):
        for i, it in enumerate(read_jsonl(path)):
            yield {"id": f"{path.stem}#{i:05d}", "book_id": path.stem, "idx": i, **it}


# ------------------------------------------------------------------ wordlist

def cmd_wordlist(args):
    """Word frequencies from Sinhala Wikipedia: the reference vocabulary for the OOV check."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("wikimedia/wikipedia", "20231101.si/train-00000-of-00001.parquet", repo_type="dataset")
    freq = collections.Counter()
    for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=2000):
        for t in batch.column(0).to_pylist():
            freq.update(WORD.findall(t))
    words = {w: n for w, n in freq.items() if n >= 3}
    write_json(COMMON / "si_wordfreq.json", words)
    print(f"{len(words)} Sinhala word forms (freq >= 3) -> {COMMON / 'si_wordfreq.json'}")


# ------------------------------------------------------------------ auto

def score(text, vocab):
    n = max(len(text), 1)
    malformed = len(MALFORMED.findall(text))
    odd = [c for c in text if not ("\u0D80" <= c <= "\u0DFF" or c.isascii() and c.isalnum() or c in OK_CHARS)]
    words = WORD.findall(text)
    oov = sum(1 for w in words if w not in vocab) / len(words) if vocab and words else None
    return {"malformed_per_1k": round(1000 * malformed / n, 2), "odd_chars": len(odd),
            "odd_examples": "".join(sorted(set(odd)))[:20], "sinhala_ratio": round(sinhala_ratio(text), 3),
            "oov_rate": None if oov is None else round(oov, 3)}


def cmd_auto(args):
    vocab = read_json(COMMON / "si_wordfreq.json", {})
    if not vocab:
        print("(no word list: run `s05_review.py wordlist` to enable the out-of-vocabulary check)")
    out = level_dir(args.level) / "03_review"
    rows, per_book = [], collections.defaultdict(list)
    for p in paragraphs(args.level):
        if p["kind"] != "p":
            continue
        s = score(p["text"], vocab)
        s.update(id=p["id"], book_id=p["book_id"], page=p["page"], n_chars=len(p["text"]))
        rows.append(s)
        per_book[p["book_id"]].append(s)

    # badness: z-like rank of each signal inside its book; used to pick the "worst" samples
    for bid, ss in per_book.items():
        for key in ("malformed_per_1k", "odd_chars", "oov_rate"):
            vals = sorted(x[key] or 0 for x in ss)
            for x in ss:                   # share of the book's paragraphs that score lower (ties -> lowest rank)
                x[f"r_{key}"] = round(bisect.bisect_left(vals, x[key] or 0) / max(len(vals) - 1, 1), 3)
        for x in ss:
            x["badness"] = round(max(x["r_malformed_per_1k"], x["r_odd_chars"], x["r_oov_rate"]) +
                                 (1 if x["malformed_per_1k"] > 1 else 0) + (1 if x["odd_chars"] else 0), 3)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "auto_scores.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {}
    print(f"{'book_id':28s} {'paras':>6s} {'malf/1k':>8s} {'odd':>6s} {'oov':>6s}  flag")
    for bid, ss in sorted(per_book.items()):
        chars = sum(x["n_chars"] for x in ss)
        mal = sum(x["malformed_per_1k"] * x["n_chars"] / 1000 for x in ss) / max(chars, 1) * 1000
        odd = sum(x["odd_chars"] for x in ss)
        oovs = [x["oov_rate"] for x in ss if x["oov_rate"] is not None]
        oov = sum(oovs) / len(oovs) if oovs else None
        flag = "CHECK" if mal > 1 or odd > 0.001 * chars else ""
        summary[bid] = {"paragraphs": len(ss), "chars": chars, "malformed_per_1k": round(mal, 2),
                        "odd_chars": odd, "mean_oov": oov and round(oov, 3), "flag": flag}
        print(f"{bid:28s} {len(ss):6d} {mal:8.2f} {odd:6d} {oov if oov is None else round(oov, 3)!s:>6s}  {flag}")
    write_json(out / "auto_summary.json", summary)
    odd_all = collections.Counter(c for r in rows for c in r["odd_examples"])
    if odd_all:
        print("\nUnexpected characters (fix in fm_converter.py PRE/POST or explain):", "".join(odd_all)[:80])


# ------------------------------------------------------------------ sample

def cmd_sample(args):
    import pymupdf
    base = level_dir(args.level)
    out = base / "03_review"
    scores = {r["id"]: r for r in read_jsonl(out / "auto_scores.jsonl")} if (out / "auto_scores.jsonl").exists() else {}
    if not scores:
        raise SystemExit("Run `s05_review.py auto` first")
    status = read_json(out / "review_status.json", {})
    inv = {b["book_id"]: b for b in read_json(base / "00_inventory" / "inventory.json")}
    rng = random.Random(SEED + args.round)
    by_book = collections.defaultdict(list)
    for p in paragraphs(args.level):
        if p["kind"] == "p" and len(p["text"]) >= 80:
            by_book[p["book_id"]].append(p)

    rdir = out / f"round{args.round}"
    rdir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for bid, ps in sorted(by_book.items()):
        if args.books and bid not in args.books:
            continue
        if status.get(bid, {}).get("status") == "pass" and not args.books:
            continue                                   # already accepted in an earlier round
        rand = rng.sample(ps, min(args.n_random, len(ps)))
        chosen = {p["id"] for p in rand}
        worst = sorted((p for p in ps if p["id"] not in chosen), key=lambda p: -scores.get(p["id"], {}).get("badness", 0))
        picked = [(p, "random") for p in rand] + [(p, "worst") for p in worst[:args.n_worst]]
        if inv[bid]["scanned"]:
            picked += [(p, "ocr") for p in rng.sample(ps, min(10, len(ps))) if p["id"] not in chosen]
        doc = pymupdf.open(ROOT / inv[bid]["file"])
        for p, why in picked:
            img = out / "pages" / bid / f"p{p['page']:04d}.jpg"
            if not img.exists():
                img.parent.mkdir(parents=True, exist_ok=True)
                doc[p["page"] - 1].get_pixmap(dpi=110).save(img, jpg_quality=80)
            rel = img.relative_to(ROOT).as_posix()
            tasks.append({"id": p["id"], "book_id": bid, "page": p["page"], "why": why, "text": p["text"],
                          "image": f"/data/local-files/?d={rel}", "image_path": str(img)})
    rng.shuffle(tasks)
    write_json(rdir / "tasks_labelstudio.json", [{"data": t} for t in tasks])
    with open(rdir / "review_sheet.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "book_id", "page", "image_path", "text", "verdict (OK/Minor errors/Broken)",
                    "error_type", "notes (wrong -> right)"])
        for t in tasks:
            w.writerow([t["id"], t["book_id"], t["page"], t["image_path"], t["text"], "", "", ""])
    print(f"{len(tasks)} samples from {len({t['book_id'] for t in tasks})} books -> {rdir}")
    print("Label Studio: set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true and "
          f"LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={ROOT}, add a Local Files storage with path "
          f"{out / 'pages'}, then import tasks_labelstudio.json (template: scripts/cpt/config/labelstudio.xml)")


# ------------------------------------------------------------------ report

def read_labels(path):
    """Label Studio JSON export (min or full) or the filled review_sheet.csv -> [{id, verdict, notes}]"""
    out = []
    if str(path).endswith(".csv"):
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                v = r.get("verdict (OK/Minor errors/Broken)") or r.get("verdict") or ""
                out.append({"id": r["id"], "verdict": v.strip(), "notes": r.get("notes (wrong -> right)", "")})
        return out
    data = json.loads(open(path, encoding="utf-8").read())
    for t in data:
        if "annotations" in t:                       # full JSON export
            res = {r["from_name"]: r["value"] for a in t["annotations"][-1:] for r in a["result"]}
            verdict = (res.get("verdict", {}).get("choices") or [""])[0]
            notes = " ".join(res.get("notes", {}).get("text", []))
            out.append({"id": t["data"]["id"], "verdict": verdict, "notes": notes})
        else:                                          # JSON-MIN export
            out.append({"id": t["id"], "verdict": t.get("verdict", ""), "notes": t.get("notes", "")})
    return out


def cmd_report(args):
    out = level_dir(args.level) / "03_review"
    labels = [l for l in read_labels(args.labels) if l["verdict"]]
    status = read_json(out / "review_status.json", {})
    per_book = collections.defaultdict(list)
    for l in labels:
        per_book[l["id"].split("#")[0]].append(l)
    print(f"{'book_id':28s} {'n':>4s} {'OK%':>6s}  repeated errors / status")
    for bid, ls in sorted(per_book.items()):
        ok = sum(l["verdict"] == "OK" for l in ls) / len(ls)
        wrong = collections.Counter()
        for l in ls:
            for m in re.finditer(r"(\S+)\s*(?:->|→)\s*(\S+)", l["notes"] or ""):
                wrong[(m[1], m[2])] += 1
        repeated = {f"{a} -> {b}": n for (a, b), n in wrong.items() if n >= 2}
        passed = ok >= 0.95 and not repeated
        status[bid] = {"status": "pass" if passed else "fail", "round": args.round, "samples": len(ls),
                       "ok_share": round(ok, 3), "repeated_errors": repeated,
                       "all_fixes": {f"{a} -> {b}": n for (a, b), n in wrong.most_common()}}
        print(f"{bid:28s} {len(ls):4d} {ok:6.1%}  {'PASS' if passed else 'FAIL'} {repeated or ''}")
    write_json(out / "review_status.json", status)
    print(f"\nwrote {out / 'review_status.json'}")
    if any(v["status"] == "fail" for v in status.values()):
        print("Failed books: add the fixes to fm_converter.py (PRE / POST) or config/books_<level>.json, re-run "
              "s02 --force / s04 for those books, then `auto`, then `sample --round <next>` and review again.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("wordlist")
    for name in ("auto", "sample", "report"):
        p = sub.add_parser(name)
        p.add_argument("--level", default="easy", choices=LEVEL_GRADES)
        if name in ("sample", "report"):
            p.add_argument("--round", type=int, default=1)
        if name == "sample":
            p.add_argument("--n-random", type=int, default=20)
            p.add_argument("--n-worst", type=int, default=20)
            p.add_argument("--books", nargs="*")
        if name == "report":
            p.add_argument("--labels", required=True, help="Label Studio export (.json) or filled review_sheet.csv")
    args = ap.parse_args()
    stdout_utf8()
    {"wordlist": cmd_wordlist, "auto": cmd_auto, "sample": cmd_sample, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
