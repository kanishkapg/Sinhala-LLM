"""Step 8 — deduplicate (exact + near-duplicate) and decontaminate against the test questions.

Reads   <level>/docs.jsonl and <level>/replay/*.jsonl
        test questions: data/SinhalaMMLU-SharedTask-Test/**/*.json and data/SinhalaMMLU/**/*.json
Writes  <level>/04_dedup/textbook.jsonl, <level>/04_dedup/<replay source>.jsonl
        <level>/04_dedup/report.json   (numbers for the system report / manifest)

    python scripts/cpt/s08_dedup_decontam.py --level easy

Rules
  * exact duplicates (same sha1 of the text) and near duplicates (MinHash on character 5-grams,
    Jaccard >= 0.8): the first copy is kept, in priority order textbooks (higher grade, non-OCR first),
    then replay sources in config order. A replay document that duplicates a textbook is dropped.
  * contamination: word 13-grams of every question+options sequence (whole question when it has
    8-12 words). Replay documents with any hit are DROPPED. Textbooks are allowed source material:
    hits are only counted and listed (use --drop-textbook-hits to remove them too).
The script never prints test questions; only counts and document ids go into the report.
"""
import argparse
import glob
import json
import re
import unicodedata

from common import (CONFIG_DIR, LEVEL_GRADES, ROOT, TEST_GLOBS, level_dir, read_json, read_jsonl,
                    stdout_utf8, write_json, write_jsonl)

N = 13
MIN_Q_WORDS = 8
PUNCT = re.compile(r"[^\w\u0D80-\u0DFF\u200D\s]")


def words(text):
    text = unicodedata.normalize("NFC", text).lower().replace("\u200D", "")
    return PUNCT.sub(" ", text).split()


def test_grams():
    """{n: set(tuple)} of n-grams from all visible test questions (+ their options)."""
    grams, n_q = {}, 0
    for pattern in TEST_GLOBS:
        for path in glob.glob(str(ROOT / pattern), recursive=True):
            with open(path, encoding="utf-8") as f:
                try:
                    items = json.load(f)
                except json.JSONDecodeError:
                    continue
            for it in items if isinstance(items, list) else []:
                if "question" not in it:
                    continue
                n_q += 1
                for seq in (words(it["question"]), words(it["question"] + " " + " ".join(it.get("choices", [])))):
                    if len(seq) >= N:
                        grams.setdefault(N, set()).update(tuple(seq[i:i + N]) for i in range(len(seq) - N + 1))
                    elif len(seq) >= MIN_Q_WORDS:
                        grams.setdefault(len(seq), set()).add(tuple(seq))
    return grams, n_q


def contaminated(text, grams):
    w = words(text)
    for n, s in grams.items():
        for i in range(len(w) - n + 1):
            if tuple(w[i:i + n]) in s:
                return True
    return False


def minhash(text, num_perm):
    from datasketch import MinHash
    m = MinHash(num_perm=num_perm, seed=1)
    t = re.sub(r"\s+", " ", text)
    m.update_batch(t[i:i + 5].encode("utf-8") for i in range(max(len(t) - 4, 1)))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="easy", choices=LEVEL_GRADES)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--drop-textbook-hits", action="store_true")
    args = ap.parse_args()
    stdout_utf8()
    from datasketch import MinHashLSH

    base = level_dir(args.level)
    order = [s["name"] for s in read_json(CONFIG_DIR / "replay_sources.json")["sources"]]
    sources = {"textbook": sorted(read_jsonl(base / "docs.jsonl"),
                                  key=lambda d: (d["extraction"] == "ocr", -d["grade"], d["id"]))}
    for name in order:
        p = base / "replay" / f"{name}.jsonl"
        if p.exists():
            sources[name] = list(read_jsonl(p))

    grams, n_q = test_grams()
    print(f"test questions indexed: {n_q} ({sum(len(s) for s in grams.values()):,} n-grams)")
    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    seen_sha, owner = {}, {}
    report = {"threshold": args.threshold, "ngram": N, "test_questions": n_q, "sources": {}}
    for name, docs in sources.items():
        kept, r = [], {"in": len(docs), "exact_dup": 0, "near_dup": 0, "contaminated": 0, "contaminated_ids": []}
        for d in docs:
            if d["sha1"] in seen_sha:
                r["exact_dup"] += 1
                continue
            m = minhash(d["text"], args.num_perm)
            if lsh.query(m):
                r["near_dup"] += 1
                continue
            if contaminated(d["text"], grams):
                r["contaminated"] += 1
                r["contaminated_ids"].append(d["id"])
                if name != "textbook" or args.drop_textbook_hits:
                    continue
            seen_sha[d["sha1"]] = d["id"]
            lsh.insert(d["id"], m)
            owner[d["id"]] = name
            kept.append(d)
        r["out"] = len(kept)
        r["tokens_out"] = sum(d["n_tokens"] for d in kept)
        if name != "textbook":
            r["contaminated_ids"] = r["contaminated_ids"][:50]
        report["sources"][name] = r
        write_jsonl(base / "04_dedup" / f"{name}.jsonl", kept)
        action = "dropped" if name != "textbook" or args.drop_textbook_hits else "kept (allowed source)"
        print(f"{name:16s} in={r['in']:6d} exact={r['exact_dup']:5d} near={r['near_dup']:5d} "
              f"test-overlap={r['contaminated']:4d} {action:22s} out={r['out']:6d} tokens={r['tokens_out']:,}")
    write_json(base / "04_dedup" / "report.json", report)
    print(f"\nreport -> {base / '04_dedup' / 'report.json'}")


if __name__ == "__main__":
    main()
