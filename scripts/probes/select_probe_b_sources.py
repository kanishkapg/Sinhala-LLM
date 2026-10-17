"""Pick the textbook sections that Probe B questions are written from.

Reads   data/processed/cpt/easy/04_dedup/textbook.jsonl, mix/summary.json (the held-out chapters)
Writes  data/probes/easy/probe_b_sources.jsonl   one row per section {"source_id", "split", "subject", "grade", "role"}
        <--dump>/<split>_<subject>.md            the section texts, to read while writing the questions

    python scripts/probes/select_probe_b_sources.py --dump <folder>

* split "train":   per subject, PER_SUBJECT sections from chapters that are in training, spread over the books
                   (grades) and over different chapters. 2 questions per section.
* split "heldout": per held-out chapter, up to HELDOUT_SECTIONS sections. ~5 questions per chapter.
* role "alternate" sections are spares, used only when a primary section has no usable facts (a list of
  exercises, a poem, a figure that did not survive extraction).
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

from probe_common import MIX_SUMMARY, PROBE_DIR, SEED, TEXTBOOKS, read_json, read_jsonl, stdout_utf8, write_jsonl

PER_SUBJECT = 11
ALTERNATES = 4
MIN_CHARS = 1000
HELDOUT_SECTIONS = 3
HELDOUT_MIN_CHARS = 700


def pick_train(sections, rng):
    """Round-robin over the subject's books, a new chapter each time, one random section per chapter."""
    by_book = defaultdict(lambda: defaultdict(list))
    for s in sections:
        by_book[s["book_id"]][s["parent_id"]].append(s)
    books = sorted(by_book)
    rng.shuffle(books)
    chapters = {b: sorted(by_book[b]) for b in books}
    for b in books:
        rng.shuffle(chapters[b])
    picked = []
    while len(picked) < PER_SUBJECT + ALTERNATES and any(chapters.values()):
        for b in books:
            if chapters[b] and len(picked) < PER_SUBJECT + ALTERNATES:
                picked.append(rng.choice(sorted(by_book[b][chapters[b].pop()], key=lambda s: s["id"])))
    # Chapters ran out (Art has 3): fill with further sections from any chapter.
    rest = sorted((s for s in sections if s not in picked), key=lambda s: s["id"])
    rng.shuffle(rest)
    picked += rest[: PER_SUBJECT + ALTERNATES - len(picked)]
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, help="folder for the readable section texts")
    args = ap.parse_args()
    stdout_utf8()

    rng = random.Random(f"{SEED}-probe-b")
    heldout = set(read_json(MIX_SUMMARY)["val_chapters"])
    rows = list(read_jsonl(TEXTBOOKS))

    train_by_subj = defaultdict(list)
    held_by_ch = defaultdict(list)
    for r in rows:
        if r["parent_id"] in heldout:
            if r["n_chars"] >= HELDOUT_MIN_CHARS:
                held_by_ch[r["parent_id"]].append(r)
        elif r["n_chars"] >= MIN_CHARS:
            train_by_subj[r["subject"]].append(r)

    out = []
    for subj in sorted(train_by_subj):
        for k, s in enumerate(pick_train(train_by_subj[subj], rng)):
            out.append((s, "train", "primary" if k < PER_SUBJECT else "alternate"))
    for ch in sorted(heldout):
        secs = sorted(held_by_ch[ch], key=lambda s: s["id"])
        if not secs:  # every section is short: take the longest one anyway
            secs = [max((r for r in rows if r["parent_id"] == ch), key=lambda r: r["n_chars"])]
        picked = sorted(rng.sample(secs, min(HELDOUT_SECTIONS, len(secs))), key=lambda s: s["id"])
        out += [(s, "heldout", "primary") for s in picked]

    write_jsonl(PROBE_DIR / "probe_b_sources.jsonl", (
        {"source_id": s["id"], "split": split, "subject": s["subject"], "grade": s["grade"], "role": role}
        for s, split, role in out))
    for split in ("train", "heldout"):
        prim = [s for s, sp, role in out if sp == split and role == "primary"]
        print(f"{split:8s}: {len(prim)} primary sections, {sum(s['n_chars'] for s in prim):,} chars")

    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)
        files = defaultdict(list)
        for s, split, role in out:
            files[f"{split}_{s['subject'].replace(' ', '_')}.md"].append(
                f"<<< {s['id']} | {split} | {role} | grade {s['grade']} | {s['chapter']} | {s.get('section') or ''} >>>\n\n{s['text']}\n")
        for name, parts in files.items():
            (args.dump / name).write_text("\n\n".join(parts), encoding="utf-8")
        print(f"section texts -> {args.dump}")


if __name__ == "__main__":
    main()
