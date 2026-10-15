"""Step 7 — build the textbook document store docs.jsonl (one lesson section per line, with metadata).

Reads   02_clean/<book_id>.jsonl, 00_inventory/inventory.json, 03_review/review_status.json
Writes  data/processed/cpt/<level>/docs.jsonl

    python scripts/cpt/s06_build_docs.py --level easy
    python scripts/cpt/s06_build_docs.py --level easy --allow-unreviewed     # dry run before the review

Only books that passed the review (s05 report) are included unless --allow-unreviewed is given.
Sections: split each chapter (h1) at its section headings (h2); join small neighbours up to ~800 tokens,
split long ones at paragraph boundaries so that no document is longer than --max-tokens.
"""
import argparse
import collections
import re

from common import (LEVEL_GRADES, PIPELINE_VERSION, TEXTBOOK_LICENSE, count_tokens, level_dir, read_json,
                    read_jsonl, sha1, stdout_utf8, write_jsonl)


def chapters(items):
    """[(seq, chapter_no, title, [items])]; text before the first h1 becomes chapter 0.

    `seq` (1, 2, 3 ... in book order) builds the id, so ids stay unique even when a printed chapter
    number is missing or repeated; `chapter_no` is the number printed in the book when there is one.
    """
    out, cur, seq = [], (0, 0, "", []), 0
    for it in items:
        if it["kind"] == "h1":
            if cur[3]:
                out.append(cur)
            seq += 1
            m = re.match(r"^\s*(\d{1,2})\s*[.)\-–:]?\s+(.*)$", it["text"])
            cur = (seq, int(m[1]) if m else seq, (m[2] if m else it["text"]).strip(), [])
        else:
            cur[3].append(it)
    if cur[3]:
        out.append(cur)
    return out


def sections(items):
    """[(heading or None, [items])] split at h2 headings."""
    out, cur = [], (None, [])
    for it in items:
        if it["kind"] == "h2":
            if cur[1]:
                out.append(cur)
            cur = (it["text"], [it])
        else:
            cur[1].append(it)
    if cur[1]:
        out.append(cur)
    return out


def chunk(items, ntok, max_tokens):
    """Split one section into pieces of <= max_tokens at paragraph boundaries."""
    pieces, cur, tot = [], [], 0
    for it in items:
        t = ntok[id(it)]
        if cur and tot + t > max_tokens:
            pieces.append(cur)
            cur, tot = [], 0
        cur.append(it)
        tot += t
    if cur:
        pieces.append(cur)
    return pieces


def split_long_paragraphs(items, max_tokens):
    """Paragraphs longer than ~max_tokens (tables or lists joined into one line) -> sentence groups."""
    out = []
    for it, n in zip(items, count_tokens([it["text"] for it in items])):
        if it["kind"] != "p" or n <= max_tokens:
            out.append(it)
            continue
        target = len(it["text"]) * (0.6 * max_tokens / n)          # characters per piece
        cur = ""
        pieces = re.split(r"(?<=[.?!।])\s+", it["text"])
        if max(map(len, pieces)) > target:                        # no sentence ends: split at spaces
            pieces = it["text"].split(" ")
        for sent in pieces:
            if cur and len(cur) + len(sent) > target:
                out.append({**it, "text": cur})
                cur = ""
            cur = f"{cur} {sent}" if cur else sent
        if cur:
            out.append({**it, "text": cur})
    return out


def build_book(book, items, args):
    items = split_long_paragraphs(items, args.max_tokens)
    # +3 per paragraph: the "\n\n" separators and "## " heading marks of the joined text
    ntok = dict(zip(map(id, items), (n + 3 for n in count_tokens([it["text"] for it in items]))))
    docs = []
    for seq, ch_no, ch_title, ch_items in chapters(items):
        units = []                                   # (section heading, [items], tokens)
        for head, sec in sections(ch_items):
            for piece in chunk(sec, ntok, args.max_tokens):
                units.append((head, piece, sum(ntok[id(i)] for i in piece)))
        groups = []
        for head, piece, t in units:                 # greedy merge up to the target size
            if groups and groups[-1][2] < args.target_tokens and groups[-1][2] + t <= args.max_tokens:
                g = groups[-1]
                groups[-1] = (g[0] or head, g[1] + piece, g[2] + t)
            else:
                groups.append((head, piece, t))
        if len(groups) > 1 and groups[-1][2] < args.min_tokens:   # tiny tail -> previous document
            h, p, t = groups.pop()
            g = groups[-1]
            groups[-1] = (g[0], g[1] + p, g[2] + t)
        for s_no, (head, piece, _) in enumerate(groups, 1):
            text = "\n\n".join(("## " + i["text"]) if i["kind"] == "h2" else i["text"] for i in piece)
            extraction = collections.Counter(i["extraction"] for i in piece).most_common(1)[0][0]
            docs.append({
                "id": f"{book['book_id']}/ch{seq:02d}/s{s_no:02d}",
                "parent_id": f"{book['book_id']}/ch{seq:02d}",
                "book_id": book["book_id"], "source": "textbook",
                "subject": book["subject"], "subject_si": book["subject_si"], "grade": book["grade"],
                "level": args.level, "medium": book["medium"], "lang": "si", "kind": "section",
                "chapter_no": ch_no, "chapter": ch_title, "section": head or ch_title,
                "page_start": min(i["page"] for i in piece), "page_end": max(i["page"] for i in piece),
                "extraction": extraction, "text": text, "n_chars": len(text), "n_tokens": None,
                "sha1": sha1(text), "license": TEXTBOOK_LICENSE, "pipeline_version": PIPELINE_VERSION,
            })
    for d, n in zip(docs, count_tokens([d["text"] for d in docs])):
        d["n_tokens"] = n
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="easy", choices=LEVEL_GRADES)
    ap.add_argument("--allow-unreviewed", action="store_true")
    ap.add_argument("--min-tokens", type=int, default=150)
    ap.add_argument("--target-tokens", type=int, default=800)
    ap.add_argument("--max-tokens", type=int, default=1800)
    args = ap.parse_args()
    stdout_utf8()

    base = level_dir(args.level)
    inv = {b["book_id"]: b for b in read_json(base / "00_inventory" / "inventory.json")}
    status = read_json(base / "03_review" / "review_status.json", {})
    docs, skipped = [], []
    for path in sorted((base / "02_clean").glob("*.jsonl")):
        bid = path.stem
        if status.get(bid, {}).get("status") != "pass" and not args.allow_unreviewed:
            skipped.append(bid)
            continue
        docs += build_book(inv[bid], list(read_jsonl(path)), args)

    dup = [i for i, n in collections.Counter(d["id"] for d in docs).items() if n > 1]
    if dup:
        raise SystemExit(f"duplicate ids: {dup[:5]}")
    docs.sort(key=lambda d: d["id"])
    write_jsonl(base / "docs.jsonl", docs)

    per = collections.defaultdict(lambda: [0, 0])
    for d in docs:
        per[(d["subject"], d["grade"])][0] += 1
        per[(d["subject"], d["grade"])][1] += d["n_tokens"]
    print(f"{'subject':20s} grade  docs   tokens")
    for (s, g), (n, t) in sorted(per.items()):
        print(f"{s:20s} {g:5d} {n:5d} {t:8d}")
    lens = sorted(d["n_tokens"] for d in docs) or [0]
    print(f"\n{len(docs)} documents, {sum(lens):,} tokens; doc length min/median/max = "
          f"{lens[0]}/{lens[len(lens) // 2]}/{lens[-1]} -> {base / 'docs.jsonl'}")
    if skipped:
        print(f"NOT included (review not passed): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
