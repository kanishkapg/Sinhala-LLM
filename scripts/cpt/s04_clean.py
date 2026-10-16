"""Step 5 — clean one book's extracted lines into headings + paragraphs.

Reads   01_extracted/<book_id>.jsonl  and  config/books_<level>.json
Writes  02_clean/<book_id>.jsonl   items {"kind": "h1"|"h2"|"p", "page", "text", "extraction"}
        02_clean/<book_id>.txt     the same as readable Markdown with <!-- p.N --> page markers (for review)
        02_clean/clean_stats.json  how many lines each rule removed, per book

    python scripts/cpt/s04_clean.py --level easy
    python scripts/cpt/s04_clean.py --level easy --books science_g7_p1_si

What it does (in this order): skip front/back-matter pages -> drop running headers/footers/page numbers
-> drop figure labels, captions and diagram blocks -> detect headings (h1 = chapter, h2 = section)
-> re-join hard-wrapped lines into paragraphs (also across pages) -> normalise bullets, blanks, Unicode
-> drop non-Sinhala paragraphs.

Tune per book in config/books_<level>.json:
  "skip_pages": [1, 2, 3, [200, 216]]   pages or [from, to] ranges to drop
  "chapter_pages": {"12": 1, "30": 2}   pages where a chapter starts that the automatic detection missed
                                        (page -> chapter number); the largest heading on the page is the title
  "chapter_min_size": 20.0              only if the book has no numbered chapter openers: font size from
                                        which a heading is a chapter title (h1); null = automatic

Chapter detection: a chapter opener is a page with a big stand-alone number (01, 4, 4.) or a big
numbered title (3. ...); the numbers must run 1, 2, 3 ... through the book (one gap allowed).
Missed chapters are not lost, their text joins the previous chapter. clean_stats.json shows
"chapter_detection": "number" | "size" | "none" per book.
"""
import argparse
import collections
import re
import statistics

from common import (CONFIG_DIR, LEVEL_GRADES, level_dir, normalize, read_json, read_jsonl, repair_sinhala,
                    sinhala_ratio, stdout_utf8, write_json, write_jsonl)

# Front/back matter markers (pages near the start/end of a book that contain one are skipped).
FRONT_MATTER = re.compile(r"ශ්\u200Dරී ලංකා මාතා|ජාතික ගීය|එක මවකගෙ|පෙරවදන|පටුන|සියලු හිමිකම්|ISBN|"
                          r"ලේඛක මණ්ඩලය|සංස්කාරක මණ්ඩලය|පාඨ මාලා|ප්\u200Dරකාශක|මුද්\u200Dරණය")
FOOTER = re.compile(r"නොමිලේ බෙදා හැරීම|For free distribution|සියලු හිමිකම් ඇවිරිණි")
PAGE_NO = re.compile(r"[\divxlIVXL\s\-–—|]{1,8}")
CAPTION = re.compile(r"^\(?\s*(\d+(\.\d+)*\s*)?(රූපය|රූප සටහන|සටහන|ප්\u200Dරස්තාරය)\s*[\d.\s]*\)?(\s|$)")
SENT_END = tuple(".?!:;)]”'\"।")
BULLET = re.compile(r"^\s*[•▪●○◦➢►✓✔\*\-–]\s*")
NUMBERED = re.compile(r"^\s*(\(?\d{1,2}[.)]|\(?[a-z]\)|\([අ-ෆ]\)|[අ-ෆ]\))\s+")
LEADER = re.compile(r"(?:[.…·]\s?){4,}|_{3,}")
SINHALA_START = re.compile(r"[\u0D80-\u0DFF]")
NUM_ONLY = re.compile(r"0?(\d{1,2})\s*\.?")                  # "01", "4", "4."
NUM_TITLE = re.compile(r"0?(\d{1,2})\s*[.)]\s*([^\d\s.].*)")  # "3. title" (not "3.1 section")
NOISE_CHARS = set("|~।॥₹£$෴")                                # music bar lines / OCR of notation and diagrams
SINHALA_WORD = re.compile(r"[\u0D80-\u0DFF\u200D]+")
FM_LEFTOVER = re.compile(r"[;%<'¾ßñúÈÍÿ]")


def noise_reason(t):
    """Lines that are not prose: music notation / OCR'd diagrams, letter-spaced text, FM codes in a Latin font."""
    n = sum(c in NOISE_CHARS for c in t)
    if n >= 3 and n >= 0.03 * len(t):
        return "notation_noise"                  # | ස ස ස | රි රි රි |   ෴ස෴ි෴෴~෴෴
    toks = t.split()
    sin = [w for w in toks if SINHALA_WORD.fullmatch(w)]
    if len(sin) >= 6 and sum(len(w) <= 2 for w in sin) >= 0.7 * len(sin):
        return "spaced_letters"                  # කු ඳ තා - කු ඳ ග ජි ත   එ ක ි ෙ න ක ා ට
    lat = [w for w in toks if re.search(r"[A-Za-z]", w)]
    if (not sin and len(lat) >= 6 and FM_LEFTOVER.search(t)
            and sum(len(re.sub(r"[^A-Za-z]", "", w)) <= 2 for w in lat) >= 0.6 * len(lat)):
        return "fm_in_latin_font"                # i x. S; NdK av h k g f . j k  ñ ,
    return None


def page_ranges(spec):
    out = set()
    for x in spec or []:
        if isinstance(x, list):
            out.update(range(x[0], x[1] + 1))
        else:
            out.add(int(x))
    return out


def consecutive_run(page_to_no):
    """Keep the longest run of chapter numbers that go up page by page (1, 2, 3 ... or 11, 12 ... in a
    part-II book; one missing number is tolerated). Numbered list items in big type do not form such a run."""
    best, cur = [], []
    for p, n in sorted(page_to_no.items()):
        if cur and cur[-1][1] < n <= cur[-1][1] + 2:
            cur.append((p, n))
        elif cur and n == cur[-1][1]:
            continue                                   # the same number again on the next page
        else:
            cur = [(p, n)]
        if len(cur) > len(best):
            best = list(cur)
    return dict(best)


def clean_book(records, cfg, stats):
    pages = {r["page"]: r for r in records if r["type"] == "page"}
    lines = [r for r in records if r["type"] == "line"]
    for l in lines:
        l["text"] = repair_sinhala(l["text"])
    n_pages = max(pages) if pages else 0
    by_page = collections.defaultdict(list)
    for l in lines:
        by_page[l["page"]].append(l)

    # --- 1. pages to skip: config + front/back matter markers in the first 12 / last 4 pages
    skip = page_ranges(cfg.get("skip_pages"))
    for p, ls in by_page.items():
        if (p <= 12 or p > n_pages - 4) and FRONT_MATTER.search(" ".join(l["text"] for l in ls)):
            skip.add(p)
    stats["skipped_pages"] = sorted(skip)

    # --- body font size = character-weighted most common size
    sizes = collections.Counter()
    for l in lines:
        if l["size"]:
            sizes[l["size"]] += len(l["text"])
    body = sizes.most_common(1)[0][0] if sizes else None

    # --- 2. running headers / footers: same text (digits ignored) on many pages
    def key(t):
        return re.sub(r"\d+", "#", t.strip())

    def in_margin(l):
        if not l["bbox"]:
            return False
        h = pages[l["page"]]["height"]
        return l["bbox"][1] < 0.08 * h or l["bbox"][3] > 0.92 * h

    margin_pages, any_pages = collections.defaultdict(set), collections.defaultdict(set)
    for l in lines:
        k = key(l["text"])
        any_pages[k].add(l["page"])
        if in_margin(l):
            margin_pages[k].add(l["page"])

    # block-level facts: labels and captions are short blocks, not short lines (the last line of a
    # paragraph is short too and must stay)
    blocks = collections.defaultdict(list)
    for l in lines:
        blocks[(l["page"], l["block"])].append(l)
    block_chars = {k: sum(len(l["text"].strip()) for l in ls) for k, ls in blocks.items()}

    def in_figure(l):
        if not l["bbox"]:
            return False
        pg = pages[l["page"]]
        cx, cy = (l["bbox"][0] + l["bbox"][2]) / 2, (l["bbox"][1] + l["bbox"][3]) / 2
        for x0, y0, x1, y1 in pg["images"]:
            if (x1 - x0) * (y1 - y0) > 0.6 * pg["width"] * pg["height"]:
                continue                     # full-page background picture, not a figure
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                return True
        return False

    def big(l):
        return bool(body and l["size"] and l["size"] >= 1.3 * body)

    def drop_reason(l):
        t = l["text"].strip()
        k = key(t)
        short_block = block_chars[(l["page"], l["block"])] < 120
        if l["page"] in skip:
            return "skipped_page"
        if big(l) and NUM_ONLY.fullmatch(t):
            return None                          # big stand-alone number = chapter opener, not a page number
        if FOOTER.search(t):
            return "footer"
        if r := noise_reason(t):
            return r
        if in_margin(l) and (PAGE_NO.fullmatch(t) or len(margin_pages[k]) >= 3):
            return "header_footer"
        if len(any_pages[k]) > 0.3 * len(by_page) and len(t) < 80:
            return "repeated_line"
        if big(l):
            return None                          # headings (also when printed over a picture)
        if CAPTION.match(t) and len(t) < 90:
            return "caption"
        if body and l["size"] and l["size"] < 0.85 * body and short_block:
            return "small_font_label"
        if short_block and in_figure(l):
            return "label_in_image"
        return None

    # --- 3. diagram blocks: many short lines, no sentences (flattened mind maps, charts)
    diagram = set()
    for k, ls in blocks.items():
        if ls[0]["extraction"] == "ocr" or len(ls) < 4:
            continue
        lens = [len(l["text"].strip()) for l in ls]
        ends = sum(l["text"].rstrip().endswith(SENT_END) for l in ls)
        if statistics.median(lens) < 18 and max(lens) < 60 and ends <= 1:
            diagram.add(k)

    kept = []
    for l in lines:
        r = "diagram_block" if (l["page"], l["block"]) in diagram and not big(l) else drop_reason(l)
        if not r and kept and kept[-1]["page"] == l["page"] and kept[-1]["text"].strip() == l["text"].strip():
            r = "duplicate_line"                      # the same text printed twice (shadow / outline effect)
        if r:
            stats[r] += 1
        else:
            kept.append(l)

    # --- 4. headings. Chapter openers: a big stand-alone number (01, 2, 13) printed next to the chapter
    # title; the title = the largest non-number lines on that page. Fallback (no such pages, or
    # chapter_min_size set in the config): every heading at least chapter_min_size points is a chapter.
    def is_number(l):
        return bool(NUM_ONLY.fullmatch(l["text"].strip()))

    chapter_no, title_ids = {}, set()
    if body:
        for l in kept:
            m = NUM_ONLY.fullmatch(l["text"].strip()) or NUM_TITLE.fullmatch(l["text"].strip())
            if m and l["size"] and l["size"] >= 1.5 * body:
                chapter_no.setdefault(l["page"], int(m[1]))
        chapter_no = consecutive_run(chapter_no)
        chapter_no.update({int(p): n for p, n in (cfg.get("chapter_pages") or {}).items()})   # manual fixes
        for p in chapter_no:
            cands = [l for l in kept if l["page"] == p and l["size"] and l["size"] >= 1.3 * body and not is_number(l)]
            if cands:
                top = max(l["size"] for l in cands)
                title_ids |= {id(l) for l in cands if l["size"] >= 0.95 * top}
    chap = cfg.get("chapter_min_size")
    by_number = chap is None and len(chapter_no) >= 3
    head_sizes = collections.Counter(l["size"] for l in kept if body and l["size"] and l["size"] >= 1.15 * body)
    if chap is None and not by_number and body:
        big = [s for s in head_sizes if s >= 1.4 * body and len({l["page"] for l in kept if l["size"] == s}) >= 3]
        chap = max(big) if big else None
    stats["body_size"], stats["chapter_min_size"] = body, chap
    stats["chapter_detection"] = "number" if by_number else ("size" if chap else "none")
    stats["heading_sizes"] = {str(s): n for s, n in sorted(head_sizes.items())}

    # front matter = everything before the first chapter (title page, credits, preface, contents)
    if by_number:
        first = min(chapter_no)
    else:
        first = min((l["page"] for l in kept if chap and l["size"] and l["size"] >= chap), default=None)
    if first and first <= 20:
        n0 = len(kept)
        kept = [l for l in kept if l["page"] >= first]
        stats["front_matter_lines"] = n0 - len(kept)
    elif first:
        stats["warning"] = (f"first chapter found on page {first}: the text before it is kept as chapter 0; "
                            "add the missed chapter pages to chapter_pages in the config")
    else:
        stats["warning"] = "no chapters found (OCR'd book?): the whole book is one chapter; set chapter_pages"

    def heading_level(l):
        t = l["text"].strip()
        if by_number and id(l) in title_ids:
            return 1
        if by_number and l["page"] in chapter_no and is_number(l) and l["size"] and l["size"] >= 1.5 * body:
            return -1                                  # the chapter number itself: folded into the title
        if not body or not l["size"] or len(t) > 120 or t.endswith((".", "?", ",")):
            return 0
        if not by_number and chap and l["size"] >= chap:
            return 1
        if l["size"] >= 1.15 * body or (l["bold"] and len(t) < 80 and not t.endswith(SENT_END)):
            return 2
        return 0

    # --- 5. build items: merge multi-line headings, re-join wrapped lines into paragraphs
    items, buf, prev = [], None, None
    pending_no = None

    def flush():
        nonlocal buf
        if buf:
            items.append(buf)
        buf = None

    for l in kept:
        t = LEADER.sub(" ______ ", l["text"]).strip()
        t = BULLET.sub("• ", t)
        lvl = heading_level(l)
        if lvl == -1:
            continue
        if lvl:
            flush()
            if lvl == 1 and not by_number and re.fullmatch(r"\d{1,2}\.?", t):   # big number printed apart
                pending_no = t.rstrip(".")
                continue
            kind = f"h{lvl}"
            last = items[-1] if items else None
            same_heading = (last and last["kind"] == kind and prev is not None and prev["page"] == l["page"]
                            and heading_level(prev) == lvl and (lvl == 1 or prev["block"] == l["block"]))
            if same_heading:
                last["text"] += " " + t                                   # heading wrapped on 2 lines
            elif any(i["kind"].startswith("h") and i["text"].endswith(t) for i in items[-3:]):
                stats["duplicate_heading"] += 1
            else:
                no = chapter_no.get(l["page"]) if by_number else pending_no
                if kind == "h1" and no and not t.startswith(str(no)):
                    t = f"{no}. {t}"
                pending_no = None
                items.append({"kind": kind, "page": l["page"], "text": t, "extraction": l["extraction"]})
        else:
            new_para = (buf is None or BULLET.match(l["text"]) or NUMBERED.match(t)
                        or (prev is not None and (prev["page"], prev["block"]) != (l["page"], l["block"])
                            and buf["text"].rstrip().endswith(SENT_END)))
            if new_para:
                flush()
                buf = {"kind": "p", "page": l["page"], "text": t, "extraction": l["extraction"]}
            else:
                if re.search(r"[\u0D80-\u0DFF]-$", buf["text"]) and SINHALA_START.match(t):
                    buf["text"] = buf["text"][:-1] + t                   # word hyphenated across lines
                else:
                    buf["text"] += " " + t
        prev = l
    flush()

    # --- 6. normalise + language filter
    out = []
    for it in items:
        it["text"] = normalize(repair_sinhala(it["text"]))
        if len(it["text"]) < 3:
            stats["too_short"] += 1
            continue
        if sinhala_ratio(it["text"]) < 0.5:
            stats["not_sinhala"] += 1
            continue
        n = sum(c in NOISE_CHARS for c in it["text"])
        if n >= 5 and n >= 0.02 * len(it["text"]):
            stats["notation_noise_paragraph"] += 1    # noise spread over lines that each passed the line check
            continue
        out.append(it)
    stats["items_out"] = len(out)
    stats["chars_out"] = sum(len(i["text"]) for i in out)
    return out


def to_markdown(items):
    md, page = [], None
    for it in items:
        if it["page"] != page:
            page = it["page"]
            md.append(f"<!-- p.{page} -->")
        md.append({"h1": "# ", "h2": "## ", "p": ""}[it["kind"]] + it["text"])
    return "\n\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="easy", choices=LEVEL_GRADES)
    ap.add_argument("--books", nargs="*")
    args = ap.parse_args()
    stdout_utf8()

    base = level_dir(args.level)
    cfg = read_json(CONFIG_DIR / f"books_{args.level}.json", {})
    src = sorted((base / "01_extracted").glob("*.jsonl"))
    if not src:
        raise SystemExit("Run s02_extract.py first")
    out = base / "02_clean"
    all_stats = read_json(out / "clean_stats.json", {})
    for path in src:
        bid = path.stem
        if args.books and bid not in args.books:
            continue
        stats = collections.Counter()
        items = clean_book(list(read_jsonl(path)), cfg.get(bid, {}), stats)
        write_jsonl(out / f"{bid}.jsonl", items)
        (out / f"{bid}.txt").write_text(to_markdown(items), encoding="utf-8", newline="\n")
        all_stats[bid] = dict(stats)
        h1 = sum(i["kind"] == "h1" for i in items)
        print(f"{bid:28s} items={len(items):5d} chars={stats['chars_out']:8d} chapters(h1)={h1:3d} "
              f"body={stats['body_size']} h1>={stats['chapter_min_size']} skipped_pages={len(stats['skipped_pages'])}")
        if h1 == 0 and items:
            print(f"   no chapter headings found - set chapter_pages (or chapter_min_size) for {bid} in config "
                  f"(heading sizes: {stats['heading_sizes']})")
        elif stats.get("warning") and items:
            print(f"   {stats['warning']}")
    write_json(out / "clean_stats.json", all_stats)
    print(f"\nNow open {out}/*.txt next to the PDFs (one chapter per book) before going on.")


if __name__ == "__main__":
    main()
