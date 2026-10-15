"""Steps 2+3 — extract every text line span by span (with font, size, bbox) and convert FM spans to Unicode.

Reads   data/processed/cpt/<level>/00_inventory/inventory.json   (run s01 first)
Writes  data/processed/cpt/<level>/01_extracted/<book_id>.jsonl  one record per page and per line:
        {"type":"page", "page", "width", "height", "images":[[x0,y0,x1,y1],...], "n_chars"}
        {"type":"line", "page", "block", "line", "bbox", "size", "bold", "fonts", "raw", "text", "extraction"}
        01_extracted/leftovers.json   characters that survived conversion inside FM spans (unmapped glyphs)

    python scripts/cpt/s02_extract.py --level easy                # all books (skips books already done)
    python scripts/cpt/s02_extract.py --level easy --force --books history_g7_si   # redo one book

Scanned books (inventory "scanned": true) get page records only; s03_ocr.py adds their lines.
"""
import argparse
import collections
import json

import pymupdf

from common import LEVEL_GRADES, ROOT, level_dir, read_json, stdout_utf8, write_json, write_jsonl
from fm_converter import PATCHES, fix_line, fm2uni, font_action, font_name

FLAGS = pymupdf.TEXTFLAGS_TEXT | pymupdf.TEXT_CID_FOR_UNKNOWN_UNICODE
OK_CHARS = set(" \u200D0123456789.,:;?!()-/%'\"=+*×÷–—‘’“”[]{}•…_<>")
OK_CHARS |= set("IVXiv")     # roman-numeral list markers: FM ¸ ¹ º µ ¶ » ¼ -> I V X i v I V (intentional)
OK_CHARS |= {"Ω", chr(0x2010)}  # Ohm sign printed inside FM text in science; Unicode hyphen


def is_bold(span) -> bool:
    f = span["font"].lower()
    return bool(span["flags"] & 16) or "bold" in f or "bld" in f


def extract_book(book, leftovers):
    doc = pymupdf.open(ROOT / book["file"])
    unknown = collections.Counter()
    for page in doc:
        pno = page.number + 1
        images = [[round(v, 1) for v in im["bbox"]] for im in page.get_image_info()
                  if (im["bbox"][2] - im["bbox"][0]) * (im["bbox"][3] - im["bbox"][1]) > 2000]
        lines = []
        if not book["scanned"]:
            for block in page.get_text("dict", flags=FLAGS, sort=True)["blocks"]:
                for lno, line in enumerate(block.get("lines", [])):
                    raw, text, fonts, sizes = "", "", [], collections.Counter()
                    bold_chars = all_chars = 0
                    suspects = []                  # (char, example) from FM spans, kept if fix_line leaves them
                    for sp in line["spans"]:
                        f = font_name(sp["font"])
                        act = font_action(f)
                        if act == "drop":
                            continue
                        if act == "unknown":
                            unknown[f] += len(sp["text"])
                        conv = fm2uni(sp["text"], f) if act == "convert" else sp["text"]
                        if act == "convert":
                            for c in conv:
                                if not ("\u0D80" <= c <= "\u0DFF" or c.isspace() or c in OK_CHARS):
                                    suspects.append((c, f"p{pno}: {sp['text'][:40]} -> {conv[:40]}"))
                        raw += sp["text"]
                        text += conv
                        if f not in fonts:
                            fonts.append(f)
                        if sp["text"].strip():
                            sizes[round(sp["size"], 1)] += len(sp["text"])
                            all_chars += len(sp["text"])
                            bold_chars += len(sp["text"]) if is_bold(sp) else 0
                    text = fix_line(text)
                    for c, example in suspects:
                        if c in text:
                            leftovers[c][0] += 1
                            leftovers[c][1].setdefault(book["book_id"], example)
                    if not text.strip():
                        continue
                    bold =bold_chars >= 0.7 * all_chars > 0          # mostly bold, not one bold word
                    lines.append({"type": "line", "page": pno, "block": block["number"], "line": lno,
                                  "bbox": [round(v, 1) for v in line["bbox"]],
                                  "size": sizes.most_common(1)[0][0] if sizes else None, "bold": bold,
                                  "fonts": fonts, "raw": raw, "text": text,
                                  "extraction": "fm_convert" if any(font_action(f) == "convert" for f in fonts) else "unicode"})
        yield {"type": "page", "page": pno, "width": round(page.rect.width, 1), "height": round(page.rect.height, 1),
               "images": images, "n_chars": sum(len(l["text"]) for l in lines)}
        yield from lines
    if unknown:
        print(f"   WARNING unknown fonts kept unconverted: {dict(unknown)} -> classify them in config/fonts.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="easy", choices=LEVEL_GRADES)
    ap.add_argument("--books", nargs="*", help="only these book_ids")
    ap.add_argument("--force", action="store_true", help="re-extract books that already have output")
    args = ap.parse_args()
    stdout_utf8()

    base = level_dir(args.level)
    books = read_json(base / "00_inventory" / "inventory.json")
    if books is None:
        raise SystemExit("Run s01_inventory.py first")
    out = base / "01_extracted"
    leftovers = collections.defaultdict(lambda: [0, {}])
    for book in books:
        if args.books and book["book_id"] not in args.books:
            continue
        path = out / f"{book['book_id']}.jsonl"
        if path.exists() and not args.force:
            print(f"skip {book['book_id']} (exists; --force to redo)")
            continue
        n = write_jsonl(path, extract_book(book, leftovers))
        print(f"{book['book_id']:28s} {n:7d} records{'  (scanned -> run s03_ocr.py)' if book['scanned'] else ''}")

    found = {c: {"count": n, "where": w} for c, (n, w) in sorted(leftovers.items(), key=lambda x: -x[1][0])}
    if args.force and not args.books:          # full re-run: start fresh so fixed characters disappear
        write_json(out / "leftovers.json", found)
    elif found:                                # partial run: keep the entries of the books not redone
        write_json(out / "leftovers.json", {**read_json(out / "leftovers.json", {}), **found})
    if leftovers:
        print("\nCharacters left after conversion inside FM spans (each must be explained or fixed):")
        for c, (n, where) in sorted(leftovers.items(), key=lambda x: -x[1][0])[:25]:
            print(f"  {c!r:8s} U+{ord(c):04X} {n:6d}  e.g. {next(iter(where.values()))}")
    write_json(out / "converter_patches.json", PATCHES)


if __name__ == "__main__":
    main()
