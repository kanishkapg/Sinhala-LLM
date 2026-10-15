"""Step 1 — inventory the textbooks of one level and check fonts, scanned pages and subject coverage.

Writes  data/processed/cpt/<level>/00_inventory/inventory.json  (+ .csv)
Creates scripts/cpt/config/books_<level>.json if missing (per-book settings you may edit later).

    python scripts/cpt/s01_inventory.py --level easy

Stop and act if the report shows:
  * UNKNOWN FONTS  -> add each one to config/fonts.json (convert / keep / drop), then re-run.
  * MISSING SUBJECTS -> download the textbooks (edupub.gov.lk) into data/raw/gN/, then re-run.
"""
import argparse
import collections
import csv
import glob
import json

import pymupdf

from common import (CONFIG_DIR, LEVEL_GRADES, ROOT, SUBJECTS, TEST_GLOBS, book_info, level_dir, list_pdfs,
                    read_json, stdout_utf8, write_json)
from fm_converter import font_action, font_name

FLAGS = pymupdf.TEXTFLAGS_TEXT | pymupdf.TEXT_CID_FOR_UNKNOWN_UNICODE
LOW_CHARS = 100          # a page with fewer extractable characters is (probably) an image
SCANNED_SHARE = 0.5      # a book with more low pages than this is treated as scanned -> OCR


def scan_book(pdf):
    info = book_info(pdf)
    doc = pymupdf.open(pdf)
    fonts = collections.Counter()
    samples = {}
    low_pages = []
    for page in doc:
        n = 0
        for block in page.get_text("dict", flags=FLAGS)["blocks"]:
            for line in block.get("lines", []):
                for sp in line["spans"]:
                    f = font_name(sp["font"])
                    t = sp["text"].strip()
                    fonts[f] += len(t)
                    n += len(t)
                    if t and f not in samples:
                        samples[f] = f"p{page.number + 1}: {t[:50]}"
        if n < LOW_CHARS:
            low_pages.append(page.number + 1)
    info.update(pages=doc.page_count, low_text_pages=low_pages,
                scanned=len(low_pages) > SCANNED_SHARE * doc.page_count,
                fonts={f: {"chars": c, "action": font_action(f), "sample": samples.get(f, "")}
                       for f, c in fonts.most_common()})
    return info


def canonical_subject(name):
    """Test-set spellings ('Health & Physical Education', 'Eastern_Music', 'Drama') -> SUBJECTS names."""
    low = name.lower().replace("_", " ")
    for key, english in (("sinhala", "Sinhala"), ("health", "Health"), ("drama", "Drama and Theatre"),
                         ("danc", "Dancing"), ("music", "Eastern Music"), ("art", "Art")):
        if key in low:
            return english
    return name


def test_subjects(level):
    """Subjects (English names) and question counts of the shared-task test set for this level."""
    counts = collections.Counter()
    for pattern in TEST_GLOBS[:1]:
        for path in glob.glob(str(ROOT / pattern), recursive=True):
            if f"{level}" not in path.replace("\\", "/").lower():
                continue
            with open(path, encoding="utf-8") as f:
                for item in json.load(f):
                    counts[item.get("subject", "?")] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="easy", choices=LEVEL_GRADES)
    args = ap.parse_args()
    stdout_utf8()

    pdfs = list_pdfs(args.level)
    if not pdfs:
        raise SystemExit(f"No PDFs in data/raw/g{LEVEL_GRADES[args.level]}")
    books = []
    for pdf in pdfs:
        print(f"scanning {pdf.relative_to(ROOT)} ...", flush=True)
        books.append(scan_book(pdf))

    out = level_dir(args.level) / "00_inventory"
    write_json(out / "inventory.json", books)
    with open(out / "inventory.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["book_id", "file", "subject", "grade", "pages", "low_text_pages", "scanned", "top_fonts"])
        for b in books:
            w.writerow([b["book_id"], b["file"], b["subject"], b["grade"], b["pages"], len(b["low_text_pages"]),
                        b["scanned"], " ".join(list(b["fonts"])[:4])])

    # per-book settings file (never overwritten: you edit it)
    cfg_path = CONFIG_DIR / f"books_{args.level}.json"
    cfg = read_json(cfg_path, {})
    for b in books:
        cfg.setdefault(b["book_id"], {"skip_pages": [], "ocr": b["scanned"], "ocr_pages": [],
                                      "chapter_pages": {}, "chapter_min_size": None, "notes": ""})
    write_json(cfg_path, cfg)

    # ---- report
    print(f"\n{'book_id':28s} {'pages':>5s} {'low':>4s} scanned  fonts")
    for b in books:
        print(f"{b['book_id']:28s} {b['pages']:5d} {len(b['low_text_pages']):4d} {str(b['scanned']):7s}  "
              f"{', '.join(list(b['fonts'])[:3])}")

    unknown = {}
    for b in books:
        for f, v in b["fonts"].items():
            if v["action"] == "unknown" and v["chars"]:
                unknown.setdefault(f, []).append(f"{b['book_id']} ({v['chars']} chars) {v['sample']}")
    if unknown:
        print("\nUNKNOWN FONTS - classify each in scripts/cpt/config/fonts.json, then re-run:")
        for f, where in unknown.items():
            print(f"  {f}")
            for w in where[:3]:
                print(f"      {w}")

    have = {b["subject"] for b in books}
    tests = collections.Counter()
    for subj, n in test_subjects(args.level).items():
        tests[canonical_subject(subj)] += n
    print("\nCoverage of test subjects:")
    missing = []
    for subj, n in tests.most_common():
        ok = subj in have
        print(f"  {subj:32s} {n:4d} questions  {'ok' if ok else 'MISSING TEXTBOOK'}")
        if not ok:
            missing.append(subj)
    if missing:
        print(f"\nMISSING SUBJECTS: {', '.join(missing)} -> download them from "
              f"http://www.edupub.gov.lk/BooksDownload.php into data/raw/g<grade>/ "
              f"(name them like 'drama g-6 S.pdf'; known subject keys: {', '.join(SUBJECTS)})")
    print(f"\nwrote {out / 'inventory.json'} and {cfg_path}")


if __name__ == "__main__":
    main()
