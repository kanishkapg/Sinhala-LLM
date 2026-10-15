"""Step 4 — OCR scanned books / pages with Tesseract (Sinhala model) and add the lines to 01_extracted.

Which pages: every page of books with "ocr": true in config/books_<level>.json (s01 sets it for scanned
books), plus the page numbers listed in "ocr_pages" for any book.

Needs Tesseract 5 on PATH (or --tesseract C:/Program Files/Tesseract-OCR/tesseract.exe) with
sin.traineddata from https://github.com/tesseract-ocr/tessdata_best in its tessdata folder.

    python scripts/cpt/s03_ocr.py --level easy
    python scripts/cpt/s03_ocr.py --level easy --books health_g8_si --force

Page images are cached in 01_extracted/ocr_pages/<book_id>/; OCR text per page in ocr_text/.
"""
import argparse
import shutil
import subprocess

import pymupdf

from common import CONFIG_DIR, LEVEL_GRADES, ROOT, level_dir, read_json, read_jsonl, stdout_utf8, write_jsonl


def ocr_page(tess, png, lang, psm):
    res = subprocess.run([tess, str(png), "stdout", "-l", lang, "--psm", str(psm)],
                         capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        raise SystemExit(f"tesseract failed on {png}: {res.stderr.strip()[:300]}")
    return res.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="easy", choices=LEVEL_GRADES)
    ap.add_argument("--books", nargs="*")
    ap.add_argument("--tesseract", default=shutil.which("tesseract") or "tesseract")
    ap.add_argument("--lang", default="sin")
    ap.add_argument("--psm", type=int, default=4, help="4 = single column of text of variable sizes")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--force", action="store_true", help="re-OCR pages that already have cached text")
    args = ap.parse_args()
    stdout_utf8()

    base = level_dir(args.level)
    inv = {b["book_id"]: b for b in read_json(base / "00_inventory" / "inventory.json")}
    cfg = read_json(CONFIG_DIR / f"books_{args.level}.json", {})
    todo = {bid: (range(1, inv[bid]["pages"] + 1) if c.get("ocr") else c.get("ocr_pages", []))
            for bid, c in cfg.items() if bid in inv and (c.get("ocr") or c.get("ocr_pages"))}
    if args.books:
        todo = {k: v for k, v in todo.items() if k in args.books}
    if not todo:
        print("Nothing to OCR (no book has ocr=true or ocr_pages in config).")
        return
    if not shutil.which(args.tesseract):
        raise SystemExit(f"Tesseract not found ({args.tesseract}). Install it or pass --tesseract <path>.")

    for bid, pages in todo.items():
        pages = set(pages)
        doc = pymupdf.open(ROOT / inv[bid]["file"])
        img_dir = base / "01_extracted" / "ocr_pages" / bid
        txt_dir = base / "01_extracted" / "ocr_text" / bid
        img_dir.mkdir(parents=True, exist_ok=True)
        txt_dir.mkdir(parents=True, exist_ok=True)
        new_lines = {}
        for pno in sorted(pages):
            txt_path = txt_dir / f"p{pno:04d}.txt"
            if args.force or not txt_path.exists():
                png = img_dir / f"p{pno:04d}.png"
                doc[pno - 1].get_pixmap(dpi=args.dpi).save(png)
                txt_path.write_text(ocr_page(args.tesseract, png, args.lang, args.psm), encoding="utf-8")
                print(f"  {bid} p{pno} OCR done", flush=True)
            # Tesseract separates paragraphs by blank lines -> one "block" per paragraph
            recs, block = [], 0
            for raw in txt_path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    block += 1
                    continue
                recs.append({"type": "line", "page": pno, "block": block, "line": len(recs), "bbox": None,
                             "size": None, "bold": False, "fonts": ["ocr"], "raw": raw, "text": raw.strip(),
                             "extraction": "ocr"})
            new_lines[pno] = recs

        # merge into 01_extracted/<book>.jsonl, replacing the lines of the OCR'd pages
        path = base / "01_extracted" / f"{bid}.jsonl"
        old = list(read_jsonl(path)) if path.exists() else []
        merged = []
        for r in old:
            if r["type"] == "line" and r["page"] in new_lines:
                continue
            merged.append(r)
            if r["type"] == "page" and r["page"] in new_lines:
                r["n_chars"] = sum(len(x["text"]) for x in new_lines[r["page"]])
                merged.extend(new_lines[r["page"]])
        write_jsonl(path, merged)
        print(f"{bid}: {len(new_lines)} pages OCR'd -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
