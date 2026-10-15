"""Steps 11+12 — tokenize, pack into fixed-length blocks, write Parquet shards, run final checks, write manifest.

Reads   <level>/mix/train.jsonl, <level>/mix/val.jsonl  (+ the reports of the earlier steps for the manifest)
Writes  <level>/packed/train/train-00000.parquet ...   column input_ids: list<int32> of exactly --block tokens
        <level>/packed/val/val-00000.parquet ...       (+ column epoch in train)
        <level>/manifest.json

    python scripts/cpt/s10_pack.py --level easy --block 2048

Format: raw text + one <|endoftext|> after every document, no BOS (Qwen3.5 has none), no chat template.
Documents are shuffled (fixed seed) inside each epoch segment, concatenated, and cut into blocks; the
remainder (< one block) at the very end is dropped. Labels = input_ids (the trainer's collator copies them).
"""
import argparse
import collections
import datetime
import random

from common import (CONFIG_DIR, EOS_TOKEN, LEVEL_GRADES, PIPELINE_VERSION, ROOT, SEED, TOKENIZER_ID, level_dir,
                    read_json, read_jsonl, sha1, stdout_utf8, tokenizer, tokenizer_hash, write_json)
from fm_converter import PATCHES

SHARD_BLOCKS = 2000          # ~16 MB per shard; keeps memory low while writing


def pack(rows, block, eos, rng, shuffle):
    """Yield (input_ids, epoch) blocks; rows are grouped by epoch and shuffled inside each epoch."""
    tok = tokenizer()
    by_epoch = collections.defaultdict(list)
    for r in rows:
        by_epoch[r.get("epoch", 0)].append(r)
    buf, buf_epoch = [], 0
    for e in sorted(by_epoch):
        docs = by_epoch[e]
        if shuffle:
            rng.shuffle(docs)
        for i in range(0, len(docs), 1000):
            batch = docs[i:i + 1000]
            for enc in tok.encode_batch([d["text"] for d in batch], add_special_tokens=False):
                if not buf:
                    buf_epoch = e
                buf.extend(enc.ids)
                buf.append(eos)
                while len(buf) >= block:
                    yield buf[:block], buf_epoch
                    buf = buf[block:]
                    buf_epoch = e


def write_shards(blocks, out_dir, prefix, with_epoch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{prefix}-*.parquet"):
        old.unlink()
    n_blocks, shard, ids, eps = 0, 0, [], []

    def flush():
        nonlocal shard, ids, eps
        if not ids:
            return
        cols = {"input_ids": pa.array(ids, type=pa.list_(pa.int32()))}
        if with_epoch:
            cols["epoch"] = pa.array(eps, type=pa.int16())
        pq.write_table(pa.table(cols), out_dir / f"{prefix}-{shard:05d}.parquet", compression="zstd")
        shard += 1
        ids, eps = [], []

    for b, e in blocks:
        ids.append(b)
        eps.append(e)
        n_blocks += 1
        if len(ids) == SHARD_BLOCKS:
            flush()
    flush()
    return n_blocks, shard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="easy", choices=LEVEL_GRADES)
    ap.add_argument("--block", type=int, default=2048, help="2048, or 4096 if GPU memory allows")
    args = ap.parse_args()
    stdout_utf8()
    import pyarrow.parquet as pq

    base = level_dir(args.level)
    train = list(read_jsonl(base / "mix" / "train.jsonl"))
    val = list(read_jsonl(base / "mix" / "val.jsonl"))
    eos = tokenizer().token_to_id(EOS_TOKEN)
    rng = random.Random(f"{SEED}-pack-{args.level}")

    # ---- Step 12 check 3 before packing: no validation chapter/document inside train
    leak = {r["parent_id"] for r in val} & {r["parent_id"] for r in train}
    if leak:
        raise SystemExit(f"validation parent_ids found in train: {sorted(leak)[:5]}")

    n_train, s_train = write_shards(pack(train, args.block, eos, rng, True), base / "packed" / "train", "train", True)
    n_val, s_val = write_shards(pack(val, args.block, eos, rng, False), base / "packed" / "val", "val", False)
    print(f"train: {n_train:,} blocks x {args.block} = {n_train * args.block:,} tokens in {s_train} shards")
    print(f"val:   {n_val:,} blocks x {args.block} = {n_val * args.block:,} tokens in {s_val} shards")

    # ---- Step 12 check 1: decode a few random blocks (read them!)
    tok = tokenizer()
    first = pq.read_table(sorted((base / "packed" / "train").glob("*.parquet"))[0]).column("input_ids").to_pylist()
    for k in rng.sample(range(len(first)), min(3, len(first))):
        ids = first[k]
        assert len(ids) == args.block
        text = tok.decode(ids, skip_special_tokens=False)
        print(f"\n--- block {k}: {ids.count(eos)} EOS ---\n{text[:400]} ...")

    # ---- manifest
    mix_summary = read_json(base / "mix" / "summary.json", {})
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "level": args.level,
        "seed": SEED,
        "tokenizer": TOKENIZER_ID,
        "tokenizer_json_sha1": tokenizer_hash(),
        "eos_token": EOS_TOKEN, "eos_id": eos, "bos": None,
        "block_size": args.block,
        "train": {"blocks": n_train, "tokens": n_train * args.block, "shards": s_train},
        "val": {"blocks": n_val, "tokens": n_val * args.block, "shards": s_val},
        "mixture": read_json(CONFIG_DIR / "mixture.json")[args.level],
        "mix_summary": mix_summary,
        "docs_jsonl_sha1": sha1((base / "docs.jsonl").read_text(encoding="utf-8")) if (base / "docs.jsonl").exists() else None,
        "inventory": [{k: b[k] for k in ("book_id", "file", "subject", "grade", "pages", "scanned")}
                      for b in read_json(base / "00_inventory" / "inventory.json", [])],
        "font_config": read_json(CONFIG_DIR / "fonts.json"),
        "converter": {"base": "Open-SL fm_abaya_to_unicode (MIT)", "patches": PATCHES},
        "books_config": read_json(CONFIG_DIR / f"books_{args.level}.json", {}),
        "ocr": {"tool": "Tesseract 5, tessdata_best sin, --psm 4, 300 dpi",
                "books": [b for b, c in read_json(CONFIG_DIR / f"books_{args.level}.json", {}).items()
                          if c.get("ocr") or c.get("ocr_pages")]},
        "review": read_json(base / "03_review" / "review_status.json", {}),
        "cleaning": read_json(base / "02_clean" / "clean_stats.json", {}),
        "replay_collection": read_json(base / "replay" / "stats.json", {}),
        "dedup_decontamination": read_json(base / "04_dedup" / "report.json", {}),
        "closed_api_use": "none (edit this if a closed model/API was used for OCR or data building - it must be disclosed)",
    }
    write_json(base / "manifest.json", manifest)
    print(f"\nmanifest -> {base / 'manifest.json'}")
    print(f"train-ready dataset: {(base / 'packed').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
