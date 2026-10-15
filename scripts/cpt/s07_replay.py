"""Step 9 — collect replay data (general Sinhala + English) by sampling random row groups from Hugging Face.

The full corpora are huge (FineWeb-2 is ~2 TB over all languages; even its Sinhala part is 1.2 M documents),
but one CPT run only needs a few million tokens of replay. So this script never downloads whole datasets:
it opens the Sinhala parquet files remotely, reads random row groups, takes a random subset of rows from
each, filters them, and stops when the token budget of the source is reached.

Reads   scripts/cpt/config/replay_sources.json
Writes  data/processed/cpt/<level>/replay/<source>.jsonl   (same schema family as docs.jsonl)
        data/processed/cpt/<level>/replay/stats.json       documents removed by each filter

    python scripts/cpt/s07_replay.py --level easy
    python scripts/cpt/s07_replay.py --level easy --only fineweb2_si --force

Set HF_TOKEN (huggingface-cli login) for faster, rate-limit-free downloads.
"""
import argparse
import collections
import random
import re
import time

from common import (CONFIG_DIR, LEVEL_GRADES, PIPELINE_VERSION, SEED, count_tokens, level_dir, normalize,
                    read_json, sha1, sinhala_ratio, stdout_utf8, write_json, write_jsonl)

HEADROOM = 1.15                    # collect 15 % more than the budget (dedup / decontamination losses)
MIN_CHARS, MAX_CHARS = 200, 20_000
BOILERPLATE = re.compile(r"cookie|javascript|lorem ipsum|all rights reserved|subscribe|login|sign in|"
                         r"privacy policy|terms of use|click here|copyright ©", re.I)
# exam material = the source of the benchmark -> never train on it (Step 8.2)
EXAM = re.compile(r"e-?thaksalawa|past[\s_-]?papers?|model[\s_-]?papers?|marking[\s_-]?scheme|"
                  r"පසුගිය\s*(?:විභාග\s*)?ප්\u200Dරශ්න|ආදර්ශ\s*ප්\u200Dරශ්න|ප්\u200Dරශ්න\s*පත්\u200Dර|ලකුණු\s*දීමේ\s*පටිපාටිය|"
                  r"පිළිතුරු\s*පත්\u200Dර|බහුවරණ\s*ප්\u200Dරශ්න|විභාග\s*ප්\u200Dරශ්න", re.I)


def retry(fn, tries=5):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:                       # network hiccups on large remote reads
            if k == tries - 1:
                raise
            wait = 5 * 2 ** k
            print(f"   retry in {wait}s after: {str(e)[:120]}", flush=True)
            time.sleep(wait)


def split_long(text):
    """Cut very long documents (long Wikipedia articles) into <= MAX_CHARS pieces at paragraph breaks."""
    if len(text) <= MAX_CHARS:
        return [text]
    out, cur = [], ""
    for para in text.split("\n\n"):
        if cur and len(cur) + len(para) > MAX_CHARS:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        out.append(cur)
    return [p[:MAX_CHARS] for p in out]


def reject_reason(text, url, lang):
    """None if the document is kept, else the name of the first filter that removed it."""
    if EXAM.search(url or "") or EXAM.search(text[:3000]):
        return "exam_material"
    if lang == "si" and sinhala_ratio(text) < 0.7:
        return "language"
    if lang == "en" and sum(c.isascii() for c in text) < 0.9 * len(text):
        return "language"
    if len(text) < MIN_CHARS:
        return "too_short"
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) >= 5:
        if len(set(lines)) / len(lines) < 0.7:
            return "repeated_lines"
        if sum(l.endswith((".", "?", "!", "”", "\"", ":", "।")) for l in lines) / len(lines) < 0.3:
            return "few_sentences"            # menus, lists of links, tables
        if sum(len(l) for l in lines) / len(lines) < 25:
            return "short_lines"
    if len(BOILERPLATE.findall(text)) > 2:
        return "boilerplate"
    if text.count("http") > 3 * len(text) / 1000:
        return "link_heavy"
    return None


def collect(src, rng, stats):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    files = sorted(retry(lambda: fs.glob(f"datasets/{src['repo']}/{src['files']}")))
    if not files:
        raise SystemExit(f"{src['name']}: no files match {src['repo']}/{src['files']}")
    groups = []
    for f in files:
        n = retry(lambda: pq.ParquetFile(fs.open(f)).metadata.num_row_groups)
        groups += [(f, g) for g in range(n)]
    rng.shuffle(groups)
    budget = int(src["budget"] * HEADROOM)
    got, docs = 0, []
    cols = [c for c in (src["text"], src.get("url"), src.get("title")) if c]
    print(f"{src['name']}: {len(files)} files, {len(groups)} row groups, budget {budget:,} tokens", flush=True)
    for f, g in groups:
        def read():
            pf = pq.ParquetFile(fs.open(f))
            names = set(pf.schema_arrow.names)
            return pf.read_row_group(g, columns=[c for c in cols if c in names]).to_pylist()
        rows = retry(read)
        rng.shuffle(rows)
        new = []
        for row in rows[:400]:                      # a few hundred rows per group spreads the sample
            stats["seen"] += 1
            url = row.get(src.get("url") or "", None)
            for piece in split_long(normalize(row[src["text"]] or "")):
                r = reject_reason(piece, url, src["lang"])
                if r:
                    stats[r] += 1
                    continue
                new.append({"text": piece, "url": url, "title": row.get(src.get("title") or "", None)})
        for d, n in zip(new, count_tokens([d["text"] for d in new])):
            d["n_tokens"] = n
            docs.append(d)
            got += n
            if got >= budget:
                break
        print(f"   row group {f.rsplit('/', 1)[-1]}#{g}: kept {len(new):4d}  total {got:,} tokens", flush=True)
        if got >= budget:
            break
    else:
        print(f"   WARNING: source exhausted at {got:,} of {budget:,} tokens")
    stats["kept"] = len(docs)
    stats["tokens"] = got
    for i, d in enumerate(docs):
        yield {"id": f"{src['name']}/{i:07d}", "parent_id": f"{src['name']}/{i:07d}", "source": src["name"],
               "lang": src["lang"], "subject": None, "grade": None, "title": d["title"], "url": d["url"],
               "text": d["text"], "n_chars": len(d["text"]), "n_tokens": d["n_tokens"], "sha1": sha1(d["text"]),
               "license": src["license"], "pipeline_version": PIPELINE_VERSION}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="easy", choices=LEVEL_GRADES)
    ap.add_argument("--only", nargs="*", help="only these source names")
    ap.add_argument("--force", action="store_true", help="re-collect sources that already have output")
    ap.add_argument("--limit-tokens", type=int, help="cap every source at this many tokens (quick test run)")
    args = ap.parse_args()
    stdout_utf8()

    cfg = read_json(CONFIG_DIR / "replay_sources.json")
    wanted = read_json(CONFIG_DIR / "mixture.json").get(args.level, {}).get("replay_tokens", {})
    out = level_dir(args.level) / "replay"
    all_stats = read_json(out / "stats.json", {})
    for src in cfg["sources"]:
        if not src["enabled"] or (args.only and src["name"] not in args.only):
            continue
        src["budget"] = wanted.get(src["name"], src["tokens"])     # mixture.json wins over the default
        if args.limit_tokens:
            src["budget"] = min(src["budget"], args.limit_tokens)
        path = out / f"{src['name']}.jsonl"
        if path.exists() and not args.force:
            print(f"skip {src['name']} (exists; --force to redo)")
            continue
        stats = collections.Counter()
        rng = random.Random(f"{SEED}-{args.level}-{src['name']}")
        n = write_jsonl(path, collect(src, rng, stats))
        all_stats[src["name"]] = {"repo": src["repo"], "files": src["files"], "license": src["license"], **stats}
        write_json(out / "stats.json", all_stats)
        print(f"{src['name']}: {n} documents, {stats['tokens']:,} tokens; removed: "
              f"{ {k: v for k, v in stats.items() if k not in ('seen', 'kept', 'tokens')} }\n")


if __name__ == "__main__":
    main()
