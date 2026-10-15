"""Step 10 — build the training mixture, render the training text, and split train / val.

Reads   <lvl>/04_dedup/textbook.jsonl for every level in mixture.json "textbook_levels" (cumulative runs)
        <level>/04_dedup/<replay source>.jsonl
        scripts/cpt/config/mixture.json
Writes  <level>/mix/train.jsonl, <level>/mix/val.jsonl   rows {"id", "source", "parent_id", "epoch", "text", "n_tokens"}
        <level>/mix/summary.json                         tokens and share per source

    python scripts/cpt/s09_mix.py --level easy

* Validation = whole textbook chapters (stratified by subject) + a small slice of each replay source,
  chosen BEFORE packing, so no validation text can end up inside a training block.
* Textbooks are repeated `textbook_epochs` times. The stream is built as epoch segments: each segment has
  one copy of every training textbook section + 1/E of the replay data (replay is never repeated).
* Every textbook section gets a Sinhala header (grade | subject | lesson | section), replay text does not.
"""
import argparse
import collections
import math
import random

from common import CONFIG_DIR, LEVEL_GRADES, SEED, count_tokens, level_dir, read_json, read_jsonl, stdout_utf8, write_json, write_jsonl


def header(d):
    parts = [f"ශ්\u200Dරේණිය: {d['grade']}", f"විෂය: {d['subject_si']}"]
    if d.get("chapter"):
        parts.append(f"පාඩම {d['chapter_no']}: {d['chapter']}")
    if d.get("section") and d["section"] != d.get("chapter"):
        parts.append(d["section"])
    return " | ".join(parts)


def split_chapters(docs, share, rng):
    """Pick validation chapters: per subject, round(share * chapters) and at least one."""
    by_subj = collections.defaultdict(set)
    for d in docs:
        by_subj[d["subject"]].add(d["parent_id"])
    val = set()
    for subj, chs in sorted(by_subj.items()):
        chs = sorted(chs)
        k = max(1, round(share * len(chs))) if len(chs) > 3 else 0
        val.update(rng.sample(chs, k))
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="easy", choices=LEVEL_GRADES)
    args = ap.parse_args()
    stdout_utf8()

    mix = read_json(CONFIG_DIR / "mixture.json")[args.level]
    rng = random.Random(f"{SEED}-mix-{args.level}")
    base = level_dir(args.level)

    textbooks = []
    for lvl in mix["textbook_levels"]:
        p = level_dir(lvl) / "04_dedup" / "textbook.jsonl"
        if not p.exists():
            raise SystemExit(f"{p} is missing: run s01-s08 for level {lvl!r} first")
        textbooks += list(read_jsonl(p))
    val_ch = split_chapters(textbooks, mix["val_textbook_chapter_share"], rng)

    train, val = [], []
    tb_train = []
    for d in textbooks:
        text = f"{header(d)}\n\n{d['text']}" if mix.get("header", True) else d["text"]
        row = {"id": d["id"], "source": "textbook", "parent_id": d["parent_id"], "text": text}
        (val if d["parent_id"] in val_ch else tb_train).append(row)

    E = mix["textbook_epochs"]
    replay_train = []
    for name, budget in mix["replay_tokens"].items():
        p = base / "04_dedup" / f"{name}.jsonl"
        if not p.exists():
            raise SystemExit(f"{p} is missing: run s07 and s08 (or remove {name} from mixture.json)")
        docs = list(read_jsonl(p))
        rng.shuffle(docs)
        n_val = max(1, math.ceil(mix["val_replay_share"] * len(docs)))
        for d in docs[:n_val]:
            val.append({"id": d["id"], "source": name, "parent_id": d["parent_id"], "text": d["text"]})
        got = 0
        for d in docs[n_val:]:
            if got >= budget:
                break
            replay_train.append({"id": d["id"], "source": name, "parent_id": d["parent_id"], "text": d["text"]})
            got += d["n_tokens"]
        if got < 0.95 * budget:
            print(f"WARNING {name}: only {got:,} of {budget:,} tokens available - raise its budget in s07")

    rng.shuffle(replay_train)
    for e in range(E):
        for r in tb_train:
            train.append({**r, "id": f"{r['id']}#e{e}", "epoch": e})
        for r in replay_train[e::E]:
            train.append({**r, "epoch": e})
    for r in val:
        r["epoch"] = 0

    for rows in (train, val):
        for r, n in zip(rows, count_tokens([r["text"] for r in rows])):
            r["n_tokens"] = n
    write_jsonl(base / "mix" / "train.jsonl", train)
    write_jsonl(base / "mix" / "val.jsonl", val)

    summary = {}
    for split, rows in (("train", train), ("val", val)):
        tok = collections.Counter()
        for r in rows:
            tok[r["source"]] += r["n_tokens"]
        total = sum(tok.values())
        summary[split] = {s: {"tokens": t, "share": round(t / total, 3)} for s, t in tok.most_common()}
        summary[split]["_total_tokens"] = total
        print(f"\n{split}: {len(rows):,} documents, {total:,} tokens")
        for s, t in tok.most_common():
            print(f"  {s:16s} {t:12,d}  {t / total:6.1%}")
    summary["val_chapters"] = sorted(val_ch)
    summary["textbook_epochs"] = E
    write_json(base / "mix" / "summary.json", summary)


if __name__ == "__main__":
    main()
