"""Data for CPT training (docs/prepare_cpt_training_pipeline.md, sections 2, 4.1, 4.2 and 8.2).

- Train: the packed Parquet shards of scripts/cpt/s10_pack.py, read IN ORDER (the three epoch segments stay in
  sequence, so the epoch-boundary steps mean "after 1 / 2 / 3 textbook epochs").
- Validation: mix/val.jsonl tokenized and packed PER SOURCE, so the Trainer logs eval_textbook_loss, eval_wiki_loss,
  eval_fineweb2_loss and eval_english_loss separately.
- Collator: labels = input_ids. Never DataCollatorForLanguageModeling: pad == eos == <|endoftext|> is the document
  separator here, and that collator would set every separator's label to -100.
"""
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch

from probe_eval import read_jsonl   # same JSONL reader as the dataset pipeline (scripts/cpt/common.py)

# mix/val.jsonl `source` -> eval set name (section 8.2)
VAL_SOURCES = {"textbook": "textbook", "wikipedia_si": "wiki", "fineweb2_si": "fineweb2", "fineweb_edu_en": "english"}


def load_train(data_dir: Path, seq_len: int, limit_per_epoch: int | None = None):
    """(dataset with one column input_ids, end index of each epoch segment), rows in file order.
    `limit_per_epoch` keeps the first N blocks of each segment and `seq_len` < 2048 truncates every block:
    both only for dry runs, so that a tiny run still crosses every epoch boundary."""
    from datasets import Dataset
    files = sorted((data_dir / "packed" / "train").glob("train-*.parquet"))
    if not files:
        raise SystemExit(f"no train-*.parquet in {data_dir / 'packed' / 'train'}: run scripts/cpt/s10_pack.py")
    ds = Dataset.from_parquet([str(f) for f in files])
    epochs = ds.data.column("epoch").to_pylist()
    if any(b < a for a, b in zip(epochs, epochs[1:])):
        raise SystemExit("the epoch column is not in order: the shards were not written by s10_pack.py in sequence")
    if limit_per_epoch:
        seen, keep = Counter(), []
        for i, e in enumerate(epochs):
            if seen[e] < limit_per_epoch:
                keep.append(i)
                seen[e] += 1
        ds, epochs = ds.select(keep), [epochs[i] for i in keep]
    block = len(ds[0]["input_ids"])
    if seq_len > block:
        raise SystemExit(f"seq_len {seq_len} is longer than the packed blocks ({block})")
    if seq_len < block:
        ds = ds.map(lambda b: {"input_ids": [x[:seq_len] for x in b["input_ids"]]}, batched=True)
    segment_ends = [i + 1 for i in range(len(epochs)) if i + 1 == len(epochs) or epochs[i + 1] != epochs[i]]
    return ds.select_columns(["input_ids"]), segment_ends


def boundary_steps(segment_ends: list[int], global_batch: int) -> dict[int, str]:
    """{optimizer step: 'e1', 'e2', ...}: the step whose batch holds the last block of each epoch segment.
    For Easy with 32 sequences per step: 3,744 / 7,474 / 11,181 blocks -> steps 117 / 234 / 350. The last one is
    the final step, which equals the Trainer's step count: ceil(ceil(N / micro) / accumulation) = ceil(N / global)."""
    return {math.ceil(end / global_batch): f"e{k}" for k, end in enumerate(segment_ends, start=1)}


def build_val_sets(val_jsonl: Path, tokenizer, seq_len: int, eos_id: int, limit_blocks: int | None = None) -> dict:
    """{name: Dataset(input_ids, labels)} per source. Documents are joined with <|endoftext|> exactly as in s10_pack.py
    and cut into seq_len blocks. The remainder is not dropped: the last block is the final seq_len tokens of the
    stream, with labels -100 on the part the previous block already scored, so every token counts once."""
    from datasets import Dataset
    texts = defaultdict(list)
    for r in read_jsonl(val_jsonl):
        texts[r["source"]].append(r["text"])
    unknown = set(texts) - set(VAL_SOURCES)
    if unknown:
        raise SystemExit(f"unknown sources in {val_jsonl}: {sorted(unknown)}: add them to VAL_SOURCES")
    sets = {}
    for source, name in VAL_SOURCES.items():
        if source not in texts:
            continue
        stream = []
        for ids in tokenizer(texts[source], add_special_tokens=False)["input_ids"]:
            stream += ids + [eos_id]
        if len(stream) < seq_len:
            print(f"WARNING: validation source {source} has only {len(stream)} tokens (< {seq_len}): skipped")
            continue
        n_full = len(stream) // seq_len
        rows = [{"input_ids": stream[k * seq_len:(k + 1) * seq_len]} for k in range(n_full)]
        for r in rows:
            r["labels"] = r["input_ids"]
        rest = len(stream) - n_full * seq_len
        if rest:
            last = stream[-seq_len:]
            rows.append({"input_ids": last, "labels": [-100] * (seq_len - rest) + last[seq_len - rest:]})
        if limit_blocks:
            rows = rows[:limit_blocks]
        sets[name] = Dataset.from_list(rows)
        print(f"val {name:<9} {len(stream):>8,} tokens -> {len(rows)} blocks")
    return sets


def collate(features: list[dict]) -> dict:
    """Stack fixed-length blocks; labels = input_ids unless the row brings its own (the last validation block).
    No attention mask: every row is exactly seq_len real tokens."""
    input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    if "labels" in features[0]:
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
    else:
        labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels}


def check_dataset(manifest: dict, train_ds, tokenizer, full_data: bool) -> None:
    """Fail early if the packed data and the tokenizer don't belong together."""
    if tokenizer.eos_token_id != manifest["eos_id"]:
        raise SystemExit(f"tokenizer eos id {tokenizer.eos_token_id} != dataset eos id {manifest['eos_id']} "
                         f"({manifest['tokenizer']}): wrong model or tokenizer")
    if full_data and len(train_ds) != manifest["train"]["blocks"]:
        raise SystemExit(f"{len(train_ds)} train blocks, manifest.json says {manifest['train']['blocks']}")
    first = train_ds[0]["input_ids"]
    n_eos = first.count(manifest["eos_id"])
    print(f"train: {len(train_ds):,} blocks x {len(first)} tokens; block 0 holds {n_eos} document separators; "
          f"pad_token == eos_token: {tokenizer.pad_token_id == tokenizer.eos_token_id} (expected, labels keep it)")
