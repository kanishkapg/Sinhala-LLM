"""Build the MCQ probes used during CPT (docs/prepare_cpt_training_pipeline.md, section 8.3).

Reads   data/SinhalaMMLU/TEST/easy/*.json                  Probe A source (public Easy questions, gold answers)
        data/SinhalaMMLU-SharedTask-Test/*/*.json          only to leave out Probe A questions that are also in it
        data/probes/easy/probe_b_authored/*.jsonl          Probe B questions as written, one row per question:
            {"source_id", "question", "answer": correct text, "distractors": [3 texts], "evidence": quote from the section}
        data/processed/cpt/easy/04_dedup/textbook.jsonl    the source sections (grade, subject)
Writes  data/probes/easy/probe_a_full.jsonl      every usable Probe A question       (epoch boundaries)
        data/probes/easy/probe_a_subset.jsonl    SUBSET_PER_SUBJECT per subject      (every 50 steps)
        data/probes/easy/probe_a_excluded.jsonl  what was left out of Probe A, and why
        data/probes/easy/probe_b.jsonl           Probe B, split "train" / "heldout"
        data/probes/easy/probe_b_review.csv      a random sample for the Sinhala reader (opens in Excel)

    python scripts/probes/build_probes.py

Every output row has the SinhalaMMLU fields (question, choices, answer as an int 1-4, metadata.subject_original)
plus "id" and "probe", so probe_common.build_prompt and the same scoring code work for both probes.
Probe B answer positions are balanced: within each split, each of 1-4 is the gold answer ~25% of the time.
"""
import csv
import random
from collections import Counter, defaultdict

from probe_common import (AUTHORED_DIR, PROBE_A_GLOB, PROBE_DIR, SEED, SHARED_TEST_GLOB, SUBJECT_MAP, TEXTBOOKS,
                          MIX_SUMMARY, load_json_items, near_duplicates, norm, read_json, read_jsonl, stdout_utf8,
                          write_jsonl)

SUBSET_PER_SUBJECT = 36  # x 14 subjects = 504
REVIEW_SAMPLE = {"train": 40, "heldout": 20}
# Gold answers stored as the option TEXT instead of its number: {(subject, q_no): option text}.
ANSWER_IS_TEXT = {("Geography", 64): "24"}


def build_probe_a():
    items = load_json_items(PROBE_A_GLOB)
    shared = load_json_items(SHARED_TEST_GLOB)
    excluded = {}
    for i, j, ratio, n in near_duplicates(items, shared):
        t = shared[j]
        excluded[i] = {"reason": "near-duplicate of a shared-task test question", "question_similarity": ratio,
                       "shared_options": n, "test_level": t["metadata"].get("difficulty"),
                       "test_subject": t["subject"], "test_q_no": t["q_no"]}

    # The "subject" field is spelled inconsistently inside some files ("Eastern_music" / "Eastern music",
    # "drama and Theatre "); subject_original is consistent, so name the subject from it.
    canonical = {si: (subject, category) for subject, category, si in SUBJECT_MAP.values()}
    full = []
    for i, it in enumerate(items):
        subject, category = canonical[it["metadata"]["subject_original"]]
        key = (subject, it["q_no"])
        row = {"id": f"A-{subject}-{it['q_no']:03d}", "probe": "A", **it, "subject": subject, "category": category}
        if key in ANSWER_IS_TEXT:
            row["answer"] = [c.strip() for c in it["choices"]].index(ANSWER_IS_TEXT[key]) + 1
            row["metadata"] = {**it["metadata"], "answer_fixed": f"gold was the option text {it['answer']!r}"}
        row["answer"] = int(row["answer"])
        if i not in excluded and len({norm(c) for c in it["choices"]}) < len(it["choices"]):
            excluded[i] = {"reason": "two options are identical"}
        if i in excluded:
            excluded[i] = {"id": row["id"], **excluded[i], "question": it["question"]}
        else:
            full.append(row)

    rng = random.Random(f"{SEED}-probe-a-subset")
    by_subj = defaultdict(list)
    for row in full:
        by_subj[row["subject"]].append(row)
    keep = set()
    for subj in sorted(by_subj):
        keep.update(r["id"] for r in rng.sample(by_subj[subj], min(SUBSET_PER_SUBJECT, len(by_subj[subj]))))
    subset = [r for r in full if r["id"] in keep]

    write_jsonl(PROBE_DIR / "probe_a_full.jsonl", full)
    write_jsonl(PROBE_DIR / "probe_a_subset.jsonl", subset)
    write_jsonl(PROBE_DIR / "probe_a_excluded.jsonl", excluded.values())
    reasons = Counter(e["reason"] for e in excluded.values())
    print(f"Probe A: {len(items)} questions, {len(excluded)} left out ({dict(reasons)}) "
          f"-> full {len(full)}, subset {len(subset)}")


def build_probe_b():
    sections = {r["id"]: r for r in read_jsonl(TEXTBOOKS)}
    heldout = set(read_json(MIX_SUMMARY)["val_chapters"])
    authored = []
    for path in sorted(AUTHORED_DIR.glob("*.jsonl")):
        authored += [{**r, "_file": path.name} for r in read_jsonl(path)]
    if not authored:
        raise SystemExit(f"no Probe B questions in {AUTHORED_DIR}")

    rows = []
    for r in authored:
        sec = sections.get(r["source_id"])
        if sec is None:
            raise SystemExit(f"{r['_file']}: unknown source_id {r['source_id']!r}")
        if len(r["distractors"]) != 3:
            raise SystemExit(f"{r['_file']}: {r['question'][:40]!r} needs exactly 3 distractors")
        subject, category, subject_si = SUBJECT_MAP[sec["subject"]]
        split = "heldout" if sec["parent_id"] in heldout else "train"
        rows.append({"probe": "B", "split": split, "subject": subject, "category": category,
                     "question": r["question"].strip(), "_answer": r["answer"].strip(),
                     "_distractors": [d.strip() for d in r["distractors"]],
                     "metadata": {"subject_original": subject_si, "difficulty": "easy", "grade": sec["grade"],
                                  "source_id": sec["id"], "evidence": r["evidence"].strip(),
                                  "author": r.get("author", "claude-opus-5-5")}})

    rng = random.Random(f"{SEED}-probe-b-options")
    out = []
    for split in ("train", "heldout"):
        part = sorted((r for r in rows if r["split"] == split), key=lambda r: (r["subject"], r["metadata"]["source_id"]))
        golds = [1 + k % 4 for k in range(len(part))]  # balanced gold positions, shuffled
        rng.shuffle(golds)
        counter = Counter()
        for r, gold in zip(part, golds):
            distractors = r.pop("_distractors")
            rng.shuffle(distractors)
            choices = distractors[: gold - 1] + [r.pop("_answer")] + distractors[gold - 1:]
            counter[r["subject"]] += 1
            out.append({"id": f"B-{split}-{r['subject']}-{counter[r['subject']]:03d}", **r,
                        "choices": choices, "answer": gold})
    out = [{k: r[k] for k in ("id", "probe", "split", "subject", "category", "question", "choices", "answer", "metadata")}
           for r in out]
    write_jsonl(PROBE_DIR / "probe_b.jsonl", out)

    sample = []
    for split, n in REVIEW_SAMPLE.items():
        part = [r for r in out if r["split"] == split]
        sample += random.Random(f"{SEED}-probe-b-review-{split}").sample(part, min(n, len(part)))
    with open(PROBE_DIR / "probe_b_review.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "subject", "question", "option 1", "option 2", "option 3", "option 4", "gold",
                    "evidence (from the textbook)", "source_id",
                    "VERDICT: ok / wrong answer / unclear / bad Sinhala / answerable without the textbook", "note"])
        for r in sample:
            w.writerow([r["id"], r["split"], r["subject"], r["question"], *r["choices"], r["answer"],
                        r["metadata"]["evidence"], r["metadata"]["source_id"], "", ""])
    n = Counter(r["split"] for r in out)
    print(f"Probe B: {len(out)} questions (train {n['train']}, heldout {n['heldout']}); "
          f"review sample {len(sample)} -> probe_b_review.csv")


def main():
    stdout_utf8()
    build_probe_a()
    build_probe_b()


if __name__ == "__main__":
    main()
