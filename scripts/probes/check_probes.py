"""Check that the probe files are well formed and that Probe B measures what it should.

    python scripts/probes/check_probes.py              # data checks (seconds)
    python scripts/probes/check_probes.py --tokenizer  # + the prompt / answer-token check (needs `tokenizers`)

Exit code 1 if any ERROR. WARN lines are for a human to look at; they do not fail the check.

Format (all probe files)  valid JSON per line, unique ids, 4 distinct non-empty options, answer an int 1-4,
                          subject_original set, every subject known
Probe A                   subset is part of full; subset balanced per subject; no near-duplicate of a
                          shared-task test question is left
Probe B                   the source section exists; split matches the held-out chapter list; "train" sections
                          really are in mix/train.jsonl (3 copies) and "heldout" ones are not;
                          the evidence quote is in its source section (word for word);
                          held-out evidence does not also appear in a training section (leakage);
                          gold positions balanced; gold is not mostly the longest option;
                          no duplicate questions; no near-duplicate of a Probe A or shared-task question
"""
import argparse
import sys
from collections import Counter

from probe_common import (MIX_SUMMARY, MIX_TRAIN, PROBE_A_GLOB, PROBE_DIR, SHARED_TEST_GLOB, SUBJECT_MAP, TEXTBOOKS,
                          build_prompt, load_json_items, near_duplicates, norm, read_json, read_jsonl, stdout_utf8)

ERRORS = []


def error(msg):
    ERRORS.append(msg)
    print(f"  ERROR {msg}")


def warn(msg):
    print(f"  WARN  {msg}")


def check_format(name, rows):
    known = {s for s, _, _ in SUBJECT_MAP.values()}
    ids = Counter(r.get("id") for r in rows)
    for i, n in ids.items():
        if n > 1:
            error(f"{name}: id {i} appears {n} times")
    for r in rows:
        rid = r.get("id", "?")
        ch = r.get("choices")
        if not isinstance(ch, list) or len(ch) != 4:
            error(f"{name} {rid}: needs 4 options, has {ch!r}")
            continue
        if any(not norm(c) for c in ch):
            error(f"{name} {rid}: empty option")
        if len({norm(c) for c in ch}) < 4:
            error(f"{name} {rid}: two options are the same: {ch}")
        if type(r.get("answer")) is not int or not 1 <= r["answer"] <= 4:
            error(f"{name} {rid}: answer must be an int 1-4, is {r.get('answer')!r}")
        if not norm(r.get("question", "")):
            error(f"{name} {rid}: empty question")
        if not r.get("metadata", {}).get("subject_original"):
            error(f"{name} {rid}: metadata.subject_original missing")
        if r.get("subject") not in known:
            error(f"{name} {rid}: unknown subject {r.get('subject')!r}")
    gold = Counter(r.get("answer") for r in rows)
    print(f"  {len(rows)} questions; gold position 1/2/3/4 = {gold[1]}/{gold[2]}/{gold[3]}/{gold[4]}")


def check_probe_a(full, subset, shared):
    full_ids = {r["id"] for r in full}
    if not {r["id"] for r in subset} <= full_ids:
        error("probe_a_subset has questions that are not in probe_a_full")
    per = Counter(r["subject"] for r in subset)
    print(f"  subset per subject: min {min(per.values())}, max {max(per.values())} ({len(per)} subjects)")
    if len(per) != 14 or max(per.values()) - min(per.values()) > 1:
        error(f"probe_a_subset is not balanced over the 14 subjects: {dict(per)}")
    dups = near_duplicates(full, shared)
    for i, j, ratio, n in dups:
        error(f"Probe A {full[i]['id']} is a near-duplicate of shared-task {shared[j]['subject']} q{shared[j]['q_no']}")
    print(f"  near-duplicates of the shared-task test left in Probe A: {len(dups)}")


def check_probe_b(rows, probe_a, shared):
    sections = {r["id"]: r for r in read_jsonl(TEXTBOOKS)}
    heldout_ch = set(read_json(MIX_SUMMARY)["val_chapters"])
    copies = Counter(r["id"].split("#")[0] for r in read_jsonl(MIX_TRAIN) if r["source"] == "textbook")
    train_texts = [norm(s["text"]) for s in sections.values() if s["parent_id"] not in heldout_ch]

    not_in_evidence = []
    for r in rows:
        rid, src = r["id"], r["metadata"].get("source_id")
        sec = sections.get(src)
        if sec is None:
            error(f"{rid}: source section {src!r} does not exist")
            continue
        split = "heldout" if sec["parent_id"] in heldout_ch else "train"
        if r["split"] != split:
            error(f"{rid}: split is {r['split']!r} but {src} is a {split} section")
        if split == "train" and copies[src] != 3:
            error(f"{rid}: {src} appears {copies[src]}x in mix/train.jsonl, expected 3")
        if split == "heldout" and copies[src]:
            error(f"{rid}: held-out section {src} is in mix/train.jsonl")
        ev = norm(r["metadata"].get("evidence", ""))
        if len(ev) < 15:
            error(f"{rid}: evidence quote missing or too short")
        elif ev not in norm(sec["text"]):
            error(f"{rid}: evidence is not word for word in {src}: {r['metadata']['evidence'][:80]!r}")
        elif split == "heldout" and any(ev in t for t in train_texts):
            error(f"{rid}: held-out evidence also appears in a TRAINING section (the control is leaked)")
        if norm(r["choices"][r["answer"] - 1]) not in ev:
            not_in_evidence.append(rid)

    split_n = Counter(r["split"] for r in rows)
    for split in ("train", "heldout"):
        part = [r for r in rows if r["split"] == split]
        gold = Counter(r["answer"] for r in part)
        print(f"  {split:8s}: {len(part)} questions, {len({r['metadata']['source_id'] for r in part})} sections, "
              f"gold 1/2/3/4 = {gold[1]}/{gold[2]}/{gold[3]}/{gold[4]}")
        if part and max(gold.values()) - min(gold.get(k, 0) for k in range(1, 5)) > 1:
            error(f"{split}: gold positions are not balanced")
        per = Counter(r["subject"] for r in part)
        print("            " + ", ".join(f"{s} {n}" for s, n in sorted(per.items())))
    if split_n["train"] < 250 or split_n["heldout"] < 80:
        warn(f"fewer questions than planned (~300 train, ~100 heldout): {dict(split_n)}")

    longest = sum(len(r["choices"][r["answer"] - 1]) == max(len(c) for c in r["choices"]) for r in rows)
    print(f"  gold is the longest option in {longest}/{len(rows)} = {longest / len(rows):.0%} (chance 25%)")
    if longest / len(rows) > 0.40:
        warn("the gold option is often the longest: a model can score by picking the longest option")
    print(f"  gold option text found word for word in the evidence: {len(rows) - len(not_in_evidence)}/{len(rows)} "
          f"(the others paraphrase it: read them in the review)")

    qn = Counter(norm(r["question"]) + "|" + "|".join(sorted(norm(c) for c in r["choices"])) for r in rows)
    for k, n in qn.items():
        if n > 1:
            error(f"duplicate Probe B question: {k[:60]!r}")
    for name, others in (("Probe A", probe_a), ("the shared-task test", shared)):
        for i, j, ratio, n in near_duplicates(rows, others):
            warn(f"{rows[i]['id']} is close to a {name} question ({others[j]['subject']} q{others[j]['q_no']})")


def check_tokenizer(probes):
    """The plain-text prompt must end so that the next token is the bare answer digit."""
    sys.path.insert(0, str(PROBE_DIR.parents[2] / "scripts" / "cpt"))
    from common import tokenizer
    tok = tokenizer()
    digits = [tok.encode(str(d), add_special_tokens=False).ids for d in range(1, 5)]
    if any(len(d) != 1 for d in digits):
        error(f"an option digit is not a single token: {digits}")
    digits = [d[0] for d in digits]
    for name, rows in probes.items():
        bad = 0
        for r in rows[:200]:
            p = build_prompt(r)
            a = tok.encode(p, add_special_tokens=False).ids
            b = tok.encode(p + str(r["answer"]), add_special_tokens=False).ids
            bad += b != a + [digits[r["answer"] - 1]]
        if bad:
            error(f"{name}: in {bad} prompts the answer digit does not tokenize as one separate token after the prompt")
        print(f"  {name}: prompt + digit = prompt tokens + one digit token ({min(200, len(rows)) - bad}/{min(200, len(rows))})")
    print(f"  digit token ids 1-4: {digits}")
    print("\n  example prompt (Probe B):\n" + "\n".join("    | " + line for line in build_prompt(probes["probe_b"][0]).split("\n")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", action="store_true", help="also check the prompt / answer-token alignment")
    args = ap.parse_args()
    stdout_utf8()

    probes = {}
    for name in ("probe_a_full", "probe_a_subset", "probe_b"):
        path = PROBE_DIR / f"{name}.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} is missing: run scripts/probes/build_probes.py")
        probes[name] = list(read_jsonl(path))
        print(f"\n[{name}] format")
        check_format(name, probes[name])

    shared = load_json_items(SHARED_TEST_GLOB)
    print("\n[Probe A]")
    check_probe_a(probes["probe_a_full"], probes["probe_a_subset"], shared)
    print(f"  source questions: {len(load_json_items(PROBE_A_GLOB))}, excluded: "
          f"{sum(1 for _ in read_jsonl(PROBE_DIR / 'probe_a_excluded.jsonl'))}")
    print("\n[Probe B]")
    check_probe_b(probes["probe_b"], probes["probe_a_full"], shared)
    if args.tokenizer:
        print("\n[tokenizer]")
        check_tokenizer(probes)

    print(f"\n{'FAILED: ' + str(len(ERRORS)) + ' error(s)' if ERRORS else 'OK: no errors'}")
    sys.exit(1 if ERRORS else 0)


if __name__ == "__main__":
    main()
