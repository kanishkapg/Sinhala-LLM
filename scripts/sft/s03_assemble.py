"""Step 5 of docs/prepare_sft_dataset.md: turn the generator output into SinhalaMMLU-style items, with the key
placed by code from an exactly balanced position schedule.

    python scripts/sft/s03_assemble.py --level easy

Rewrites 02_items/ completely from 01_generated/ every time (deterministic). Positions are scheduled per
(wave, subject, qtype) cell, so adding a later wave never moves the options of earlier items.
"""
import argparse
from collections import Counter, defaultdict
from itertools import combinations

from sft_common import (SEED, Paths, combination_letters, ensure_dirs, level_arg, load_config, normalize,
                        read_jsonl, read_jsonl_tolerant, seeded_rng, share_table, stable_int, stdout_utf8,
                        subject_info, write_jsonl)


def combo_text(S: frozenset, n: int) -> str:
    L = sorted(S)
    if len(L) == 1:
        return f"{L[0]} පමණි"
    joined = ", ".join(L[:-1]) + " හා " + L[-1]
    return joined + (" සියල්ලම" if len(L) == n else " පමණි")


def combo_options(n: int, T: frozenset, target: int, key: str) -> tuple[list[frozenset], int]:
    """Choose 3 distractor subsets so that, in canonical order (by size, then letters), the true set T lands on
    `target` when possible. Prefers distractors of T's size, the full set, and neighbouring sizes."""
    letters = combination_letters(n)
    full = frozenset(letters)
    cands = [frozenset(c) for r in range(1, n + 1) for c in combinations(letters, r) if frozenset(c) != T]

    def order(S):
        return (len(S), sorted(S))

    def pref(S):
        s = 2 if len(S) == len(T) else 0
        s += 1 if S == full else 0
        s += 1 if abs(len(S) - len(T)) == 1 else 0
        s -= 2 if len(S) == 1 and len(T) > 1 else 0
        return s

    best, best_score = None, None
    for D in combinations(cands, 3):
        opts = sorted([T, *D], key=order)
        pos = opts.index(T) + 1
        score = (10 if pos == target else 0) + sum(pref(S) for S in D)
        tie = stable_int(key, *["".join(sorted(S)) for S in D]) % 1000 / 1000
        if best_score is None or score + tie > best_score:
            best, best_score = (opts, pos), score + tie
    return best


def main():
    ap = argparse.ArgumentParser(description="Assemble generated questions into items")
    level_arg(ap)
    args = ap.parse_args()
    stdout_utf8()
    paths, cfg = Paths(args.level), load_config(args.level)
    ensure_dirs(paths)

    jobs = {j["job_id"]: j for j in read_jsonl(paths.plan / "jobs.jsonl")}
    if (paths.plan / "jobs_topup.jsonl").exists():
        jobs |= {j["job_id"]: j for j in read_jsonl(paths.plan / "jobs_topup.jsonl")}
    records = {}
    for f in paths.all_wave_files():
        for r in read_jsonl_tolerant(f):
            if r.get("status") == "ok":
                records[r["job_id"]] = r                     # the latest ok record of a job wins

    raw, skipped, bad = [], [], []
    for jid, rec in sorted(records.items()):
        job = jobs.get(jid)
        if not job:
            print(f"WARNING: {jid} is not in the plan, ignored")
            continue
        planned = {s["slot"]: s for s in job["slots"]}
        seen = Counter()
        for q in rec["questions"]:
            slot = q.get("slot")
            seen[slot] += 1
            iid = f"{jid}-q{slot}" + (f"-{seen[slot]}" if seen[slot] > 1 else "")
            base = {"item_id": iid, "job_id": jid, "wave": job["wave"], "subject": job["subject"], "slot": slot,
                    "qtype": q.get("qtype"), "cognitive": q.get("cognitive")}
            if q.get("skip_reason"):
                skipped.append({**base, "skip_reason": q["skip_reason"]})
                continue
            if slot not in planned:
                bad.append({**base, "reasons": ["slot_not_planned"]})
                continue
            if q.get("qtype") != planned[slot]["qtype"]:
                bad.append({**base, "reasons": [f"qtype_changed:{planned[slot]['qtype']}->{q.get('qtype')}"]})
                continue
            raw.append((job, rec, q, base))

    # position schedule per (wave, subject, qtype) cell
    cells = defaultdict(list)
    for job, rec, q, base in raw:
        cells[(job["wave"], job["subject"], base["qtype"])].append(base["item_id"])
    target = {}
    for cell, ids in cells.items():
        ids.sort(key=lambda i: stable_int(SEED, "pos", i))
        off = stable_int(SEED, "posoff", *cell) % 4
        for k, iid in enumerate(ids):
            target[iid] = (off + k) % 4 + 1

    items = []
    for job, rec, q, base in raw:
        iid = base["item_id"]
        mmlu_subject, category, so = subject_info(job["subject"])
        stem = normalize(q.get("question", ""))
        meta = {"subject_original": so, "mmlu_subject": mmlu_subject, "difficulty": cfg["level"],
                "grade": job["grade"], "source_ids": job["section_ids"], "source_id": job["section_ids"][0],
                "chapter_id": job["chapter_id"], "job_id": job["job_id"], "wave": job["wave"], "slot": base["slot"],
                "qtype": base["qtype"], "cognitive": base["cognitive"], "evidence": q.get("evidence", ""),
                "explanation_en": q.get("explanation_en", ""), "probe_b_source": job["probe_b_source"],
                "generator": rec.get("model"), "instructions_sha1": rec.get("instructions_sha1"),
                "target_position": target[iid]}
        if base["qtype"] == "combination":
            sts = q.get("statements") or []
            if not 3 <= len(sts) <= 4:
                bad.append({**base, "reasons": ["combination_statement_count"]})
                continue
            letters = combination_letters(len(sts))
            T = frozenset(letters[i] for i, s in enumerate(sts) if s.get("true"))
            if not T:
                bad.append({**base, "reasons": ["combination_no_true_statement"]})
                continue
            closing = normalize(q.get("closing", "")) or "ඉහත ප්‍රකාශ අතුරින් නිවැරදි වන්නේ,"
            lines = [f"{letters[i]} - {normalize(s.get('text', ''))}" for i, s in enumerate(sts)]
            stem = stem + "\n" + "\n".join(lines) + "\n" + closing
            opts, pos = combo_options(len(sts), T, target[iid], iid)
            choices = [combo_text(S, len(sts)) for S in opts]
            meta["statements"] = [{"label": letters[i], "text": normalize(s.get("text", "")), "true": bool(s.get("true")),
                                   "evidence": s.get("evidence", "")} for i, s in enumerate(sts)]
            meta["key_text"] = choices[pos - 1]
            meta["closing"] = closing
        else:
            key = normalize(q.get("correct", ""))
            ds = [normalize(d) for d in (q.get("distractors") or [])]
            if not key or len(ds) != 3 or not all(ds):
                bad.append({**base, "reasons": ["schema_options"]})
                continue
            rng = seeded_rng(SEED, "distractors", iid)
            rng.shuffle(ds)
            pos = target[iid]
            choices = ds[: pos - 1] + [key] + ds[pos - 1:]
            meta["key_text"] = key
        meta["position_on_schedule"] = pos == target[iid]
        items.append({"id": f"sft-{iid}", "subject": job["subject"], "category": category, "question": stem,
                      "choices": choices, "answer": pos, "metadata": meta})

    items.sort(key=lambda x: x["id"])
    write_jsonl(paths.items / "items_raw.jsonl", items)
    write_jsonl(paths.items / "skipped.jsonl", skipped)
    write_jsonl(paths.items / "assemble_rejected.jsonl", bad)

    print(f"jobs with output: {len(records)}; items {len(items)}; skipped slots {len(skipped)}; "
          f"assembly rejects {len(bad)}")
    print("items per wave:", dict(sorted(Counter(i["metadata"]["wave"] for i in items).items())))
    print("key positions:", share_table(Counter(i["answer"] for i in items)))
    off = sum(not i["metadata"]["position_on_schedule"] for i in items)
    if off:
        print(f"note: {off} combination items could not take their scheduled position (natural option order)")
    if skipped:
        print("top skip reasons:")
        for r, n in Counter(s["skip_reason"][:70] for s in skipped).most_common(8):
            print(f"  {n:>4}  {r}")
    if bad:
        print("assembly rejects:", dict(Counter(r for b in bad for r in b["reasons"])))
    print(f"-> {paths.items}")


if __name__ == "__main__":
    main()
