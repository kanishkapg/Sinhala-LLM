"""Step 1 + 2 of docs/prepare_sft_dataset.md: freeze the inputs, then plan every question slot.

    python scripts/sft/s01_plan.py --level easy            # writes 00_plan/{sections,jobs}.jsonl, quotas.json
    python scripts/sft/s01_plan.py --level easy --topup    # after s06: top-up jobs for cells below 90% of quota

Deterministic (seed 42): the same inputs always give the same plan. Re-running overwrites 00_plan/ only when
nothing has been generated yet (or with --force), because generated jobs refer to these job ids.
"""
import argparse
import math
from collections import Counter, defaultdict

from sft_common import (QTYPES, COGNITIVE, SEED, Paths, ensure_dirs, file_sha1, level_arg, load_config, norm,
                        read_json, read_jsonl, read_jsonl_tolerant, seeded_rng, stable_int, stdout_utf8,
                        update_manifest, write_json, write_jsonl, CONFIG_DIR)


# ---------------------------------------------------------------- step 1: sections

def load_sections(paths: Paths, cfg: dict, exemplar_ids: set[str]) -> tuple[list[dict], dict]:
    man = read_json(paths.cpt_manifest)
    val_chapters = set(man["mix_summary"]["val_chapters"])
    pb_sources = {r["source_id"] for r in read_jsonl(paths.probes / "probe_b_sources.jsonl")
                  if r.get("role") == "primary" and r.get("split") == "train"}
    rows = list(read_jsonl(paths.textbook))
    stats = Counter()
    out = []
    for r in rows:
        stats["all"] += 1
        if r["parent_id"] in val_chapters:
            stats["excluded_heldout_chapter"] += 1
            continue
        if r["n_tokens"] < cfg["plan"]["min_section_tokens"]:
            stats["excluded_short"] += 1
            continue
        if r["id"] in exemplar_ids:
            stats["excluded_exemplar_source"] += 1
            continue
        stats["eligible"] += 1
        pb = r["id"] in pb_sources
        stats["probe_b_source"] += pb
        out.append({"id": r["id"], "parent_id": r["parent_id"], "book_id": r["book_id"], "subject": r["subject"],
                    "grade": r["grade"], "chapter_no": r["chapter_no"], "chapter": r["chapter"],
                    "section": r["section"], "n_tokens": r["n_tokens"], "probe_b_source": pb, "text": r["text"]})
    stats["eligible_tokens"] = sum(s["n_tokens"] for s in out)
    return out, dict(stats)


def check_exemplars(paths: Paths, level: str) -> tuple[list[dict], set[str]]:
    ex_path = CONFIG_DIR / f"exemplars_{level}.json"
    ex = read_json(ex_path)
    by_id = {r["id"]: r for r in read_jsonl(paths.textbook)}
    errors = []
    for e in ex["exemplars"]:
        src = by_id.get(e["source_id"])
        if not src:
            errors.append(f"{e['source_id']}: not in textbook.jsonl")
            continue
        quotes = [e["evidence"]] if e["evidence"] else []
        quotes += [s["evidence"] for s in e.get("statements", [])]
        for q in quotes:
            if norm(q) not in norm(src["text"]):
                errors.append(f"{e['source_id']} ({e['qtype']}): evidence not found in the section: {q[:60]!r}")
    for msg in errors:
        print("EXEMPLAR ERROR:", msg)
    if errors:
        raise SystemExit("fix config/exemplars_*.json first")
    if not ex.get("approved"):
        print("NOTE: exemplars are not approved yet (config/exemplars_%s.json 'approved': false). "
              "Have the Sinhala reader check them before wave 0." % level)
    return ex["exemplars"], {e["source_id"] for e in ex["exemplars"]}


# ---------------------------------------------------------------- step 2: quotas and jobs

def subject_quotas(sections: list[dict], cfg: dict) -> dict:
    p = cfg["plan"]
    tok = Counter()
    for s in sections:
        tok[s["subject"]] += s["n_tokens"]
    total_tok = sum(tok.values())
    n = len(tok)
    w = {s: (0.5 / n + 0.5 * tok[s] / total_tok) * p["subject_weight_factor"].get(s, 1.0) for s in tok}
    z = sum(w.values())
    out = {}
    for s in sorted(tok):
        target = round(p["total_questions"] * w[s] / z)
        cap = tok[s] // p["tokens_per_question_cap"]
        out[s] = {"tokens": tok[s], "target": target, "cap": cap, "quota": min(target, cap)}
    return out


def build_jobs(sections: list[dict], cfg: dict, level: str) -> list[dict]:
    """One job per section; consecutive short sections of the same chapter are merged (doc 2.2)."""
    p = cfg["plan"]
    jobs, cur = [], None

    def flush():
        nonlocal cur
        if cur:
            jobs.append(cur)
        cur = None

    for s in sections:                      # file order = book / chapter / section order
        short = s["n_tokens"] < p["merge_below_tokens"]
        if (cur and short and cur["chapter_id"] == s["parent_id"] and cur["mergeable"]
                and cur["n_tokens"] + s["n_tokens"] <= p["merge_max_tokens"]):
            cur["section_ids"].append(s["id"])
            cur["section_titles"].append(s["section"])
            cur["n_tokens"] += s["n_tokens"]
            cur["probe_b_source"] = cur["probe_b_source"] or s["probe_b_source"]
            continue
        flush()
        cur = {"job_id": f"{level}-" + s["id"].replace("/", "-"), "subject": s["subject"], "grade": s["grade"],
               "book_id": s["book_id"], "chapter_id": s["parent_id"], "chapter": s["chapter"],
               "section_ids": [s["id"]], "section_titles": [s["section"]], "n_tokens": s["n_tokens"],
               "probe_b_source": s["probe_b_source"], "mergeable": short}
    flush()
    for j in jobs:
        j["merged"] = len(j["section_ids"]) > 1
        j["cap"] = min(p["max_slots_per_merged_job"] if j["merged"] else p["max_slots_per_section"],
                       max(1, math.ceil(j["n_tokens"] / p["tokens_per_question_cap"])))
        del j["mergeable"]
    return jobs


def allocate(jobs: list[dict], quota: int, min_per_chapter: int) -> None:
    """Largest-remainder allocation of `quota` slots over a subject's jobs, proportional to tokens."""
    T = sum(j["n_tokens"] for j in jobs)
    for j in jobs:
        j["_f"] = quota * j["n_tokens"] / T
        j["n_slots"] = min(j["cap"], max(1, math.floor(j["_f"])))
    diff = quota - sum(j["n_slots"] for j in jobs)
    while diff > 0:
        cands = [j for j in jobs if j["n_slots"] < j["cap"]]
        if not cands:
            break
        cands.sort(key=lambda j: (j["_f"] - j["n_slots"], j["job_id"]), reverse=True)
        for j in cands[:diff]:
            j["n_slots"] += 1
        diff = quota - sum(j["n_slots"] for j in jobs)
    while diff < 0:
        cands = sorted([j for j in jobs if j["n_slots"] > 1], key=lambda j: (j["n_slots"] - j["_f"], j["job_id"]),
                       reverse=True)
        if not cands:
            break
        for j in cands[:-diff]:
            j["n_slots"] -= 1
        diff = quota - sum(j["n_slots"] for j in jobs)
    # every chapter gets at least `min_per_chapter` slots, paid for by the most over-served job
    by_ch = defaultdict(list)
    for j in jobs:
        by_ch[j["chapter_id"]].append(j)
    for ch, js in sorted(by_ch.items()):
        while sum(j["n_slots"] for j in js) < min_per_chapter:
            room = [j for j in js if j["n_slots"] < j["cap"]]
            if not room:
                break
            max(room, key=lambda j: j["n_tokens"])["n_slots"] += 1
            donors = sorted([j for j in jobs if j["chapter_id"] != ch and j["n_slots"] > 1 and
                             sum(x["n_slots"] for x in by_ch[j["chapter_id"]]) > min_per_chapter],
                            key=lambda j: j["n_slots"] - j["_f"], reverse=True)
            if donors:
                donors[0]["n_slots"] -= 1
    for j in jobs:
        del j["_f"]


def exact_counts(n: int, quota: dict) -> dict:
    """Split n into the quota shares exactly (largest remainder)."""
    raw = {k: n * v for k, v in quota.items()}
    out = {k: math.floor(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    return out


def assign_types(jobs: list[dict], cfg: dict, subject: str) -> dict:
    n = sum(j["n_slots"] for j in jobs)
    tcount = exact_counts(n, cfg["plan"]["qtype_quota"])
    ccount = exact_counts(n, cfg["plan"]["cognitive_quota"])
    types = [t for t in QTYPES for _ in range(tcount[t])]              # grouped by type
    cogs = [c for c in COGNITIVE for _ in range(ccount[c])]
    rng = seeded_rng(SEED, "types", subject)
    rng.shuffle(cogs)
    order = sorted(jobs, key=lambda j: stable_int(SEED, "deal", j["job_id"]))
    for j in order:
        j["slots"] = []
    # deal the grouped type list round-robin over the jobs: each job gets types far apart in the list = different
    i = 0
    for t in types:
        for _ in range(len(order)):
            j = order[i % len(order)]
            i += 1
            if len(j["slots"]) < j["n_slots"]:
                j["slots"].append({"slot": len(j["slots"]) + 1, "qtype": t, "cognitive": None})
                break
    k = 0
    for j in order:
        for s in j["slots"]:
            s["cognitive"] = cogs[k]
            k += 1
    return {"n": n, "qtype": tcount, "cognitive": ccount}


def assign_waves(jobs: list[dict], cfg: dict) -> None:
    """Alternate jobs inside every chapter between wave 1 and wave 2; then take wave 0 (calibration) out of
    wave 2: `wave0_jobs_per_subject` jobs per subject, covering every question type."""
    by_ch = defaultdict(list)
    for j in jobs:
        by_ch[j["chapter_id"]].append(j)
    for ch, js in by_ch.items():
        start = stable_int(SEED, "wave", ch) % 2
        for i, j in enumerate(js):
            j["wave"] = 1 + (i + start) % 2
    by_subj = defaultdict(list)
    for j in jobs:
        if j["wave"] == 2 and len(j["slots"]) >= 3:
            by_subj[j["subject"]].append(j)
    chosen = []
    for s, js in sorted(by_subj.items()):
        js = sorted(js, key=lambda j: stable_int(SEED, "wave0", j["job_id"]))
        chosen += js[: cfg["plan"]["wave0_jobs_per_subject"]]
    covered = {sl["qtype"] for j in chosen for sl in j["slots"]}
    for t in QTYPES:                          # make sure every type appears in calibration
        if t not in covered:
            extra = sorted([j for js in by_subj.values() for j in js if j not in chosen
                            and any(sl["qtype"] == t for sl in j["slots"])],
                           key=lambda j: stable_int(SEED, "wave0x", j["job_id"]))
            if extra:
                chosen.append(extra[0])
                covered |= {sl["qtype"] for sl in extra[0]["slots"]}
    for j in chosen:
        j["wave"] = 0


# ---------------------------------------------------------------- top-up

def topup(paths: Paths, cfg: dict, level: str) -> None:
    jobs = list(read_jsonl(paths.plan / "jobs.jsonl"))
    quotas = read_json(paths.plan / "quotas.json")
    items_path = paths.final / "items.jsonl"
    if not items_path.exists():
        raise SystemExit("run s06_dedup.py first (top-up compares accepted items with the quotas)")
    items = list(read_jsonl(items_path))
    gen_items = list(read_jsonl(paths.items / "items_raw.jsonl"))
    acc = Counter((i["subject"], i["metadata"]["qtype"]) for i in items)
    gen = Counter(i["subject"] for i in gen_items)
    acc_subj = Counter(i["subject"] for i in items)
    acc_job = Counter(i["metadata"]["job_id"] for i in items)
    acc_ch = Counter(i["metadata"]["chapter_id"] for i in items)
    stems_by_job = defaultdict(list)
    for i in items:
        stems_by_job[i["metadata"]["job_id"]].append(i["question"])
    used = Counter()
    for j in jobs:
        used[j["job_id"]] += len(j["slots"])
    new_jobs = {}

    def add_slot(base: dict, qtype: str, cognitive: str):
        jid = base["job_id"] + "-t"
        nj = new_jobs.setdefault(jid, {**{k: base[k] for k in base if k not in ("slots", "wave", "job_id")},
                                        "job_id": jid, "wave": 3, "slots": [], "parent_job": base["job_id"],
                                        "avoid_stems": stems_by_job.get(base["job_id"], [])})
        nj["slots"].append({"slot": len(nj["slots"]) + 1, "qtype": qtype, "cognitive": cognitive})

    for subj, q in quotas["subjects"].items():
        rate = max(0.5, acc_subj[subj] / gen[subj]) if gen[subj] else 0.8
        sj = sorted([j for j in jobs if j["subject"] == subj],
                    key=lambda j: (acc_job[j["job_id"]], stable_int(SEED, "topup", j["job_id"])))
        for t, planned in q["qtype"].items():
            have = acc[(subj, t)]
            if have >= 0.9 * planned:
                continue
            need = math.ceil((planned - have) / rate)
            for j in sj:
                if need <= 0:
                    break
                limit = cfg["plan"]["max_slots_per_merged_job"] if j.get("merged") else cfg["plan"]["max_slots_per_section"]
                nj = new_jobs.get(j["job_id"] + "-t")
                if used[j["job_id"]] + (len(nj["slots"]) if nj else 0) >= limit:
                    continue
                add_slot(j, t, "recall" if t in ("completion", "direct", "blank") else "understanding")
                need -= 1
    for ch in sorted({j["chapter_id"] for j in jobs}):
        if acc_ch[ch] == 0:
            j = max([j for j in jobs if j["chapter_id"] == ch], key=lambda j: j["n_tokens"])
            add_slot(j, "completion", "recall")
            add_slot(j, "direct", "recall")
    out = sorted(new_jobs.values(), key=lambda j: j["job_id"])
    write_jsonl(paths.plan / "jobs_topup.jsonl", out)
    print(f"top-up: {len(out)} jobs, {sum(len(j['slots']) for j in out)} slots -> 00_plan/jobs_topup.jsonl "
          "(run: s02_generate.py --wave 3)")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    level_arg(ap)
    ap.add_argument("--topup", action="store_true", help="plan top-up jobs (wave 3) after s06")
    ap.add_argument("--force", action="store_true", help="re-plan even though generated jobs exist")
    args = ap.parse_args()
    stdout_utf8()
    paths, cfg = Paths(args.level), load_config(args.level)
    ensure_dirs(paths)
    if args.topup:
        return topup(paths, cfg, args.level)
    if any(read_jsonl_tolerant(f) for f in paths.all_wave_files()) and not args.force:
        raise SystemExit("01_generated/ already has jobs: re-planning would change job ids. Use --force only if "
                         "you mean to start over (and move 01_generated/ away).")

    exemplars, ex_ids = check_exemplars(paths, args.level)
    sections, stats = load_sections(paths, cfg, ex_ids)
    quotas = subject_quotas(sections, cfg)
    jobs = build_jobs(sections, cfg, args.level)
    type_counts = {}
    for subj in sorted(quotas):
        sj = [j for j in jobs if j["subject"] == subj]
        allocate(sj, quotas[subj]["quota"], cfg["plan"]["min_slots_per_chapter"])
        type_counts[subj] = assign_types(sj, cfg, subj)
    jobs = [j for j in jobs if j["slots"]]
    assign_waves(jobs, cfg)
    for j in jobs:
        j.pop("n_slots", None)
        j.pop("cap", None)

    write_jsonl(paths.plan / "sections.jsonl", sections)
    write_jsonl(paths.plan / "jobs.jsonl", jobs)
    wave_slots = Counter()
    for j in jobs:
        wave_slots[j["wave"]] += len(j["slots"])
    q_out = {"subjects": {s: {**quotas[s], **type_counts[s]} for s in quotas},
             "total_slots": sum(wave_slots.values()), "slots_per_wave": dict(sorted(wave_slots.items())),
             "jobs_per_wave": dict(sorted(Counter(j["wave"] for j in jobs).items())),
             "chapters": len({j["chapter_id"] for j in jobs})}
    write_json(paths.plan / "quotas.json", q_out)
    update_manifest(paths, "plan", {"textbook_sha1": file_sha1(paths.textbook),
                                    "cpt_manifest_sha1": file_sha1(paths.cpt_manifest),
                                    "exemplars_sha1": file_sha1(CONFIG_DIR / f"exemplars_{args.level}.json"),
                                    "config_sha1": file_sha1(CONFIG_DIR / f"sft_{args.level}.json"),
                                    "sections": stats, **q_out})

    # ---- report + checks (doc Step 1 / 2.5)
    print(f"sections: {stats}")
    print(f"\n{'subject':<20}{'tokens':>9}{'target':>8}{'cap':>6}{'quota':>7}{'slots':>7}{'jobs':>6}{'chapters':>9}")
    for s, q in quotas.items():
        sj = [j for j in jobs if j["subject"] == s]
        print(f"{s:<20}{q['tokens']:>9}{q['target']:>8}{q['cap']:>6}{q['quota']:>7}"
              f"{sum(len(j['slots']) for j in sj):>7}{len(sj):>6}{len({j['chapter_id'] for j in sj}):>9}")
    print(f"\ntotal slots {q_out['total_slots']}; per wave {q_out['slots_per_wave']}; jobs per wave {q_out['jobs_per_wave']}")
    tq = Counter(sl["qtype"] for j in jobs for sl in j["slots"])
    print("types:", ", ".join(f"{t} {tq[t]} ({100 * tq[t] / q_out['total_slots']:.1f}%)" for t in QTYPES))
    problems = []
    w1_ch = {j["chapter_id"] for j in jobs if j["wave"] == 1}
    missing = {j["chapter_id"] for j in jobs} - w1_ch
    if missing:
        print(f"note: {len(missing)} chapters have no wave-1 job (single-job chapters that went to wave 0/2)")
    for j in jobs:
        lim = cfg["plan"]["max_slots_per_merged_job"] if j["merged"] else cfg["plan"]["max_slots_per_section"]
        if len(j["slots"]) > lim:
            problems.append(f"{j['job_id']}: {len(j['slots'])} slots > {lim}")
    for p in problems[:20]:
        print("PROBLEM:", p)
    w0 = [j for j in jobs if j["wave"] == 0]
    print(f"wave 0 (calibration): {len(w0)} jobs, {sum(len(j['slots']) for j in w0)} slots, types "
          f"{sorted({sl['qtype'] for j in w0 for sl in j['slots']})}")
    print(f"\n-> {paths.plan}")


if __name__ == "__main__":
    main()
