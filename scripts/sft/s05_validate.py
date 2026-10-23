"""Step 7 of docs/prepare_sft_dataset.md: blind validation by a second model, then the accept rule.

    python scripts/sft/s05_validate.py --level easy --wave 0          # validate, then decide
    python scripts/sft/s05_validate.py --level easy --decide-only     # only re-apply the accept rule

The validator sees the section and each question with its options as they will be trained, but NOT the key, the
evidence, the explanation or the question type. It answers each question itself; an item is accepted only if its
answer equals the key and every quality flag passes. Same subscription guards as s02 (claude_runner.py).
Resumable: results are appended per job; a job whose items changed (re-assembled) is validated again.
"""
import argparse
import json
import threading
import time
from collections import Counter, defaultdict

from claude_runner import ClaudeRunner, GuardStop, run_pool, usage_summary
from sft_common import (PROMPT_DIR, Paths, append_jsonl, ensure_dirs, level_arg, load_config, read_jsonl,
                        read_jsonl_tolerant, sha1, stdout_utf8, update_manifest, write_jsonl)

SYSTEM = ("You are a careful Sri Lankan school examiner who checks Sinhala multiple-choice questions. Follow the "
          "user's instructions exactly and answer only through the required JSON schema.")
RESULT = {"type": "object", "additionalProperties": False,
          "required": ["item_id", "answer", "confidence", "single_best", "also_defensible", "supported_by_section",
                       "self_contained", "needs_figure", "language_ok", "grade_ok", "implausible_distractors",
                       "source_garbled", "issues"],
          "properties": {"item_id": {"type": "string"}, "answer": {"type": "integer"},
                         "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                         "single_best": {"type": "boolean"},
                         "also_defensible": {"type": "array", "items": {"type": "integer"}},
                         "supported_by_section": {"type": "boolean"}, "self_contained": {"type": "boolean"},
                         "needs_figure": {"type": "boolean"}, "language_ok": {"type": "boolean"},
                         "grade_ok": {"type": "boolean"}, "implausible_distractors": {"type": "integer"},
                         "source_garbled": {"type": "boolean"},
                         "issues": {"type": "array", "items": {"type": "string"}}}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"],
          "properties": {"items": {"type": "array", "items": RESULT}}}


def payload_sha(items: list[dict]) -> str:
    return sha1(json.dumps([[i["id"], i["question"], i["choices"]] for i in items], ensure_ascii=False))


def job_text(job_items: list[dict], sections: dict) -> str:
    m = job_items[0]["metadata"]
    src = "\n\n".join(sections[s] for s in m["source_ids"])
    qs = []
    for it in job_items:
        opts = "\n".join(f"{k}. {o}" for k, o in enumerate(it["choices"], 1))
        qs.append(f"[item_id: {it['id']}]\nප්‍රශ්නය: {it['question']}\n{opts}")
    return (f"=== SOURCE (subject: {job_items[0]['subject']}, grade {m['grade']}) ===\n{src}\n"
            f"=== QUESTIONS FROM THIS SOURCE ===\n" + "\n\n".join(qs) + "\n")


def decide(item: dict, v: dict, cfg: dict) -> list[str]:
    r = []
    if v.get("answer") != item["answer"]:
        r.append("validator_disagrees" if v.get("answer") else "validator_no_correct_option")
    if not v.get("single_best") or v.get("also_defensible"):
        r.append("not_single_best")
    for flag in ["supported_by_section", "self_contained", "language_ok", "grade_ok"]:
        if not v.get(flag):
            r.append(f"not_{flag}")
    if v.get("needs_figure"):
        r.append("needs_figure")
    if v.get("source_garbled"):
        r.append("source_garbled")
    if v.get("confidence") == "low":
        r.append("low_confidence")
    if (v.get("implausible_distractors") or 0) > 1:
        r.append("implausible_distractors")
    return r


def run_decisions(paths: Paths, cfg: dict) -> None:
    passed = list(read_jsonl(paths.checked / "passed.jsonl"))
    val = {}
    for r in read_jsonl_tolerant(paths.validated / "validation.jsonl"):
        if r.get("status") == "ok":
            val[(r["job_id"], r["payload_sha1"])] = {x["item_id"]: x for x in r["results"]}
    by_job = defaultdict(list)
    for it in passed:
        by_job[it["metadata"]["job_id"]].append(it)
    accepted, rejected, pending = [], [], 0
    for jid, its in by_job.items():
        res = val.get((jid, payload_sha(its)))
        for it in its:
            v = (res or {}).get(it["id"])
            if v is None:
                pending += 1
                continue
            reasons = decide(it, v, cfg)
            rec = {**it, "metadata": {**it["metadata"], "validation": v}}
            if reasons:
                rejected.append({**rec, "reject_reasons": reasons})
            else:
                accepted.append(rec)
    accepted.sort(key=lambda x: x["id"])
    rejected.sort(key=lambda x: x["id"])
    write_jsonl(paths.validated / "accepted.jsonl", accepted)
    write_jsonl(paths.validated / "rejected.jsonl", rejected)
    done = len(accepted) + len(rejected)
    rate = len(accepted) / done if done else float("nan")
    print(f"\ndecisions: {done} validated items -> accepted {len(accepted)} ({100 * rate:.1f}%), rejected "
          f"{len(rejected)}; {pending} passed items not validated yet")
    for r, k in Counter(x for it in rejected for x in it["reject_reasons"]).most_common():
        print(f"  {r:<32}{k:>6} ({100 * k / max(done, 1):.1f}%)")
    for key, name in [(lambda it: it["subject"], "subject"), (lambda it: it["metadata"]["qtype"], "type")]:
        tot, dis = Counter(), Counter()
        for it in accepted + rejected:
            tot[key(it)] += 1
        for it in rejected:
            if "validator_disagrees" in it["reject_reasons"] or "validator_no_correct_option" in it["reject_reasons"]:
                dis[key(it)] += 1
        warn = [f"{k} {100 * dis[k] / tot[k]:.0f}%" for k in sorted(tot) if tot[k] >= 10 and dis[k] / tot[k] > 0.15]
        print(f"key disagreement per {name}: " + ", ".join(f"{k} {100 * dis[k] / tot[k]:.0f}%" for k in sorted(tot)))
        if warn:
            print(f"  ABOVE 15% (read 20 of them before the next wave): {', '.join(warn)}")
    if done and rate < cfg["gates"]["validator_accept_min"]:
        print(f"GATE: acceptance {100 * rate:.1f}% < {100 * cfg['gates']['validator_accept_min']:.0f}%: "
              "fix the generator prompt before generating more")
    update_manifest(paths, "validation", {"validated": done, "accepted": len(accepted), "pending": pending,
                                          "acceptance": round(rate, 4) if done else None})


def main():
    ap = argparse.ArgumentParser(description="Blind validation of the checked items")
    level_arg(ap)
    ap.add_argument("--wave", type=int, default=None, help="only items of this wave (default: all)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--jobs-per-call", type=int, default=None)
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--decide-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-unverified-extra-usage", action="store_true")
    args = ap.parse_args()
    stdout_utf8()
    paths, cfg = Paths(args.level), load_config(args.level)
    ensure_dirs(paths)
    if args.decide_only:
        return run_decisions(paths, cfg)
    vcfg = cfg["validator"]
    model, effort = args.model or vcfg["model"], args.effort or vcfg["effort"]
    if model == cfg["generator"]["model"]:
        print("WARNING: validator model = generator model; a different model catches more errors")
    per_call = args.jobs_per_call or vcfg.get("jobs_per_call", 3)
    sections = {s["id"]: s["text"] for s in read_jsonl(paths.plan / "sections.jsonl")}
    by_job = defaultdict(list)
    for it in read_jsonl(paths.checked / "passed.jsonl"):
        if args.wave is None or it["metadata"]["wave"] == args.wave:
            by_job[it["metadata"]["job_id"]].append(it)
    out = paths.validated / "validation.jsonl"
    done = {(r["job_id"], r["payload_sha1"]) for r in read_jsonl_tolerant(out) if r.get("status") == "ok"}
    todo = [(jid, its) for jid, its in sorted(by_job.items()) if (jid, payload_sha(its)) not in done]
    groups = [todo[i:i + per_call] for i in range(0, len(todo), per_call)]
    if args.max_calls:
        groups = groups[: args.max_calls]
    instr = (PROMPT_DIR / "validator.md").read_text(encoding="utf-8")
    print(f"validation: {len(by_job)} jobs with checked items, {len(todo)} to validate -> {len(groups)} calls, "
          f"model {model}, effort {effort}")
    if args.dry_run:
        if groups:
            print("\n" + instr + "\n\n" + "\n".join(job_text(its, sections) for _, its in groups[0]))
        return
    if groups:
        runner = ClaudeRunner(cfg["guard"], paths.cli_cwd, paths.gen / "usage_log.jsonl",
                              args.allow_unverified_extra_usage)
        runner.preflight()
        update_manifest(paths, "validator", {"model": model, "effort": effort, "instructions_sha1": sha1(instr)})
        lock = threading.Lock()
        stats = Counter()

        def run(group):
            prompt = instr + "\n\n" + "\n".join(job_text(its, sections) for _, its in group)
            try:
                res = runner.call(prompt, model=model, effort=effort, system=SYSTEM, schema=SCHEMA,
                                  timeout=vcfg["timeout_s"], tag=f"val:{group[0][0]}")
            except GuardStop:
                raise
            except Exception as e:
                with lock:
                    for jid, its in group:
                        append_jsonl(out, {"job_id": jid, "payload_sha1": payload_sha(its), "status": "error",
                                           "error": str(e)[:500], "ts": time.time()})
                        stats["error"] += 1
                return
            got = {x.get("item_id"): x for x in res["data"].get("items", [])}
            with lock:
                for jid, its in group:
                    results = [got[i["id"]] for i in its if i["id"] in got]
                    status = "ok" if len(results) == len(its) else "error"
                    append_jsonl(out, {"job_id": jid, "payload_sha1": payload_sha(its), "status": status,
                                       "model": model, "effort": effort, "ts": time.time(), "results": results,
                                       **({"error": "items missing from the validator output"} if status == "error" else {})})
                    stats[status] += 1
                n = stats["ok"] + stats["error"]
                if n % 15 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)} jobs validated ({stats['error']} errors)", flush=True)

        run_pool(runner, run, groups, args.workers)
        print(usage_summary(paths.gen / "usage_log.jsonl", cfg["guard"]["window_hours"]))
    run_decisions(paths, cfg)
    print(f"-> {paths.validated}")


if __name__ == "__main__":
    main()
