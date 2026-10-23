"""Step 4 of docs/prepare_sft_dataset.md: generate the questions with Claude, through the Claude Code CLI on your
claude.ai subscription (see claude_runner.py for the guards that keep it from ever billing extra).

    python scripts/sft/s02_generate.py --level easy --wave 0 --dry-run     # print one prompt, call nothing
    python scripts/sft/s02_generate.py --level easy --wave 0               # calibration (26 jobs)
    python scripts/sft/s02_generate.py --level easy --wave 1 --workers 2   # pilot half
    python scripts/sft/s02_generate.py --level easy --wave 2
    python scripts/sft/s02_generate.py --level easy --wave 3               # top-up (after s01 --topup)

Resumable: every finished job is appended to 01_generated/wave<N>.jsonl at once; a re-run skips jobs already
"ok" and retries the failed ones. Ctrl+C at any time is safe. When the Pro usage limit is reached the script
waits for the reset and continues by itself.
"""
import argparse
import json
import threading
import time

from claude_runner import ClaudeRunner, GuardStop, run_pool, usage_summary
from sft_common import (PROMPT_DIR, CONFIG_DIR, Paths, append_jsonl, ensure_dirs, level_arg, load_config, read_json,
                        read_jsonl, read_jsonl_tolerant, sha1, stdout_utf8, subject_info, update_manifest)

SYSTEM = ("You are an expert Sri Lankan school examiner who writes Sinhala multiple-choice questions. Follow the "
          "user's instructions exactly and answer only through the required JSON schema.")

STATEMENT = {"type": "object", "additionalProperties": False,
             "required": ["label", "text", "true", "evidence"],
             "properties": {"label": {"type": "string", "enum": ["A", "B", "C", "D"]}, "text": {"type": "string"},
                            "true": {"type": "boolean"}, "evidence": {"type": "string"}}}
QUESTION = {"type": "object", "additionalProperties": False,
            "required": ["slot", "qtype", "cognitive", "question", "closing", "correct", "distractors",
                         "statements", "evidence", "explanation_en", "skip_reason"],
            "properties": {"slot": {"type": "integer"},
                           "qtype": {"type": "string", "enum": ["completion", "identification", "direct", "negation",
                                                                "blank", "statement", "combination"]},
                           "cognitive": {"type": "string", "enum": ["recall", "understanding", "application"]},
                           "question": {"type": "string"}, "closing": {"type": "string"},
                           "correct": {"type": "string"},
                           "distractors": {"type": "array", "items": {"type": "string"}},
                           "statements": {"type": "array", "items": STATEMENT},
                           "evidence": {"type": "string"}, "explanation_en": {"type": "string"},
                           "skip_reason": {"type": "string"}}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["jobs"],
          "properties": {"jobs": {"type": "array", "items": {
              "type": "object", "additionalProperties": False, "required": ["job_id", "questions"],
              "properties": {"job_id": {"type": "string"}, "questions": {"type": "array", "items": QUESTION}}}}}}

EXEMPLAR_FIELDS = ["slot", "qtype", "cognitive", "question", "closing", "correct", "distractors", "statements",
                   "evidence", "explanation_en", "skip_reason"]


def instructions(level: str) -> str:
    ex = read_json(CONFIG_DIR / f"exemplars_{level}.json")["exemplars"]
    blocks = []
    for e in ex:
        blocks.append(f"Example — {e['subject']}, grade {e['grade']}, qtype={e['qtype']}, cognitive={e['cognitive']}:\n"
                      + json.dumps({k: e[k] for k in EXEMPLAR_FIELDS}, ensure_ascii=False, indent=1))
    text = (PROMPT_DIR / "generator.md").read_text(encoding="utf-8")
    return text.replace("{EXEMPLARS}", "\n\n".join(blocks))


def job_block(job: dict, sections: dict) -> str:
    so = subject_info(job["subject"])[2]
    slots = "\n".join(f"  {s['slot']}. qtype={s['qtype']}, cognitive={s['cognitive']}" for s in job["slots"])
    text = "\n\n".join(sections[sid]["text"] for sid in job["section_ids"])
    avoid = ""
    if job.get("avoid_stems"):
        avoid = ("Already asked from this text (do NOT test these facts again):\n"
                 + "\n".join(f"  - {s}" for s in job["avoid_stems"]) + "\n")
    return (f"=== JOB {job['job_id']} ===\n"
            f"Subject: {job['subject']} ({so}) | Grade: {job['grade']}\n"
            f"Chapter: {job['chapter']}\n"
            f"Section: {' / '.join(job['section_titles'])}\n"
            f"Slots:\n{slots}\n{avoid}"
            f"--- SOURCE TEXT ---\n{text}\n--- END SOURCE TEXT ---\n")


def build_prompt_text(instr: str, group: list[dict], sections: dict) -> str:
    jobs = "\n".join(job_block(j, sections) for j in group)
    ids = ", ".join(j["job_id"] for j in group)
    return (f"{instr}\n\n# Jobs\n\nThere {'is 1 job' if len(group) == 1 else f'are {len(group)} jobs'} below. "
            "Each job has its own source text: write each job's questions ONLY from that job's text.\n\n"
            f"{jobs}\nReturn {{\"jobs\": [...]}} with one entry per job ({ids}), each with one question object "
            "per slot, in slot order.")


def load_jobs(paths: Paths, wave: int) -> list[dict]:
    src = paths.plan / ("jobs_topup.jsonl" if wave == 3 else "jobs.jsonl")
    if not src.exists():
        raise SystemExit(f"missing {src}: run s01_plan.py{' --topup' if wave == 3 else ''} first")
    return [j for j in read_jsonl(src) if j["wave"] == wave]


def main():
    ap = argparse.ArgumentParser(description="Generate MCQs with Claude (subscription only)")
    level_arg(ap)
    ap.add_argument("--wave", type=int, required=True, choices=[0, 1, 2, 3])
    ap.add_argument("--workers", type=int, default=2, help="parallel CLI calls (the usage limit is shared)")
    ap.add_argument("--jobs-per-call", type=int, default=None, help="default: generator.jobs_per_call in the config (1)")
    ap.add_argument("--max-calls", type=int, default=None, help="stop after N calls (e.g. to try a few first)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and exit, no calls")
    ap.add_argument("--allow-unverified-extra-usage", action="store_true")
    args = ap.parse_args()
    stdout_utf8()
    paths, cfg = Paths(args.level), load_config(args.level)
    ensure_dirs(paths)
    gcfg = cfg["generator"]
    model, effort = args.model or gcfg["model"], args.effort or gcfg["effort"]
    per_call = args.jobs_per_call or gcfg.get("jobs_per_call", 1)

    sections = {s["id"]: s for s in read_jsonl(paths.plan / "sections.jsonl")}
    jobs = load_jobs(paths, args.wave)
    out_file = paths.wave_file(args.wave)
    done = {r["job_id"] for r in read_jsonl_tolerant(out_file) if r.get("status") == "ok"}
    todo = [j for j in jobs if j["job_id"] not in done]
    instr = instructions(args.level)
    instr_sha = sha1(instr)
    snap = paths.gen / "prompts" / f"generator_{instr_sha[:10]}.md"
    if not snap.exists():
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text(instr, encoding="utf-8")
    groups = [todo[i:i + per_call] for i in range(0, len(todo), per_call)]
    if args.max_calls:
        groups = groups[: args.max_calls]
    print(f"wave {args.wave}: {len(jobs)} jobs, {len(done)} done, {len(todo)} to do -> {len(groups)} calls "
          f"({per_call} job(s)/call), model {model}, effort {effort}, instructions {instr_sha[:10]}")
    if not read_json(CONFIG_DIR / f"exemplars_{args.level}.json").get("approved"):
        print("WARNING: exemplars not approved by the Sinhala reader yet (config/exemplars_*.json).")
    if args.dry_run:
        if groups:
            print("\n" + build_prompt_text(instr, groups[0], sections))
            print(f"\n[prompt: {len(build_prompt_text(instr, groups[0], sections))} characters]")
        return
    if not groups:
        print("nothing to do")
        return

    runner = ClaudeRunner(cfg["guard"], paths.cli_cwd, paths.gen / "usage_log.jsonl",
                          args.allow_unverified_extra_usage)
    runner.preflight()
    update_manifest(paths, f"generation_wave{args.wave}", {
        "model": model, "effort": effort, "instructions_sha1": instr_sha, "backend": "claude-code-cli (claude.ai subscription)",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    lock = threading.Lock()
    stats = {"ok": 0, "error": 0, "questions": 0, "skipped": 0}
    t0 = time.time()

    def run(group: list[dict]):
        prompt = build_prompt_text(instr, group, sections)
        tag = f"gen:w{args.wave}:{group[0]['job_id']}"
        try:
            res = runner.call(prompt, model=model, effort=effort, system=SYSTEM, schema=SCHEMA,
                              timeout=gcfg["timeout_s"], tag=tag)
        except GuardStop:
            raise
        except Exception as e:                     # recorded, retried on the next run
            with lock:
                for j in group:
                    append_jsonl(out_file, {"job_id": j["job_id"], "wave": args.wave, "status": "error",
                                            "error": str(e)[:500], "ts": time.time()})
                    stats["error"] += 1
            return
        got = {r.get("job_id"): r.get("questions", []) for r in res["data"].get("jobs", [])}
        if len(group) == 1 and group[0]["job_id"] not in got and len(got) == 1:
            got = {group[0]["job_id"]: next(iter(got.values()))}     # model mangled the id of a single job
        cost = (res["raw"].get("total_cost_usd") or 0) / len(group)
        with lock:
            for j in group:
                qs = got.get(j["job_id"])
                if qs is None:
                    append_jsonl(out_file, {"job_id": j["job_id"], "wave": args.wave, "status": "error",
                                            "error": "job missing from the model output", "ts": time.time()})
                    stats["error"] += 1
                    continue
                append_jsonl(out_file, {"job_id": j["job_id"], "wave": args.wave, "status": "ok", "model": model,
                                        "effort": effort, "instructions_sha1": instr_sha, "ts": time.time(),
                                        "notional_cost": round(cost, 5), "questions": qs})
                stats["ok"] += 1
                stats["questions"] += sum(1 for q in qs if not q.get("skip_reason"))
                stats["skipped"] += sum(1 for q in qs if q.get("skip_reason"))
            n = stats["ok"] + stats["error"]
            if n % 10 == 0 or n == len(todo):
                rate = n / max(time.time() - t0, 1) * 3600
                print(f"  {n}/{len(todo)} jobs ({stats['error']} errors), {stats['questions']} questions, "
                      f"{stats['skipped']} skipped slots, {rate:.0f} jobs/h", flush=True)

    run_pool(runner, run, groups, args.workers)
    print(f"\nwave {args.wave}: {stats}")
    print(usage_summary(paths.gen / "usage_log.jsonl", cfg["guard"]["window_hours"]))
    print(f"-> {out_file}")


if __name__ == "__main__":
    main()
