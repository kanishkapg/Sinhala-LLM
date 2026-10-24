"""Fill in the missing `answer` key of the SinhalaMMLU shared-task test set with Claude, and write one JSONL per
level to data/SinhalaMMLU-SharedTask-Test/with_answers/.

The shared-task test files (data/SinhalaMMLU-SharedTask-Test/{easy,medium,hard}/*.json) ship questions and choices
only. This script asks Claude for the correct choice of every question and writes, per level, one JSONL line per
question: the original object with `"answer": <1-based choice number>` inserted after `choices`, exactly like the
labelled files in data/SinhalaMMLU/.

    python scripts/answer_sharedtask_test.py --dry-run            # print the first prompt and the call plan, no calls
    python scripts/answer_sharedtask_test.py --levels easy        # answer easy (642 questions)
    python scripts/answer_sharedtask_test.py                      # all three levels (2238 questions)
    python scripts/answer_sharedtask_test.py --pass 2             # re-ask the low/medium-confidence ones (2nd opinion)
    python scripts/answer_sharedtask_test.py --pass 3             # tie-break only where pass 1 and 2 disagree
    python scripts/answer_sharedtask_test.py --rebuild-only       # rewrite the JSONLs from the cache, no calls
    python scripts/answer_sharedtask_test.py --calibrate 40 --levels easy   # accuracy check against gold labels

Credits (same guards as scripts/sft, see scripts/sft/claude_runner.py):
  * The Claude Code CLI runs on your claude.ai subscription; every API-key / Bedrock / Vertex variable is removed
    from its environment first, so a call can never fall back to paid API billing.
  * Pre-flight refuses to start unless `claude auth status` reports a claude.ai login with an allowed subscription.
  * Every response must report extra usage OFF (`fast_mode_disabled_reason == "extra_usage_disabled"`); otherwise the
    run stops before the next call, so nothing past the subscription allowance is ever billed. Override only with
    --allow-unverified-extra-usage, after checking claude.ai -> Settings -> Usage yourself.
  * When the 5-hour or weekly limit is hit, the run pauses until the reset the CLI reports and then continues by
    itself (--stop-on-limit exits instead). Nothing is billed while waiting.
  * --max-calls / --max-questions stop early; guard.max_notional_usd_per_window keeps part of the allowance free.

Resuming: every answered question is appended (and fsynced) to .cache/<level>.answers.jsonl as soon as its call
returns, keyed by question id *and* a hash of the question text. Ctrl+C, a usage limit, a crash or a reboot therefore
costs at most the calls in flight: re-run the same command and it continues with the unanswered questions only. The
output JSONLs are rebuilt from that cache at the end of every run, so they are never left half written.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _sub in ("sft", "cpt", "probes"):                  # reuse the SFT pipeline's billing guards and helpers
    sys.path.insert(0, str(HERE / _sub))

from claude_runner import ClaudeRunner, GuardStop, fmt_time, run_pool, usage_summary       # noqa: E402
from sft_common import (ROOT, append_jsonl, read_json, read_jsonl_tolerant, sha1, stdout_utf8,  # noqa: E402
                        write_json)

LEVELS = ["easy", "medium", "hard"]
TEST_DIR = ROOT / "data" / "SinhalaMMLU-SharedTask-Test"
OUT_DIR = TEST_DIR / "with_answers"
CACHE_DIR = OUT_DIR / ".cache"
N_CHOICES = 4
VERSION = "answer-key-v1"

DEFAULTS = {
    "model": "opus",
    "effort": "high",
    "batch_size": 10,              # questions per CLI call (one call answers a whole batch)
    "workers": 2,                  # parallel calls; the usage limit is shared, so 2-3 is plenty
    "timeout_s": 900,
    "recheck_confidence": ["low", "medium"],   # which pass-1 answers --pass 2 asks again
    "pass2": {"model": "opus", "effort": "high"},
    "pass3": {"model": "opus", "effort": "high"},
    "guard": {                     # overridden by scripts/sft/config/sft_easy.json's guard block when present
        "claude_exe": None,
        "allowed_subscriptions": ["pro", "max"],
        "require_extra_usage_disabled": True,
        "limit_wait_minutes": 30,
        "window_hours": 5,
        "max_notional_usd_per_window": None,
        "max_budget_usd_per_call": 3.0,
        "max_retries": 3,
    },
}

SYSTEM = ("You are an expert on the Sri Lankan school curriculum (grades 6-11, O/L and A/L) and a careful examiner. "
          "You answer Sinhala multiple-choice questions exactly as the official marking scheme would, and you reply "
          "only through the required JSON schema.")

INSTRUCTIONS = """\
Answer each Sinhala multiple-choice question below as the official Sri Lankan marking scheme would.

Rules:
  * Every question has exactly 4 numbered choices and exactly one correct answer. Give its number, 1-4.
  * Answer from the Sri Lankan school syllabus for the subject and grade given, not from general world knowledge,
    wherever the two could differ (spelling conventions, religious terms, local geography, prescribed texts).
  * Read the stem to the end before choosing. Watch for negation and superlatives — නොවන්නේ, නොවේ, වැරදි, අයත්
    නොවන්නේ, වඩාත්ම, හොඳම — where the answer is the choice that does NOT fit.
  * For questions that list statements A, B, C and ask which are correct, decide each statement separately first,
    then pick the choice naming exactly that set.
  * For calculations, work the arithmetic through and match the result to a choice; do not pick by resemblance.
  * Answer every id in the list exactly once, even when unsure: give your best choice with confidence "low" rather
    than leaving it out.

For each question return:
  qid          the id exactly as written in its === header ===
  answer       the number of the correct choice, 1-4
  confidence   "high"   = you know the syllabus fact and the choice is unambiguous
               "medium" = you reached it by reasoning or elimination
               "low"    = a guess between choices you cannot separate
  reason       at most 20 English words: the fact or step that decides it (used for human review)
"""

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["answers"],
    "properties": {"answers": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["qid", "answer", "confidence", "reason"],
        "properties": {
            "qid": {"type": "string"},
            "answer": {"type": "integer", "minimum": 1, "maximum": N_CHOICES},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "reason": {"type": "string"}}}}},
}


# ---------------------------------------------------------------- config

def load_config(path: str | None) -> dict:
    """DEFAULTS <- the SFT pipeline's guard block (one source of truth for the billing guards) <- --config file."""
    cfg = json.loads(json.dumps(DEFAULTS))
    sft_guard = read_json(HERE / "sft" / "config" / "sft_easy.json", {}) or {}
    if sft_guard.get("guard"):
        cfg["guard"].update({k: v for k, v in sft_guard["guard"].items() if k in cfg["guard"]})
    if path:
        user = read_json(Path(path))
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


# ---------------------------------------------------------------- questions

def qhash(item: dict) -> str:
    """Identity of the question itself: a cached answer is dropped if the question or its choices ever change."""
    return sha1("\x1f".join([item.get("question", "")] + list(item.get("choices", []))))[:16]


def load_items(level: str) -> list[dict]:
    files = sorted((TEST_DIR / level).glob("*.json"))
    if not files:
        raise SystemExit(f"no question files in {TEST_DIR / level}")
    out = []
    for path in files:
        items = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            raise SystemExit(f"{path}: expected a JSON array of questions")
        for pos, item in enumerate(items):
            if len(item.get("choices", [])) != N_CHOICES:
                raise SystemExit(f"{path}: q_no {item.get('q_no')} has {len(item.get('choices', []))} choices, "
                                 f"expected {N_CHOICES}")
            out.append({"uid": f"{path.stem}#{item['q_no']}", "level": level, "file": path.name, "pos": pos,
                        "qhash": qhash(item), "item": item})
    seen = Counter(r["uid"] for r in out)
    dups = [u for u, n in seen.items() if n > 1]
    if dups:
        raise SystemExit(f"{level}: duplicate question ids (file#q_no): {dups[:5]}")
    return out


def item_block(qid: str, item: dict) -> str:
    md = item.get("metadata") or {}
    subject = item.get("subject", "?")
    if md.get("subject_original"):
        subject += f" ({md['subject_original']})"
    facts = " | ".join(x for x in [f"Subject: {subject}",
                                   f"Category: {item['category']}" if item.get("category") else "",
                                   f"Grade: {md['grade']}" if md.get("grade") else "",
                                   f"Level: {md['difficulty']}" if md.get("difficulty") else ""] if x)
    choices = "\n".join(f"  {i}. {c}" for i, c in enumerate(item["choices"], 1))
    return f"=== {qid} ===\n{facts}\nQuestion: {item['question'].strip()}\nChoices:\n{choices}\n"


def build_prompt(group: list[dict]) -> str:
    qids = [f"q{i}" for i in range(1, len(group) + 1)]
    blocks = "\n".join(item_block(qid, r["item"]) for qid, r in zip(qids, group))
    return (f"{INSTRUCTIONS}\n# Questions\n\n{blocks}\n"
            f"Return {{\"answers\": [...]}} with exactly {len(group)} objects, one per id "
            f"({', '.join(qids)}), in that order.")


# ---------------------------------------------------------------- cache (resume) and resolution

def cache_file(level: str) -> Path:
    return CACHE_DIR / f"{level}.answers.jsonl"


def load_cache(path: Path, by_uid: dict[str, dict]) -> dict[str, list[dict]]:
    """uid -> its cached answer rows, oldest first. Rows for unknown ids or stale question text are ignored."""
    rows: dict[str, list[dict]] = {}
    stale = 0
    for r in read_jsonl_tolerant(path):
        known = by_uid.get(r.get("uid"))
        if not known or r.get("answer") not in range(1, N_CHOICES + 1):
            continue
        if r.get("qhash") != known["qhash"]:
            stale += 1
            continue
        rows.setdefault(r["uid"], []).append(r)
    for uid in rows:
        rows[uid].sort(key=lambda r: (r.get("pass", 1), r.get("ts", 0)))
    if stale:
        print(f"  {path.name}: ignored {stale} cached answer(s) whose question text has changed since")
    return rows


class StopOnLimitRunner(ClaudeRunner):
    """--stop-on-limit: a usage limit ends the run instead of waiting for the reset (nothing is billed either way)."""

    def _pause(self, reset_at: float | None, message: str) -> None:
        self.stopped = ("usage limit reached" + (f", resets at {fmt_time(reset_at)}" if reset_at else "")
                        + " -- stopping as asked (--stop-on-limit). Re-run to continue from the cache.")
        raise GuardStop(self.stopped)


def resolve(rows: list[dict]) -> dict:
    """One answer out of all passes for a question: majority vote, ties broken by the latest pass."""
    votes = Counter(r["answer"] for r in rows)
    top = max(votes.values())
    winners = [a for a, n in votes.items() if n == top]
    answer = rows[-1]["answer"] if len(winners) > 1 else winners[0]
    confs = [r.get("confidence", "low") for r in rows]
    return {"answer": answer, "passes": len(rows), "votes": dict(sorted(votes.items())),
            "unanimous": len(votes) == 1, "confidence": min(confs, key=["low", "medium", "high"].index),
            "needs_review": len(votes) > 1 or (len(rows) == 1 and confs[0] == "low"),
            "reason": rows[-1].get("reason", "")}


def needs_call(rows: list[dict], pass_no: int, recheck_conf: list[str], recheck_all: bool) -> bool:
    if not rows:
        return True                                        # never answered: every pass owes it a first answer
    if max(r.get("pass", 1) for r in rows) >= pass_no:
        return False                                       # this pass already answered it
    if pass_no == 1 or recheck_all:
        return True
    info = resolve(rows)
    if pass_no == 2:
        return info["confidence"] in recheck_conf or info["needs_review"]
    return not info["unanimous"]                           # pass 3+: only still-disagreeing questions


# ---------------------------------------------------------------- output

def with_answer(item: dict, answer: int) -> dict:
    """The original question object with `answer` inserted after `choices` (the labelled-file key order)."""
    out = {}
    for k, v in item.items():
        out[k] = v
        if k == "choices":
            out["answer"] = answer
    if "answer" not in out:
        out["answer"] = answer
    return out


def rebuild(level: str) -> dict:
    """Write with_answers/<level>.jsonl (+ .audit.jsonl) from the cache. Safe to run any time."""
    items = load_items(level)
    cache = load_cache(cache_file(level), {r["uid"]: r for r in items})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path, audit_path = OUT_DIR / f"{level}.jsonl", OUT_DIR / f"{level}.audit.jsonl"
    tmp_out, tmp_audit = out_path.with_suffix(".jsonl.tmp"), audit_path.with_suffix(".jsonl.tmp")
    conf, n, review = Counter(), 0, 0
    with open(tmp_out, "w", encoding="utf-8", newline="\n") as f, \
            open(tmp_audit, "w", encoding="utf-8", newline="\n") as fa:
        for rec in items:
            rows = cache.get(rec["uid"])
            if not rows:
                continue
            info = resolve(rows)
            f.write(json.dumps(with_answer(rec["item"], info["answer"]), ensure_ascii=False) + "\n")
            fa.write(json.dumps({"uid": rec["uid"], "file": rec["file"], "q_no": rec["item"]["q_no"],
                                 **{k: info[k] for k in ["answer", "confidence", "passes", "votes", "unanimous",
                                                         "needs_review", "reason"]}}, ensure_ascii=False) + "\n")
            conf[info["confidence"]] += 1
            review += bool(info["needs_review"])
            n += 1
    os.replace(tmp_out, out_path)
    os.replace(tmp_audit, audit_path)
    stat = {"questions": len(items), "answered": n, "missing": len(items) - n,
            "confidence": dict(conf), "needs_review": review,
            "file": str(out_path.relative_to(ROOT)).replace("\\", "/")}
    return stat


def write_manifest(stats: dict[str, dict], cfg: dict) -> None:
    m = read_json(OUT_DIR / "manifest.json", {}) or {}
    m.update({"version": VERSION, "source": "data/SinhalaMMLU-SharedTask-Test/<level>/*.json",
              "answer_index": "1-based, as in data/SinhalaMMLU",
              "generated_by": "scripts/answer_sharedtask_test.py (Claude Code CLI, claude.ai subscription)",
              "model": cfg["model"], "effort": cfg["effort"],
              "updated": time.strftime("%Y-%m-%dT%H:%M:%S")})
    m.setdefault("levels", {}).update(stats)
    write_json(OUT_DIR / "manifest.json", m)


# ---------------------------------------------------------------- answering

class Budget:
    """Shared stop-early counters, so --max-calls / --max-questions span all levels."""

    def __init__(self, max_calls: int | None, max_questions: int | None):
        self.lock = threading.Lock()
        self.calls_left = max_calls if max_calls is not None else float("inf")
        self.questions_left = max_questions if max_questions is not None else float("inf")

    def take(self, n_questions: int) -> bool:
        with self.lock:
            if self.calls_left < 1 or self.questions_left < n_questions:
                return False
            self.calls_left -= 1
            self.questions_left -= n_questions
            return True

    def plan(self, groups: list[list[dict]]) -> list[list[dict]]:
        keep, calls, qs = [], self.calls_left, self.questions_left
        for g in groups:
            if calls < 1 or qs < len(g):
                break
            calls, qs = calls - 1, qs - len(g)
            keep.append(g)
        return keep


def answer_batches(runner: ClaudeRunner, groups: list[list[dict]], *, out_file: Path, model: str, effort: str,
                   timeout: int, pass_no: int, level: str, workers: int, budget: Budget) -> dict:
    stats = Counter()
    lock = threading.Lock()
    t0 = time.time()
    total = sum(len(g) for g in groups)

    def run(group: list[dict]) -> None:
        if not budget.take(len(group)):
            return
        qids = {f"q{i}": r for i, r in enumerate(group, 1)}
        tag = f"ans:{level}:p{pass_no}:{group[0]['uid']}"
        try:
            res = runner.call(build_prompt(group), model=model, effort=effort, system=SYSTEM, schema=SCHEMA,
                              timeout=timeout, tag=tag)
        except GuardStop:
            raise
        except Exception as e:                      # not cached -> the next run asks these questions again
            with lock:
                stats["failed_calls"] += 1
                stats["missing"] += len(group)
                print(f"  call failed ({len(group)} questions left for the next run): {str(e)[:200]}", flush=True)
            return
        got, bad = {}, 0
        for a in res["data"].get("answers") or []:
            rec = qids.get(str(a.get("qid", "")).strip().lower())
            if rec is None or a.get("answer") not in range(1, N_CHOICES + 1) or rec["uid"] in got:
                bad += 1
                continue
            got[rec["uid"]] = (rec, a)
        cost = (res["raw"].get("total_cost_usd") or 0) / max(len(group), 1)
        with lock:
            for rec, a in got.values():
                append_jsonl(out_file, {"uid": rec["uid"], "qhash": rec["qhash"], "file": rec["file"],
                                        "q_no": rec["item"]["q_no"], "answer": int(a["answer"]),
                                        "confidence": a.get("confidence", "low"), "reason": (a.get("reason") or "")[:300],
                                        # the nth opinion for *this* question, not the --pass flag: a question first
                                        # answered during a --pass 2 run is still only on its first opinion
                                        "pass": rec.get("next_pass", pass_no), "model": model, "effort": effort,
                                        "notional_cost": round(cost, 5), "ts": time.time()})
                stats[a.get("confidence", "low")] += 1
            stats["answered"] += len(got)
            stats["missing"] += len(group) - len(got)
            stats["unusable_rows"] += bad
            stats["calls"] += 1
            done = stats["answered"] + stats["missing"]
            if stats["calls"] % 5 == 0 or done >= total:
                rate = done / max(time.time() - t0, 1) * 3600
                print(f"  {level} pass {pass_no}: {done}/{total} questions, {stats['answered']} answered "
                      f"({stats['high']} high / {stats['medium']} medium / {stats['low']} low), "
                      f"{rate:.0f} questions/h", flush=True)

    run_pool(runner, run, groups, workers)
    return dict(stats)


# ---------------------------------------------------------------- calibration (accuracy against gold labels)

def gold_items(level: str, n: int, seed: int) -> list[dict]:
    """Questions from the labelled data/SinhalaMMLU set for this level, to measure how good the answers are."""
    pool = []
    for path in sorted((ROOT / "data" / "SinhalaMMLU").rglob("*.json")):
        if level not in path.parts and level not in path.stem.lower():
            continue
        items = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            continue
        for item in items:
            if item.get("answer") in range(1, N_CHOICES + 1) and len(item.get("choices", [])) == N_CHOICES:
                pool.append({"uid": f"gold/{path.stem}#{item['q_no']}", "level": level, "file": path.name,
                             "qhash": qhash(item), "gold": item["answer"],
                             "item": {k: v for k, v in item.items() if k != "answer"}})
    random.Random(seed).shuffle(pool)
    return pool[:n]


def calibrate(runner: ClaudeRunner, level: str, cfg: dict, args, budget: Budget) -> None:
    items = gold_items(level, args.calibrate, args.seed)
    if not items:
        print(f"{level}: no labelled questions in data/SinhalaMMLU to calibrate against")
        return
    out_file = CACHE_DIR / f"calibration_{level}.jsonl"
    by_uid = {r["uid"]: r for r in items}
    cache = load_cache(out_file, by_uid)
    todo = [r for r in items if not cache.get(r["uid"])]
    groups = budget.plan([todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)])
    print(f"calibration {level}: {len(items)} gold questions, {len(items) - len(todo)} cached, {len(groups)} call(s)")
    if groups:
        answer_batches(runner, groups, out_file=out_file, model=cfg["model"], effort=cfg["effort"],
                       timeout=cfg["timeout_s"], pass_no=1, level=f"cal-{level}", workers=args.workers,
                       budget=budget)
    hits = [(by_uid[u]["gold"] == rs[-1]["answer"], rs[-1].get("confidence", "low"))
            for u, rs in load_cache(out_file, by_uid).items()]
    if not hits:
        return
    by_conf = Counter(c for _, c in hits)
    ok_conf = Counter(c for ok, c in hits if ok)
    acc = sum(1 for ok, _ in hits if ok) / len(hits)
    print(f"\ncalibration {level}: {sum(1 for ok, _ in hits if ok)}/{len(hits)} correct = {acc:.1%} "
          f"({cfg['model']}, effort {cfg['effort']})")
    for c in ["high", "medium", "low"]:
        if by_conf[c]:
            print(f"  {c:6s}: {ok_conf[c]}/{by_conf[c]} = {ok_conf[c] / by_conf[c]:.1%}")
    print(f"-> {out_file}")


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="Answer the shared-task test set with Claude (subscription only)")
    ap.add_argument("--levels", default="easy,medium,hard", help="comma-separated: easy,medium,hard")
    ap.add_argument("--pass", dest="pass_no", type=int, default=1, choices=[1, 2, 3],
                    help="1: answer every unanswered question; 2: ask the low/medium-confidence ones again; "
                         "3: tie-break the ones where the passes still disagree")
    ap.add_argument("--recheck-all", action="store_true", help="--pass 2/3: ask every question again, not just the "
                                                               "uncertain ones (costs a full pass)")
    ap.add_argument("--batch-size", type=int, default=None, help=f"questions per call (default {DEFAULTS['batch_size']})")
    ap.add_argument("--workers", type=int, default=None, help=f"parallel calls (default {DEFAULTS['workers']})")
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-calls", type=int, default=None, help="stop after N CLI calls (across all levels)")
    ap.add_argument("--max-questions", type=int, default=None, help="stop after N questions (across all levels)")
    ap.add_argument("--stop-on-limit", action="store_true", help="exit when the usage limit is hit instead of "
                                                                 "waiting for the reset (re-run later to continue)")
    ap.add_argument("--calibrate", type=int, metavar="N", default=0,
                    help="instead of answering: answer N labelled questions from data/SinhalaMMLU and report accuracy")
    ap.add_argument("--seed", type=int, default=42, help="--calibrate sample")
    ap.add_argument("--config", default=None, help="JSON file overriding the model/batch/guard defaults")
    ap.add_argument("--rebuild-only", action="store_true", help="rewrite the JSONLs from the cache, make no calls")
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and the call plan, make no calls")
    ap.add_argument("--allow-unverified-extra-usage", action="store_true",
                    help="skip the extra-usage-OFF check (only if you have verified it at claude.ai yourself)")
    args = ap.parse_args()
    stdout_utf8()

    cfg = load_config(args.config)
    cfg["model"] = args.model or cfg["model"]
    cfg["effort"] = args.effort or cfg["effort"]
    args.batch_size = args.batch_size or cfg["batch_size"]
    args.workers = args.workers or cfg["workers"]
    if args.pass_no > 1 and not (args.model or args.effort):
        cfg.update(cfg.get(f"pass{args.pass_no}") or {})
    levels = [l.strip() for l in args.levels.split(",") if l.strip()]
    if bad := [l for l in levels if l not in LEVELS]:
        raise SystemExit(f"unknown level(s) {bad}; choose from {LEVELS}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    budget = Budget(args.max_calls, args.max_questions)

    # ---- plan: what each level still owes this pass
    plan, prompts = {}, []
    for level in levels:
        items = load_items(level)
        cache = load_cache(cache_file(level), {r["uid"]: r for r in items})
        todo = []
        for r in items:
            rows = cache.get(r["uid"], [])
            if needs_call(rows, args.pass_no, cfg["recheck_confidence"], args.recheck_all):
                r["next_pass"] = max((x.get("pass", 1) for x in rows), default=0) + 1
                todo.append(r)
        groups = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
        plan[level] = {"items": items, "todo": todo, "groups": groups,
                       "answered": sum(1 for r in items if cache.get(r["uid"]))}
        if groups:
            prompts.append(groups[0])
        print(f"{level}: {len(items)} questions, {plan[level]['answered']} already answered, "
              f"{len(todo)} to ask in pass {args.pass_no} -> {len(groups)} call(s) of up to {args.batch_size}")

    if args.rebuild_only:
        stats = {level: rebuild(level) for level in levels}
        write_manifest(stats, cfg)
        for level, s in stats.items():
            print(f"{level}: {s['answered']}/{s['questions']} answered -> {s['file']}")
        return

    if args.dry_run:
        preview = Budget(args.max_calls, args.max_questions)
        total_calls = 0
        for p in plan.values():                      # same order and caps as the real run
            for g in preview.plan(p["groups"]):
                if preview.take(len(g)):
                    total_calls += 1
        capped = total_calls < sum(len(p["groups"]) for p in plan.values())
        print(f"\nplan: {total_calls} call(s){' (capped)' if capped else ''}, "
              f"model {cfg['model']}, effort {cfg['effort']}, "
              f"{args.workers} worker(s), timeout {cfg['timeout_s']}s")
        if prompts:
            print("\n" + "=" * 70 + "\nfirst prompt:\n" + "=" * 70)
            print(build_prompt(prompts[0]))
            print(f"[{len(build_prompt(prompts[0]))} characters]")
        return

    if args.calibrate:
        runner = ClaudeRunner(cfg["guard"], CACHE_DIR / ".cli_cwd", CACHE_DIR / "usage_log.jsonl",
                              args.allow_unverified_extra_usage)
        runner.preflight()
        for level in levels:
            calibrate(runner, level, cfg, args, budget)
        print("\n" + usage_summary(CACHE_DIR / "usage_log.jsonl", cfg["guard"]["window_hours"]))
        return

    if not any(p["groups"] for p in plan.values()):
        print("nothing to ask: rebuilding the output files from the cache")
        stats = {level: rebuild(level) for level in levels}
        write_manifest(stats, cfg)
        for level, s in stats.items():
            print(f"{level}: {s['answered']}/{s['questions']} answered -> {s['file']}")
        return

    cls = StopOnLimitRunner if args.stop_on_limit else ClaudeRunner
    runner = cls(cfg["guard"], CACHE_DIR / ".cli_cwd", CACHE_DIR / "usage_log.jsonl",
                 args.allow_unverified_extra_usage)
    runner.preflight()

    totals, stats = Counter(), {}
    try:
        for level in levels:
            groups = budget.plan(plan[level]["groups"])
            if groups:
                print(f"\n{level}: asking {sum(len(g) for g in groups)} question(s) in {len(groups)} call(s)")
                got = answer_batches(runner, groups, out_file=cache_file(level), model=cfg["model"],
                                     effort=cfg["effort"], timeout=cfg["timeout_s"], pass_no=args.pass_no,
                                     level=level, workers=args.workers, budget=budget)
                totals.update(got)
            if runner.stopped:
                break
    finally:
        for level in levels:
            stats[level] = rebuild(level)
        write_manifest(stats, cfg)

    print(f"\npass {args.pass_no} totals: {dict(totals)}")
    for level in levels:
        s = stats[level]
        print(f"{level}: {s['answered']}/{s['questions']} answered, confidence {s['confidence']}, "
              f"{s['needs_review']} to review -> {s['file']}")
    print("\n" + usage_summary(CACHE_DIR / "usage_log.jsonl", cfg["guard"]["window_hours"]))
    missing = sum(s["missing"] for s in stats.values())
    if missing:
        print(f"\n{missing} question(s) still unanswered: re-run the same command to continue "
              "(answered ones are never asked again).")
    else:
        print("\nAll questions answered. Next: `--pass 2` for a second opinion on the uncertain ones, and review "
              f"the needs_review rows in {OUT_DIR.relative_to(ROOT)}/<level>.audit.jsonl")
    if runner.stopped:
        raise SystemExit(f"stopped: {runner.stopped}")


if __name__ == "__main__":
    main()
