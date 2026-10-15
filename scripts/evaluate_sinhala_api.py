"""Evaluate SinhalaMMLU through the Google AI Studio (Gemini) API instead of local weights.

Why this exists: it needs no 8.6GB download and no GPU. Use it for fast iteration.

IMPORTANT — read before submitting results from this script:
  * Gemma 3 is no longer served on the Gemini API. The only Gemma models offered are
    gemma-4-26b-a4b-it (26B total / 4B active) and gemma-4-31b-it (31B). Both are over
    the shared task's 8B parameter cap that evaluate_sinhala.py asserts, so numbers from
    here are NOT a valid competition submission. Use evaluate_sinhala.py for that.
  * The API does not expose logprobs for these models ("Logprobs is not enabled for this
    model"), so the answer is parsed out of generated text rather than scored from the
    logits over tokens "1".."5". A reply that contains no digit is recorded as unparsed.

Usage:
    python scripts/evaluate_sinhala_api.py
    $env:API_MODEL = "gemma-4-31b-it"; python scripts/evaluate_sinhala_api.py
"""

import glob
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8")  # Sinhala text on a cp1252 console


def load_env(path=".env"):
    """Read GOOGLE_API_KEY (and friends) out of .env into the environment."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env()

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise SystemExit("GOOGLE_API_KEY is not set. Add it to .env as GOOGLE_API_KEY=...")

MODEL = os.environ.get("API_MODEL", "gemma-4-26b-a4b-it")
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

# Free-tier limits are per project and Google does not publish them per model; 30 RPM is
# the widely reported Gemma figure. Lower RPM if you see sustained 429s.
RPM = int(os.environ.get("API_RPM", "30"))
WORKERS = int(os.environ.get("API_WORKERS", "8"))
MAX_TOKENS = int(os.environ.get("API_MAX_TOKENS", "1024"))  # Gemma 4 thinks; it needs room

DATA_ROOT = "data/SinhalaMMLU/TEST"
LEVELS = ["easy", "medium", "hard"]
CHECKPOINT = "api_predictions.checkpoint.jsonl"


class RateLimiter:
    """Spread requests so we never exceed RPM calls in any rolling minute."""

    def __init__(self, per_minute):
        self._interval = 60.0 / max(per_minute, 1)
        self._lock = threading.Lock()
        self._next = time.monotonic()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._interval
        if wait:
            time.sleep(wait)


limiter = RateLimiter(RPM)


def build_prompt(item):
    """Same template as the local script, plus an explicit digit-only instruction.

    The local script never needs this: it reads the logits for "1".."5" directly. Here the
    answer has to survive a round trip through free text, so we ask for a bare digit.
    """
    subject = item.get("metadata", {}).get("subject_original", item["subject"])
    n_choices = len(item["choices"])
    labels = ", ".join(str(i) for i in range(1, n_choices + 1))
    options = "\n".join(f"{i}. {choice.strip()}" for i, choice in enumerate(item["choices"], 1))
    return (
        f"මෙය {subject} විෂයයට අදාළ බහුවරණ ප්‍රශ්නයකි.\n"
        f"පහත ප්‍රශ්නයට {labels} යන පිළිතුරුවලින් නිවැරදි හෝ ඉතාමත් ගැළපෙන පිළිතුර තෝරාගන්න.\n\n"
        f"ප්‍රශ්නය: {item['question']}\n{options}\n\n"
        f"Answer with only the digit {1}-{n_choices}. No words, no explanation."
    )


def answer_text(candidate):
    """Return the reply, skipping Gemma 4's reasoning parts (which carry thought=True)."""
    for part in candidate.get("content", {}).get("parts", []):
        if not part.get("thought"):
            return part.get("text", "")
    return ""


def call_api(prompt, attempts=5):
    """POST one prompt, retrying 429/500/503 with exponential backoff and jitter."""
    body = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": MAX_TOKENS, "temperature": 0},
    }).encode()

    for attempt in range(attempts):
        limiter.acquire()
        request = urllib.request.Request(f"{ENDPOINT}?key={API_KEY}")
        request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, body, timeout=120) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as e:
            retryable = e.code in (429, 500, 502, 503, 504)
            if not retryable or attempt == attempts - 1:
                detail = e.read().decode(errors="replace")[:200]
                raise RuntimeError(f"HTTP {e.code}: {detail}") from e
            time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
    raise RuntimeError("unreachable")


def predict(item):
    """Return (predicted_choice_or_None, raw_reply). None means nothing parseable came back."""
    n_choices = len(item["choices"])
    response = call_api(build_prompt(item))
    raw = answer_text(response["candidates"][0])

    match = re.search(r"[1-9]", raw)
    if not match:
        return None, raw
    choice = int(match.group())
    if not 1 <= choice <= n_choices:
        return None, raw
    return choice, raw


def load_checkpoint():
    """Resume a previous run; a full sweep takes hours and free-tier quota resets daily."""
    done = {}
    if not os.path.exists(CHECKPOINT):
        return done
    with open(CHECKPOINT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # half-written final line from an interrupted run
            done[(row["level"], row["q_no"])] = row
    return done


def main():
    print(f"Model: {MODEL}  |  {WORKERS} workers  |  {RPM} req/min")
    results = load_checkpoint()
    if results:
        print(f"Resuming: {len(results)} answers already in {CHECKPOINT}")

    tasks = []
    for level in LEVELS:
        for path in sorted(glob.glob(os.path.join(DATA_ROOT, level, "*.json"))):
            with open(path, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    if (level, str(item["q_no"])) not in results:
                        tasks.append((level, item))

    limit = int(os.environ.get("API_LIMIT", "0"))  # smoke-test a slice before the full run
    if limit:
        tasks = tasks[:limit]

    if not tasks:
        print("Nothing left to do; writing output from the checkpoint.")
    else:
        print(f"{len(tasks)} questions to send\n")

    write_lock = threading.Lock()
    counter = {"done": 0, "unparsed": 0, "errors": 0}
    started = time.time()

    with open(CHECKPOINT, "a", encoding="utf-8") as ckpt:

        def work(task):
            level, item = task
            try:
                choice, raw = predict(item)
            except Exception as e:
                with write_lock:
                    counter["errors"] += 1
                    print(f"  [{level}] q{item['q_no']} failed: {e}")
                return

            row = {
                "q_no": str(item["q_no"]),
                "subject": item["subject"],
                "category": item.get("category", ""),
                "level": level,
                "answer": str(choice) if choice else "",
                "gold": str(item.get("answer", "")),
                "raw": raw[:200],
            }
            with write_lock:
                results[(level, row["q_no"])] = row
                ckpt.write(json.dumps(row, ensure_ascii=False) + "\n")
                ckpt.flush()  # survive a Ctrl+C mid-run
                counter["done"] += 1
                if choice is None:
                    counter["unparsed"] += 1
                n = counter["done"]
                if n % 25 == 0 or n == len(tasks):
                    rate = n / max(time.time() - started, 1e-9)
                    eta = (len(tasks) - n) / rate / 60 if rate else 0
                    print(f"  {n}/{len(tasks)}  unparsed={counter['unparsed']}  "
                          f"errors={counter['errors']}  ETA {eta:.0f} min")

        if tasks:
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                list(pool.map(work, tasks))

    # Score whatever has gold answers (the hidden test set has none)
    print("\n" + "=" * 46)
    accuracy = {}
    for level in LEVELS:
        rows = [r for (lv, _), r in results.items() if lv == level]
        graded = [r for r in rows if r.get("gold") not in (None, "")]
        if graded:
            correct = sum(r["answer"] == r["gold"] for r in graded)
            accuracy[level] = correct / len(graded)
            print(f"{level:>6}: {accuracy[level]:.2%} ({correct}/{len(graded)})")
    if accuracy:
        print(f" macro: {sum(accuracy.values()) / len(accuracy):.2%}   (official metric)")

    unparsed = sum(1 for r in results.values() if not r["answer"])
    if unparsed:
        print(f"\nWARNING: {unparsed} replies had no parseable digit and are blank "
              f"in the submission. Rerun to retry them, or raise API_MAX_TOKENS.")
    print("=" * 46)

    ordered = [results[k] for k in sorted(results, key=lambda k: (LEVELS.index(k[0]), k[1]))]
    with open("predictions.jsonl", "w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps({k: row[k] for k in
                                ("q_no", "subject", "category", "level", "answer")},
                               ensure_ascii=False) + "\n")
    with zipfile.ZipFile("predictions.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("predictions.jsonl")
    print(f"Saved {len(ordered)} predictions to predictions.zip")


if __name__ == "__main__":
    main()
