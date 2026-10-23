"""Run Claude through the Claude Code CLI on your claude.ai subscription (Pro / Max), never on API billing.

Guards, in order:
  1. The child process never sees an API key: ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, ANTHROPIC_BASE_URL and the
     Bedrock / Vertex / Foundry switches are removed from its environment, so the CLI can only use the claude.ai
     login. `--bare` (which forces API-key auth) is never used.
  2. Pre-flight: `claude auth status` must say authMethod "claude.ai" and a subscription in `allowed_subscriptions`.
  3. Extra usage: every response reports `fast_mode_disabled_reason`; unless it says "extra_usage_disabled" the run
     stops before the next call (turn extra usage off at claude.ai -> Settings -> Usage, or pass
     --allow-unverified-extra-usage if you have checked it yourself).
  4. Usage limit reached (5-hour session or weekly): all workers pause until the reset time printed by the CLI
     (or `limit_wait_minutes` if no time is given), then the next real call checks again. Nothing is ever retried
     in a way that could spend money: with extra usage off, a limited call is simply refused.
  5. Optional own cap: `max_notional_usd_per_window` stops early within each `window_hours` window, measured by the
     CLI's `total_cost_usd` (an API-price *estimate*; on a subscription nothing is charged per call). Use it to keep
     part of your Pro allowance free for your own chats.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from sft_common import append_jsonl, read_jsonl_tolerant

SCRUB_ENV = ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_PROFILE",
             "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY",
             "AWS_BEARER_TOKEN_BEDROCK", "ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_IDENTITY_TOKEN",
             "ANTHROPIC_IDENTITY_TOKEN_FILE", "CLAUDE_CODE_SIMPLE"]

LIMIT_RE = re.compile(r"usage limit|limit reached|hit your (?:\w+ )?limit|session limit|weekly limit|"
                      r"out of (?:extra )?usage|rate.?limit|resets? (?:at|in)\b", re.I)
AUTH_RE = re.compile(r"invalid api key|please run /login|not logged in|authentication|unauthori[sz]ed|oauth", re.I)
TRANSIENT_RE = re.compile(r"overloaded|timed? ?out|econnreset|network|socket|503|502|500|529", re.I)


class LimitReached(Exception):
    def __init__(self, message: str, reset_at: float | None):
        super().__init__(message)
        self.reset_at = reset_at


class GuardStop(SystemExit):
    """A guard failed: stop the whole run (never retried)."""


def find_claude(configured: str | None) -> str:
    cands = [configured, os.environ.get("CLAUDE_EXE"), shutil.which("claude"), shutil.which("claude.exe")]
    home = Path.home()
    for pattern in [home / ".vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude.exe",
                    home / ".vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude",
                    home / ".local/bin/claude", home / ".claude/local/claude"]:
        found = sorted(glob.glob(str(pattern)))
        if found:
            cands.append(found[-1])                  # highest extension version
    for c in cands:
        if c and Path(c).exists():
            return str(c)
    raise GuardStop("Claude Code CLI not found. Install it (https://claude.com/claude-code) or set CLAUDE_EXE / "
                    "guard.claude_exe to the path of claude(.exe).")


def parse_reset(text: str) -> float | None:
    """Reset time from a usage-limit message, as a Unix time. Handles 'limit reached|1761234567',
    'resets 5pm', 'resets at 17:30', 'resets in 2 hours'. None if no time is given."""
    m = re.search(r"\|(\d{10})\b", text)
    if m:
        return float(m[1])
    now = datetime.now()
    m = re.search(r"resets?\s+in\s+(\d+)\s*(h|hours?|hrs?|m|min|minutes?)\b", text, re.I)
    if m:
        n = int(m[1])
        return (now + (timedelta(hours=n) if m[2].lower().startswith("h") else timedelta(minutes=n))).timestamp()
    m = re.search(r"resets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text, re.I)
    if m:
        h, mi, ap = int(m[1]), int(m[2] or 0), (m[3] or "").lower()
        if ap == "pm" and h < 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        if h < 24 and mi < 60:
            t = now.replace(hour=h, minute=mi, second=0, microsecond=0)
            if t <= now:
                t += timedelta(days=1)
            return t.timestamp()
    return None


def fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


class ClaudeRunner:
    def __init__(self, guard: dict, cwd: Path, usage_log: Path, allow_unverified_extra_usage: bool = False):
        self.guard = guard
        self.exe = find_claude(guard.get("claude_exe"))
        self.cwd = cwd
        self.cwd.mkdir(parents=True, exist_ok=True)
        self.usage_log = usage_log
        self.allow_unverified = allow_unverified_extra_usage
        self.env = {k: v for k, v in os.environ.items() if k not in SCRUB_ENV}
        self.env["PYTHONIOENCODING"] = "utf-8"
        self.lock = threading.Lock()
        self.pause_until = 0.0
        self.stopped: str | None = None
        self.calls_ok = 0
        self.extra_usage_state: str | None = None

    # ------------------------------------------------------------ pre-flight

    def preflight(self) -> dict:
        removed = [k for k in SCRUB_ENV if k in os.environ]
        if removed:
            print(f"guard: removed {', '.join(removed)} from the CLI's environment (subscription login only)")
        out = subprocess.run([self.exe, "auth", "status", "--json"], capture_output=True, env=self.env,
                             cwd=self.cwd, timeout=60)
        try:
            st = json.loads(out.stdout.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            raise GuardStop(f"could not read `claude auth status`: {out.stdout[:300]!r} {out.stderr[:300]!r}")
        if not st.get("loggedIn"):
            raise GuardStop("Claude CLI is not logged in: run `claude auth login` with your claude.ai account")
        if st.get("authMethod") != "claude.ai":
            raise GuardStop(f"Claude CLI auth method is {st.get('authMethod')!r}, not 'claude.ai': this would bill "
                            "an API account. Log in with your claude.ai (Pro) account.")
        sub = st.get("subscriptionType")
        if sub not in self.guard["allowed_subscriptions"]:
            raise GuardStop(f"subscription {sub!r} is not in allowed_subscriptions {self.guard['allowed_subscriptions']}")
        print(f"guard: claude.ai login OK ({st.get('email')}, subscription: {sub}); CLI {self.exe}")
        return st

    # ------------------------------------------------------------ pausing

    def _usage_in_window(self) -> tuple[float, float]:
        """(notional USD spent in the current window, time the oldest call in it leaves the window)."""
        win = self.guard["window_hours"] * 3600
        now = time.time()
        rows = [r for r in read_jsonl_tolerant(self.usage_log) if r.get("ok") and r["ts"] >= now - win]
        if not rows:
            return 0.0, now
        return sum(r.get("cost") or 0 for r in rows), min(r["ts"] for r in rows) + win

    def _wait(self) -> None:
        while True:
            if self.stopped:
                raise GuardStop(self.stopped)
            with self.lock:
                until = self.pause_until
                cap = self.guard.get("max_notional_usd_per_window")
                if cap and until <= time.time():
                    spent, frees_at = self._usage_in_window()
                    if spent >= cap:
                        self.pause_until = until = frees_at + 60
                        print(f"guard: own cap reached ({spent:.2f} >= {cap} notional USD in "
                              f"{self.guard['window_hours']} h): pausing until {fmt_time(until)}", flush=True)
            left = until - time.time()
            if left <= 0:
                return
            time.sleep(min(left, 300))

    def _pause(self, reset_at: float | None, message: str) -> None:
        with self.lock:
            if reset_at:
                until = reset_at + 120
                why = f"until the reset at {fmt_time(reset_at)}"
            else:
                until = time.time() + 60 * self.guard["limit_wait_minutes"]
                why = f"{self.guard['limit_wait_minutes']} min (no reset time in the message), then try again"
            if until > self.pause_until:
                self.pause_until = until
                print(f"\nguard: usage limit reached ({message.strip()[:160]!r}). Waiting {why}. "
                      "Nothing is billed while waiting; Ctrl+C is safe (finished jobs are saved).", flush=True)

    # ------------------------------------------------------------ one call

    def _cmd(self, model: str, effort: str, system: str, schema: dict) -> list[str]:
        cmd = [self.exe, "-p", "--model", model, "--effort", effort, "--output-format", "json",
               "--tools", "", "--safe-mode", "--no-session-persistence", "--disable-slash-commands",
               "--system-prompt", system, "--json-schema", json.dumps(schema, ensure_ascii=False)]
        if self.guard.get("max_budget_usd_per_call"):
            cmd += ["--max-budget-usd", str(self.guard["max_budget_usd_per_call"])]
        return cmd

    def _log(self, tag: str, model: str, effort: str, res: dict | None, ok: bool, note: str = "") -> None:
        u = (res or {}).get("usage") or {}
        append_jsonl(self.usage_log, {
            "ts": time.time(), "tag": tag, "model": model, "effort": effort, "ok": ok,
            "cost": (res or {}).get("total_cost_usd"), "input_tokens": u.get("input_tokens"),
            "output_tokens": u.get("output_tokens"), "cache_read": u.get("cache_read_input_tokens"),
            "cache_creation": u.get("cache_creation_input_tokens"), "note": note[:200]})

    def call(self, prompt: str, *, model: str, effort: str, system: str, schema: dict, timeout: int,
             tag: str) -> dict:
        """Returns {'data': structured output, 'raw': CLI result JSON}. Raises GuardStop on a guard failure and
        RuntimeError after `max_retries` non-limit failures."""
        attempts = 0
        while True:
            self._wait()
            attempts += 1
            try:
                p = subprocess.run(self._cmd(model, effort, system, schema), input=prompt.encode("utf-8"),
                                   capture_output=True, env=self.env, cwd=self.cwd, timeout=timeout)
            except subprocess.TimeoutExpired:
                self._log(tag, model, effort, None, False, "timeout")
                if attempts >= self.guard["max_retries"]:
                    raise RuntimeError(f"timeout after {timeout}s ({attempts} attempts)")
                continue
            stdout = p.stdout.decode("utf-8", "replace").strip()
            stderr = p.stderr.decode("utf-8", "replace").strip()
            res = None
            for line in reversed(stdout.splitlines() or [""]):
                try:
                    cand = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(cand, dict) and cand.get("type") == "result":
                    res = cand
                    break
            if res is None:
                try:
                    res = json.loads(stdout)
                except json.JSONDecodeError:
                    res = None
            text = " ".join(str(x) for x in [(res or {}).get("result"), (res or {}).get("api_error_status"),
                                             stderr, "" if res else stdout[:500]] if x)
            failed = res is None or res.get("is_error") or res.get("subtype") != "success"
            if failed and (LIMIT_RE.search(text) or (res or {}).get("api_error_status") == 429):
                self._log(tag, model, effort, res, False, "limit: " + text)
                self._pause(parse_reset(text), text)
                attempts -= 1                         # waiting for a reset is not a failed attempt
                continue
            if failed and AUTH_RE.search(text):
                self.stopped = f"authentication problem, stopping: {text[:300]}"
                raise GuardStop(self.stopped)
            if res is not None:
                self._check_extra_usage(res)
            data = (res or {}).get("structured_output")
            if data is None and res and isinstance(res.get("result"), str):
                try:
                    data = json.loads(re.sub(r"^```(?:json)?|```$", "", res["result"].strip()))
                except json.JSONDecodeError:
                    data = None
            if failed or data is None:
                self._log(tag, model, effort, res, False, "error: " + text)
                if attempts >= self.guard["max_retries"]:
                    raise RuntimeError(f"CLI call failed ({attempts} attempts): {text[:400]}")
                time.sleep(min(60, 10 * attempts) if TRANSIENT_RE.search(text) else 5)
                continue
            self._log(tag, model, effort, res, True)
            with self.lock:
                self.calls_ok += 1
            return {"data": data, "raw": res}

    def _check_extra_usage(self, res: dict) -> None:
        state = res.get("fast_mode_disabled_reason")
        self.extra_usage_state = state
        if not self.guard.get("require_extra_usage_disabled") or self.allow_unverified:
            return
        if state != "extra_usage_disabled":
            self.stopped = (f"extra usage is not confirmed OFF (CLI reports fast_mode_disabled_reason={state!r}). "
                            "With extra usage on, calls past your Pro limit would be billed. Turn it off at "
                            "claude.ai -> Settings -> Usage, or re-run with --allow-unverified-extra-usage if you "
                            "have checked it yourself.")
            raise GuardStop(self.stopped)


def run_pool(runner: "ClaudeRunner", fn, groups: list, workers: int) -> None:
    """Run fn(group) on a thread pool. On a guard stop or Ctrl+C, queued calls are cancelled (not drained);
    calls already running finish and are saved."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [ex.submit(fn, g) for g in groups]
        for f in as_completed(futures):
            f.result()
    except GuardStop as e:
        runner.stopped = runner.stopped or str(e)
        print(f"\nSTOPPED by a guard: {e}")
    except KeyboardInterrupt:
        runner.stopped = "interrupted"
        print("\ninterrupted: finished jobs are saved; re-run the same command to continue "
              "(waiting for the calls already running)")
    finally:
        ex.shutdown(wait=True, cancel_futures=True)


def usage_summary(usage_log: Path, window_hours: float) -> str:
    rows = read_jsonl_tolerant(usage_log)
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return "no successful calls logged yet"
    limits = [r for r in rows if str(r.get("note", "")).startswith("limit")]
    cost = sum(r.get("cost") or 0 for r in ok)
    by_model = {}
    for r in ok:
        m = by_model.setdefault(r["model"], [0, 0.0, 0, 0])
        m[0] += 1
        m[1] += r.get("cost") or 0
        m[2] += r.get("input_tokens") or 0
        m[3] += r.get("output_tokens") or 0
    lines = [f"{len(ok)} successful calls, {len(limits)} limit hits, notional API-price total ${cost:.2f} "
             "(an estimate; nothing is billed per call on a subscription with extra usage off)"]
    for m, (n, c, i, o) in by_model.items():
        lines.append(f"  {m}: {n} calls, notional ${c:.2f} (${c / n:.3f}/call), output tokens/call {o / n:.0f}")
    # calls completed between consecutive limit hits = what one usage window holds
    if limits:
        ts = [r["ts"] for r in limits]
        first_ok = min(r["ts"] for r in ok)
        n_before = sum(1 for r in ok if first_ok <= r["ts"] < ts[0])
        lines.append(f"  calls before the first limit hit: {n_before} (≈ calls per {window_hours} h window)")
    return "\n".join(lines)
