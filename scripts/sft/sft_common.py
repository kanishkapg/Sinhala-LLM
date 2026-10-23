"""Shared paths and helpers for the SFT dataset pipeline (see docs/prepare_sft_dataset.md).

Run every script from the repository root with the `sin-llm` environment:
    python scripts/sft/s01_plan.py --level easy
"""
import glob
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cpt"))
sys.path.insert(0, str(HERE.parent / "probes"))

from common import (ROOT, SEED, normalize, read_json, read_jsonl, sha1, sinhala_ratio,  # noqa: E402,F401
                    stdout_utf8, write_json, write_jsonl)
from probe_common import SUBJECT_MAP, build_prompt, load_json_items, near_duplicates, norm  # noqa: E402,F401

CONFIG_DIR = HERE / "config"
PROMPT_DIR = HERE / "prompts"
PIPELINE_VERSION = "sft-v1"

QTYPES = ["completion", "identification", "direct", "negation", "blank", "statement", "combination"]
COGNITIVE = ["recall", "understanding", "application"]
SHARED_TEST_GLOB = "data/SinhalaMMLU-SharedTask-Test/*/*.json"

# Subject names used by the shared-task test files -> our textbook subject names (metadata only, never content).
TEST_SUBJECT_ALIASES = {
    "Art": "Art", "Arts": "Art", "Buddhism": "Buddhism", "Catholicism": "Catholicism",
    "Christianity": "Christianity", "Civics": "Civics", "Dancing": "Dancing", "dancing": "Dancing",
    "Drama and Theatre": "Drama and Theatre", "drama_and_theatre": "Drama and Theatre",
    "Eastern_Music": "Eastern Music", "Eastern Music": "Eastern Music", "Eastern_music": "Eastern Music",
    "Geography": "Geography", "Health and Physical Education": "Health", "Health & Physical Education": "Health",
    "Health and Physical Science": "Health", "History": "History", "Islam": "Islam", "Science": "Science",
    "science": "Science", "Sinhala Language and Literature": "Sinhala",
    "Sinhala_language_and_literature": "Sinhala",
}


# ---------------------------------------------------------------- paths and config

def load_config(level: str) -> dict:
    path = CONFIG_DIR / f"sft_{level}.json"
    if not path.exists():
        raise SystemExit(f"missing config {path}")
    return read_json(path)


class Paths:
    def __init__(self, level: str):
        self.level = level
        self.cpt = ROOT / "data" / "processed" / "cpt" / level
        self.textbook = self.cpt / "04_dedup" / "textbook.jsonl"
        self.cpt_manifest = self.cpt / "manifest.json"
        self.probes = ROOT / "data" / "probes" / level
        # SFT_DATA_ROOT redirects all outputs (used for dry runs / tests outside the repo)
        self.root = Path(os.environ.get("SFT_DATA_ROOT") or ROOT / "data" / "processed" / "sft") / level
        self.plan = self.root / "00_plan"
        self.gen = self.root / "01_generated"
        self.items = self.root / "02_items"
        self.checked = self.root / "03_checked"
        self.validated = self.root / "04_validated"
        self.final = self.root / "05_final"
        self.rendered = self.root / "06_rendered"
        self.review = self.root / "review"
        self.manifest = self.root / "manifest.json"
        self.cli_cwd = self.root / ".cli_cwd"       # empty folder the Claude CLI runs in (no CLAUDE.md, no project)

    def wave_file(self, wave: int) -> Path:
        return self.gen / f"wave{wave}.jsonl"

    def all_wave_files(self) -> list[Path]:
        return sorted(self.gen.glob("wave*.jsonl"))


# ---------------------------------------------------------------- files

def append_jsonl(path: Path, row: dict) -> None:
    """Append one line and force it to disk: a crash or a dropped drive loses at most the line being written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl_tolerant(path: Path) -> list[dict]:
    """Like read_jsonl, but skips a truncated last line (left by a crash during an append)."""
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if n == len(lines):
                print(f"WARNING: {path.name}: ignoring a truncated last line")
            else:
                raise SystemExit(f"{path}:{n}: invalid JSON")
    return rows


def update_manifest(paths: Paths, key: str, value) -> None:
    m = read_json(paths.manifest, {}) or {}
    m.setdefault("pipeline_version", PIPELINE_VERSION)
    m.setdefault("level", paths.level)
    m[key] = value
    m["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json(paths.manifest, m)


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- deterministic randomness

def stable_int(*parts) -> int:
    """Seeded, order-independent integer from any parts (used instead of random() wherever stability across
    re-runs matters: positions, splits, subject-string variants)."""
    return int(hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12], 16)


def seeded_rng(*parts) -> random.Random:
    return random.Random(stable_int(*parts))


# ---------------------------------------------------------------- subjects

def subject_info(subject: str) -> tuple[str, str, str]:
    """Textbook subject -> (SinhalaMMLU subject, category, official subject_original)."""
    if subject not in SUBJECT_MAP:
        raise SystemExit(f"subject {subject!r} is not in probe_common.SUBJECT_MAP")
    return SUBJECT_MAP[subject]


def subject_string_variants() -> dict[str, list[str]]:
    """Every subject_original string the model may meet at test time: the Probe A / SUBJECT_MAP string plus the
    strings in the shared-task test files' metadata (section 3.3 of the doc). Metadata only."""
    variants = defaultdict(set)
    for subj, (_, _, so) in SUBJECT_MAP.items():
        variants[subj].add(so)
    for path in sorted(glob.glob(str(ROOT / "data/SinhalaMMLU-SharedTask-Test/easy/*.json"))):
        with open(path, encoding="utf-8") as f:
            for it in json.load(f):
                ours = TEST_SUBJECT_ALIASES.get(it["subject"])
                if ours:
                    variants[ours].add(it["metadata"]["subject_original"])
    return {k: sorted(v) for k, v in variants.items()}


# ---------------------------------------------------------------- Sinhala text helpers

def nozwj(s: str) -> str:
    """Remove ZWJ / ZWNJ so marker matching does not depend on how a conjunct was typed."""
    return re.sub("[‌‍]", "", s)


def has_marker(text: str, markers: list[str]) -> bool:
    t = nozwj(text)
    return any(nozwj(m) in t for m in markers)


WORD_RE = re.compile(r"[඀-෿‍]+|[A-Za-z]+|\d+")


def words(text: str) -> set[str]:
    return {nozwj(w) for w in WORD_RE.findall(text) if len(nozwj(w)) >= 3}


def combination_letters(n: int) -> list[str]:
    return ["A", "B", "C", "D"][:n]


# ---------------------------------------------------------------- reporting

def share_table(counter: Counter, total: int | None = None) -> str:
    total = total or sum(counter.values()) or 1
    return ", ".join(f"{k}: {v} ({100 * v / total:.1f}%)" for k, v in counter.most_common())


def ensure_dirs(paths: Paths) -> None:
    for d in [paths.plan, paths.gen, paths.items, paths.checked, paths.validated, paths.final, paths.rendered,
              paths.review, paths.cli_cwd]:
        d.mkdir(parents=True, exist_ok=True)


def level_arg(ap) -> None:
    ap.add_argument("--level", default="easy", choices=["easy", "medium", "hard"])
