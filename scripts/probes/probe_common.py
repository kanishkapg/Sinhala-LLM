"""Shared paths and helpers for the MCQ probes (see docs/prepare_cpt_training_pipeline.md, section 8.3).

Run the probe scripts from the repo root: python scripts/probes/<script>.py
"""
import difflib
import glob
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpt"))
from common import ROOT, SEED, read_json, read_jsonl, stdout_utf8, write_json, write_jsonl  # noqa: E402,F401

LEVEL = "easy"
PROBE_DIR = ROOT / "data" / "probes" / LEVEL
AUTHORED_DIR = PROBE_DIR / "probe_b_authored"
CPT_DIR = ROOT / "data" / "processed" / "cpt" / LEVEL
TEXTBOOKS = CPT_DIR / "04_dedup" / "textbook.jsonl"
MIX_TRAIN = CPT_DIR / "mix" / "train.jsonl"
MIX_SUMMARY = CPT_DIR / "mix" / "summary.json"
PROBE_A_GLOB = "data/SinhalaMMLU/TEST/easy/*.json"
SHARED_TEST_GLOB = "data/SinhalaMMLU-SharedTask-Test/*/*.json"

# Textbook subject -> (subject, category, subject_original) exactly as in the Probe A files, so both
# probes build the same official prompt ("මෙය {subject_original} විෂයයට අදාළ බහුවරණ ප්‍රශ්නයකි.").
SUBJECT_MAP = {
    "Art": ("Arts", "humanities", "චිත්‍ර"),
    "Buddhism": ("Buddhism", "humanities", "බුද්ධ ධර්මය"),
    "Catholicism": ("Catholicism", "humanities", "කතෝලික ධර්මය"),
    "Christianity": ("Christianity", "humanities", "ක්‍රිස්තියානි ධර්මය"),
    "Civics": ("Civics", "social_science", "පුරවැසි අධ්‍යාපනය"),
    "Eastern Music": ("Eastern_music", "humanities", "පෙරදිග සංගීතය"),
    "Geography": ("Geography", "social_science", "භූගෝල විද්‍යාව"),
    "Health": ("Health and Physical Science", "social_science", "සෞඛ්‍ය හා ශාරීරික විද්‍යාව"),
    "History": ("History", "humanities", "ඉතිහාසය"),
    "Islam": ("Islam", "humanities", "ඉස්ලාම් ධර්මය"),
    "Sinhala": ("Sinhala_language_and_literature", "language", "සිංහල භාෂාව සහ සාහිත්‍යය"),
    "Dancing": ("dancing", "humanities", "නර්තනය"),
    "Drama and Theatre": ("drama_and_theatre", "humanities", "නාට්‍ය හා රංග කලාව"),
    "Science": ("science", "stem", "විද්‍යාව"),
}


# Official SinhalaMMLU prompt (scripts/eval_baseline.py), as plain text for the Base model: no chat template.
# The prompt ends with "පිළිතුර: " INCLUDING the space: Qwen tokenizes ": 3" as ":" + " " + "3", so after the
# space the next token is the bare digit ("1".."4" = ids 16..19). Without the space it would be " ".
INTRO = "මෙය {} විෂයයට අදාළ බහුවරණ ප්‍රශ්නයකි.\n"
INSTRUCTION = "පහත ප්‍රශ්නයට 1, 2, 3, 4 යන පිළිතුරුවලින් නිවැරදි හෝ ඉතාමත් ගැළපෙන පිළිතුර තෝරන්න.\n\n"


def build_prompt(item: dict) -> str:
    options = "\n".join(f"{i}. {c.strip()}" for i, c in enumerate(item["choices"], start=1))
    return (f"{INTRO.format(item['metadata']['subject_original'])}{INSTRUCTION}"
            f"ප්‍රශ්නය: {item['question']}\n{options}\nපිළිතුර: ")


def norm(s) -> str:
    """Comparison key: NFC, no zero-width joiners, punctuation, Markdown '#' and whitespace removed.
    '•' goes too: the FM conversion turned some books' full stops and hyphens into it."""
    s = unicodedata.normalize("NFC", str(s))
    s = re.sub(r"[​-‍﻿]", "", s)
    return re.sub(r"[\s.,?!:;\"'“”‘’()\[\]_\-–—#•]+", "", s)


def load_json_items(pattern: str) -> list[dict]:
    items = []
    for path in sorted(glob.glob(str(ROOT / pattern))):
        with open(path, encoding="utf-8") as f:
            items += json.load(f)
    return items


def near_duplicates(items: list[dict], others: list[dict], q_ratio=0.8, min_shared=2) -> list[tuple[int, int, float, int]]:
    """(i, j, question similarity, shared options) for item i that is the same question as others[j]:
    question text >= q_ratio similar AND >= min_shared identical options (after norm)."""
    by_choice = defaultdict(set)
    for j, o in enumerate(others):
        for c in o["choices"]:
            by_choice[norm(c)].add(j)
    hits = []
    for i, it in enumerate(items):
        shared = defaultdict(int)
        for c in {norm(c) for c in it["choices"]}:
            for j in by_choice[c]:
                shared[j] += 1
        for j, n in shared.items():
            if n >= min_shared:
                r = difflib.SequenceMatcher(None, norm(it["question"]), norm(others[j]["question"])).ratio()
                if r >= q_ratio:
                    hits.append((i, j, round(r, 2), n))
    return hits
