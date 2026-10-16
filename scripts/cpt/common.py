"""Shared settings and helpers for the CPT dataset pipeline (see docs/prepare_cpt_dataset.md).

Every stage script imports this module. Paths are relative to the repository root, so run the
scripts from the repo root:  python scripts/cpt/s01_inventory.py --level easy
"""
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
TEST_GLOBS = ["data/SinhalaMMLU-SharedTask-Test/**/*.json", "data/SinhalaMMLU/**/*.json"]
CONFIG_DIR = Path(__file__).resolve().parent / "config"

PIPELINE_VERSION = "cpt-v1"
SEED = 42
TOKENIZER_ID = "Qwen/Qwen3.5-4B-Base"
EOS_TOKEN = "<|endoftext|>"

LEVEL_GRADES = {"easy": [6, 7, 8], "medium": [9, 10, 11], "hard": [12, 13]}

# File-name subject in data/raw/gN -> (subject, Sinhala subject name of the official eval prompt).
# The Sinhala names equal `metadata.subject_original` in the test files, so the header prepended to
# textbook text (Step 10) uses the words the model will see in the evaluation prompt.
SUBJECTS = {
    "buddhism": ("Buddhism", "බුද්ධ ධර්මය"),
    "catholic": ("Catholicism", "කතෝලික ධර්මය"),
    "christianism": ("Christianity", "ක්\u200Dරිස්තියානි ධර්මය"),
    "civic": ("Civics", "පුරවැසි අධ්\u200Dයාපනය"),
    "geo": ("Geography", "භූගෝල විද්\u200Dයාව"),
    "health": ("Health", "සෞඛ්\u200Dය"),
    "history": ("History", "ඉතිහාසය"),
    "islam": ("Islam", "ඉස්ලාම් ධර්මය"),
    "science": ("Science", "විද්\u200Dයාව"),
    "sinhala": ("Sinhala", "සිංහල"),
    "art": ("Art", "චිත්\u200Dර කලාව"),
    "dancing": ("Dancing", "නර්තනය"),
    "drama": ("Drama and Theatre", "නාට්\u200Dය හා රංග කලාව"),
    "music": ("Eastern Music", "පෙරදිග සංගීතය"),
}

TEXTBOOK_LICENSE = "Educational Publications Department, Sri Lanka (free-distribution textbook)"


def level_dir(level: str) -> Path:
    return ROOT / "data" / "processed" / "cpt" / level


def book_info(pdf: Path) -> dict:
    """'data/raw/g7/science g-7 p-II S.pdf' -> book_id 'science_g7_p2_si' + subject/grade fields."""
    m = re.match(r"(?P<subj>[a-z_ ]+?)\s+g-(?P<g>\d+)(?:\s+p-(?P<part>[IV]+))?\s+(?P<med>[A-Z])$", pdf.stem)
    if not m:
        raise SystemExit(f"Unexpected file name {pdf.name!r}: use '<subject> g-<grade> [p-I] S.pdf'")
    subj = m["subj"].strip().replace(" ", "_")
    if subj not in SUBJECTS:
        raise SystemExit(f"{pdf.name}: subject {subj!r} is not in SUBJECTS (scripts/cpt/common.py)")
    part = {"I": "p1", "II": "p2", "III": "p3"}.get(m["part"] or "", "")
    medium = {"S": "si", "E": "en", "T": "ta"}[m["med"]]
    grade = int(m["g"])
    return {
        "book_id": f"{subj}_g{grade}_{part + '_' if part else ''}{medium}",
        "file": str(pdf.relative_to(ROOT)).replace("\\", "/"),
        "subject": SUBJECTS[subj][0],
        "subject_si": SUBJECTS[subj][1],
        "grade": grade,
        "part": part,
        "medium": medium,
    }


def list_pdfs(level: str) -> list[Path]:
    pdfs = []
    for g in LEVEL_GRADES[level]:
        pdfs += sorted((RAW / f"g{g}").glob("*.pdf"))
    return pdfs


# ---------------------------------------------------------------- JSONL / JSON helpers

def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise SystemExit(f"{path}:{n}: invalid JSON ({e})")


def write_jsonl(path: Path, rows) -> int:
    """UTF-8, one object per line, no blank lines, LF endings, Sinhala kept readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- text helpers

_CTRL = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F\u00AD\u200B\u200C\u2028\u2029\u2060\uFEFF]")


def normalize(text: str) -> str:
    """NFC, canonical ZWJ conjuncts, no control chars / NBSP / zero-width spaces (ZWJ is kept)."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00A0", " ").replace("\t", " ")
    text = _CTRL.sub("", text)
    text = text.replace("\u200D්", "්\u200D")        # ZWJ before hal -> hal + ZWJ
    text = re.sub("\u200D+", "\u200D", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Textbook-only repairs (s04). Symbol-font private-use glyphs and odd quote/space characters seen in the Easy books.
_CHAR_MAP = [("", "•"), ("", "•"), ("", "√"), ("", " "), ("²•", "•"),
             ("ˮ", "”"), ("ʻ", "‘")]
_SPACES = re.compile(r"[ -  ]")
_MIDDOT_BULLET = re.compile(r"(?<!\S)·(?=\s)")
_C = "[ක-ෆ]"                                  # Sinhala consonant
_SIGN = "[ංඃ්-ෟෲෳ]"       # hal, vowel signs, anusvara / visarga
_WORD_START = r"(?<![඀-෿‍])"
_KOMBUVA = [                                             # kombuva (ෙ) left in visual order at a word start
    (re.compile(_WORD_START + f"ෙ({_C})ෙ"), "\\1ෛ"),    # ෙසෙල   -> සෛල
    (re.compile(_WORD_START + f"ෙ({_C})ා"), "\\1ො"),    # ෙදාර   -> දොර
    (re.compile(_WORD_START + f"ෙ({_C})්"), "\\1ේ"),    # ෙඡ්ද   -> ඡේද
    (re.compile(_WORD_START + f"ෙ({_C})ෟ"), "\\1ෞ"),
    (re.compile(_WORD_START + f"ෙ({_C})(?!{_SIGN})"), "\\1ෙ"),   # ෙකටියෙන් -> කෙටියෙන්
]
_VISARGA_COLON = re.compile(r" ?(?<![඀-෿‍])ඃ")   # උදා ඃ-  /  පූර්ණක ඃ  -> උදා:-  /  පූර්ණක:
_SPLIT_SIGN = re.compile(f"({_C}) +(?={_SIGN})")        # මෙන ්ම -> මෙන්ම (a sign never starts a word)


def repair_sinhala(text: str) -> str:
    """Fix glyph-level damage in extracted textbook text: Symbol-font bullets, stray quotes and spaces,
    kombuva left before its consonant, and a space between a consonant and its sign. Idempotent."""
    for a, b in _CHAR_MAP:
        text = text.replace(a, b)
    text = _SPACES.sub(" ", text)
    text = _MIDDOT_BULLET.sub("•", text)
    for rx, b in _KOMBUVA:
        text = rx.sub(b, text)
    text = _VISARGA_COLON.sub(":", text)
    return _SPLIT_SIGN.sub(r"\1", text)


def sinhala_ratio(text: str) -> float:
    """Share of Sinhala characters among letters (digits, punctuation and spaces are ignored)."""
    sin = lat = 0
    for c in text:
        if "\u0D80" <= c <= "\u0DFF":
            sin += 1
        elif c.isalpha():
            lat += 1
    return sin / (sin + lat) if sin + lat else 0.0


# ---------------------------------------------------------------- tokenizer

_TOK = None


def tokenizer():
    """Qwen3.5 tokenizer via the light `tokenizers` package (downloads tokenizer.json once)."""
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_pretrained(TOKENIZER_ID)
        if _TOK.token_to_id(EOS_TOKEN) is None:
            raise SystemExit(f"{EOS_TOKEN} is not in the {TOKENIZER_ID} vocabulary")
    return _TOK


def count_tokens(texts: list[str]) -> list[int]:
    if not texts:
        return []
    return [len(e.ids) for e in tokenizer().encode_batch(texts, add_special_tokens=False)]


def tokenizer_hash() -> str:
    from huggingface_hub import hf_hub_download
    return sha1(Path(hf_hub_download(TOKENIZER_ID, "tokenizer.json")).read_text(encoding="utf-8"))


def stdout_utf8() -> None:
    """Windows consoles default to cp1252, which cannot print Sinhala."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
