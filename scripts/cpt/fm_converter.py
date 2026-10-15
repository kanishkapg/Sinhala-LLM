"""Legacy FM-font Sinhala -> Unicode (Step 3 of docs/prepare_cpt_dataset.md).

Base table: Open-SL fm_abaya_to_unicode (MIT, ported in fm_rules.py). The patches below were found by
auditing every Easy textbook; each one is listed in PATCHES so it can go into manifest.json and the
system report. Add new rules here (never in fm_rules.py) when the human review finds a repeated error.
"""
import json
import re
from functools import lru_cache

from common import CONFIG_DIR
from fm_rules import RULES

PATCHES = [
    r"P1 upstream regex bugs: f\da / f\d (\d = any digit) -> f|da / f|d (ඳෝ ඳො); f\\{da / f\\{d -> f{da / f{d (ඥෝ ඥො)",
    "P2 nasal prefix ` + consonant code -> the code of the pre-nasalised letter (`o -> | = ඳ, `v -> ~ = ඬ, "
    "`. -> Õ = ඟ, `n -> U = ඹ, `ÿ -> ÷ = ඳු, `È -> ¢ = ඳි ...), mapped before the base table so the "
    "f (ෙ) prefix rules still apply; any ` left before ද ඩ ග බ after conversion is fixed in POST",
    "P3 ¥ -> දූ and ¨ -> ලු (upstream: missing / ලූ); ª -> ඳු, ý -> ඡි, \\ -> ඳ (rare glyphs found in the audit)",
    "P4 delete the • -> x rule; ¡ and ² -> •; ⋆ (from ³) -> •",
    "P5 % -> ්\u200dර without a leading ZWJ; strip stray '=' from upstream outputs (ඡ= රැ= ඳැ= ඤු= ඤූ=)",
    "P6 per-font punctuation kept as-is (config/fonts.json: punct_keep)",
    "P7 ligatures ﬁ ﬂ -> fi fl before conversion",
    "P8 delete the early & -> ) rule: its output was turned into * by the later ) -> * rule; the final "
    "& -> ) rule (after it) gives the right )",
    "P9 · before a digit -> × (magnification ×5, ×40 in science)",
    "P10 vowel sign written before the rakaransaya / yansaya -> after it (ගාබි්\u200Dරයෙල් -> ගාබ්\u200Dරියෙල්)",
    "P11 Easy s02 leftovers, checked on the printed page: `ã -> ඬී, `t and `: -> ළු (මාළු ඇතුළු), ` dropped "
    "before ¯ ® | (මජ්ඣිම සඳහා), consonant + ` + vowel sign (o`d -> ඳා), `` -> `, `c -> ඦ, μ after a letter -> ද්‍ය (වෛද්‍ය), "
    "Ω before ¿ and the Ohm sign U+2126 dropped (එළවළු, මනඃ)",
    "P12 roman numerals: » -> I (the table used » as its own placeholder for the comma), ¼ -> V",
    "P13 · not before a digit -> • (bullets); ASCII control characters removed",
    "P15 last Easy leftovers: `X -> ඬ (බුඬ ගජ්ඣ), `Co -> ඤ (alphabet table), a line holding only ` -> dropped",
    "P14 per line, across spans: ` + ඵ -> ළු; Ω doubled by an FM span and a Times span -> one Ω; same for a » bullet",
]

# P2 + P3 + P4: applied before the base table, longest first. Outputs are FM codes, not Unicode, so the
# table's reordering rules (e.g. the f = ෙ prefix) still see ASCII.
PRE = [("``", "`"), ("`ÿ", "÷"), ("`¥", "|Q"), ("`÷", "÷"), ("`È", "¢"), ("`§", "£"), ("`Þ", "|d"), ("`±", "|e"),
       ("`â", "~a"), ("`ï", "Ua"), ("`o", "|"), ("`v", "~"), ("`.", "Õ"), ("`n", "U"), ("`M", "¿"),
       ("`ã", "~S"), ("`t", "¿"), ("`:", "¿"), ("`¯", "¯"), ("`®", "®"), ("`|", "|"),        # P11
       ("`X", "~"), ("`Co", "["),                                                           # P15
       ("Ω¿", "¿"), (chr(0x2126), ""),
       ("¥", "oQ"), ("¨", ",q"), ("ª", "÷"), ("ý", "Ps"), ("\\", "|"), ("¡", "•"), ("²", "•")]
_NASAL = {"o": "|", "v": "~", ".": "Õ", "n": "U"}
PRE_RE = [(re.compile(r"`([sSdeEqQ])([ov.n])"), lambda m: m[1] + _NASAL[m[2]]),   # ` before the vowel sign: y`so -> ys| (හිඳ)
          (re.compile(r"([ov.n])`(?=[sSdeEqQ])"), lambda m: _NASAL[m[1]]),       # ` after the consonant, before a vowel sign: o`d -> |d (ඳා)
          (re.compile(r"(?<=[^\s\d(\-])μ"), "oH")]                           # ffjμjrhd -> වෛද්‍යවරයා, úμdudk (μA stays)
# Fixes on the *Unicode output*, applied last. Add the "wrong -> right" pairs the review finds here.
POST = [("`ද", "ඳ"), ("`ඩ", "ඬ"), ("`ග", "ඟ"), ("`බ", "ඹ"), ("`ජ", "ඦ"), ("⋆", "•"), ("¼", "V")]
POST_RE = [(re.compile(r"·(?=\d)"), "×"),
           (re.compile(r"·"), "•"),                                                   # P13
           (re.compile(r"[\x00-\x08\x0b-\x1f\x7f]"), ""),
           (re.compile("([\u0DCF-\u0DDF])(\u0DCA\u200D[\u0DBA\u0DBB])"), r"\2\1")]   # vowel sign before rakaransaya/yansaya -> after it
# P14: on the joined line (spans of different fonts), see fix_line()
LINE_POST = [("`ඵ", "ළු"), (chr(0x2126), "Ω"), ("I»", "»")]   # I» = FM » under a Unicode » bullet
LINE_POST_RE = [(re.compile(r"Ω\s*(?=Ω)"), ""),
                (re.compile(r"^\s*`\s*$"), "")]                     # P15: a line that is only ` (icon)

_PUA = 0xE000  # private-use code points protect punct_keep characters from the table
_ROMAN_I = chr(0xE0FF)  # P12: FM » (roman I) must not meet the table's own » (comma placeholder)


def _compile():
    out = []
    for pat, rep in RULES:
        if pat == "•" or pat == r"\&":      # P4, P8
            continue
        pat = {r"f\da": r"f\|da", r"f\d": r"f\|d",                         # P1
               r"f\\{da": r"f\{da", r"f\\{d": r"f\{d"}.get(pat, pat)
        rep = rep.replace("\u200D්\u200Dර", "්\u200Dර")                     # P5
        rep = re.sub(r"([\u0D80-\u0DFF])=$", r"\1", rep)
        out.append((re.compile(pat), rep))
    return out


_TABLE = _compile()


def _load_fonts():
    cfg = json.loads((CONFIG_DIR / "fonts.json").read_text(encoding="utf-8"))
    return (set(cfg["convert"]), set(cfg["keep"]), tuple(cfg.get("keep_prefixes", [])), set(cfg["drop"]),
            tuple(cfg.get("drop_prefixes", [])), {k: str(v) for k, v in cfg.get("punct_keep", {}).items()})


CONVERT, KEEP, KEEP_PREFIXES, DROP, DROP_PREFIXES, PUNCT_KEEP = _load_fonts()


def font_name(raw: str) -> str:
    """Drop the subset prefix, spaces and a ',Bold' style suffix: 'ABCDEF+FMAbhaya x' -> 'FMAbhayax'.
    '#20' (an escaped space) is kept on purpose: those fonts have no usable text layer (config: drop)."""
    name = re.sub(r"\s+", "", raw.split("+")[-1])
    return re.sub(r",(Bold|Italic|BoldItalic)$", "", name)


def font_action(font: str) -> str:
    """'convert' | 'keep' | 'drop' | 'unknown' for a font name without subset prefix."""
    if font in CONVERT:
        return "convert"
    if font in KEEP or (KEEP_PREFIXES and font.startswith(KEEP_PREFIXES)):
        return "keep"
    if font in DROP or font.startswith(DROP_PREFIXES):
        return "drop"
    return "unknown"


@lru_cache(maxsize=500_000)
def fm2uni(text: str, font: str = "") -> str:
    keep = PUNCT_KEEP.get(font, "")
    for i, c in enumerate(keep):                                             # P6
        text = text.replace(c, chr(_PUA + i))
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")                       # P7
    text = text.replace("»", _ROMAN_I)                                       # P12
    for rx, b in PRE_RE:
        text = rx.sub(b, text)
    for a, b in PRE:
        text = text.replace(a, b)
    for rx, rep in _TABLE:
        text = rx.sub(rep, text)
    for i, c in enumerate(keep):
        text = text.replace(chr(_PUA + i), c)
    text = text.replace(_ROMAN_I, "I")
    for a, b in POST:
        text = text.replace(a, b)
    for rx, b in POST_RE:
        text = rx.sub(b, text)
    return text


def fix_line(text: str) -> str:
    """P14: fixes that need the whole line (a glyph split over two fonts)."""
    for a, b in LINE_POST:
        text = text.replace(a, b)
    for rx, b in LINE_POST_RE:
        text = rx.sub(b, text)
    return text


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for s in sys.argv[1:] or ["b;sydih", "ish¨ u Ôùkag", "i`oyd", "m%Yakh", "fmdf;a"]:
        print(f"{s!r} -> {fm2uni(s)}")
