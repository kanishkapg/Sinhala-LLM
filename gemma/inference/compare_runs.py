"""Compare two prediction runs of modal_predict.py question by question.

    python gemma/inference/compare_runs.py outputs/predictions_detail.jsonl outputs/run2/predictions_detail.jsonl

Reports how often the two runs chose the same option, overall and per level, how the disagreements line up with
the proxy answer key, and whether they sit on the questions the model was least sure about. Paths are taken as
given, or resolved against gemma/inference/ when relative.

Greedy decoding (temperature 0, fixed seed) makes a repeat run close to identical, but not bit-identical: vLLM
batches whatever requests are in flight, so a question can be summed in a different order and flip on a
near-tie. Disagreements therefore mark questions the model is genuinely torn between, which is useful in
itself - they are the ones a different inference setup would also flip.

Exits 1 if the two runs cover different questions (so a half-finished run is never read as perfect agreement).
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST = HERE.parents[1] / "data/SinhalaMMLU-SharedTask-Test"
LEVELS = ("easy", "medium", "hard")


def load(path: str) -> dict:
    p = Path(path)
    p = p if p.is_absolute() or p.exists() else HERE / path
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise SystemExit(f"{p} is empty")
    if "pred" not in rows[0]:  # a predictions.jsonl, not a predictions_detail.jsonl
        rows = [r | {"pred": int(r["answer"])} for r in rows]
    return {(r["level"], r["subject"], r["q_no"]): r for r in rows}


def proxy_key() -> dict:
    gold = {}
    for level in LEVELS:
        path = TEST / "with_answers" / f"{level}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line:
                    q = json.loads(line)
                    gold[(level, q["subject"], q["q_no"])] = q["answer"]
    return gold


def accuracy(rows: dict, keys: list, gold: dict) -> str:
    graded = [rows[k]["pred"] == gold[k] for k in keys if k in gold]
    return f"{100 * sum(graded) / len(graded):.1f}%" if graded else "-"


def main(path_a: str, path_b: str, show: int = 15) -> None:
    a, b = load(path_a), load(path_b)
    shared = a.keys() & b.keys()
    # A --limit run against a full one is a cheap partial check, so compare the overlap; but two runs that each
    # hold questions the other lacks are not two runs of the same thing, and silently intersecting them would
    # hide a half-finished run.
    if not (a.keys() <= b.keys() or b.keys() <= a.keys()):
        raise SystemExit(f"the runs cover different questions: {len(a)} vs {len(b)} rows, "
                         f"{len(a.keys() - b.keys())} only in the first, {len(b.keys() - a.keys())} only in the "
                         "second, so neither is a subset of the other")
    if not shared:
        raise SystemExit("the two runs have no question in common")
    if len(shared) < max(len(a), len(b)):
        print(f"[note] comparing the {len(shared)} questions both runs hold ({len(a)} vs {len(b)} rows)\n")
    gold = proxy_key()
    keys = sorted(shared, key=lambda k: (LEVELS.index(k[0]), k[1], k[2]))
    flips = [k for k in keys if a[k]["pred"] != b[k]["pred"]]

    print(f"run A: {path_a}\nrun B: {path_b}\n{len(keys)} questions compared\n")
    print(f"{'level':<8}{'n':>6}{'same':>8}{'agree':>9}{'A vs key':>10}{'B vs key':>10}")
    for level in LEVELS:
        lv = [k for k in keys if k[0] == level]
        if lv:
            same = sum(a[k]["pred"] == b[k]["pred"] for k in lv)
            print(f"{level:<8}{len(lv):>6}{same:>8}{100 * same / len(lv):>8.1f}%"
                  f"{accuracy(a, lv, gold):>10}{accuracy(b, lv, gold):>10}")
    print(f"{'all':<8}{len(keys):>6}{len(keys) - len(flips):>8}{100 * (1 - len(flips) / len(keys)):>8.1f}%"
          f"{accuracy(a, keys, gold):>10}{accuracy(b, keys, gold):>10}")

    if not flips:
        print("\nthe two runs chose the same option for every question")
        return

    print(f"\n{len(flips)} disagreements")
    if gold:
        both = [k for k in flips if k in gold]
        print(f"  against the proxy key (itself only ~85-90% correct): A right {sum(a[k]['pred'] == gold[k] for k in both)}"
              f", B right {sum(b[k]['pred'] == gold[k] for k in both)}, both wrong "
              f"{sum(a[k]['pred'] != gold[k] and b[k]['pred'] != gold[k] for k in both)} of {len(both)}")
    for name, rows in (("A", a), ("B", b)):
        if "source" in rows[flips[0]]:
            odd = Counter(s for k in flips for s in [rows[k]["source"]] if s != "parsed")
            capped = sum(rows[k].get("capped", False) for k in flips)
            print(f"  run {name} on these: {len(flips) - sum(odd.values())} parsed normally, "
                  + (", ".join(f"{n} {s}" for s, n in odd.items()) or "0 forced") + f", {capped} hit the budget")
    print(f"\nfirst {min(show, len(flips))} disagreements (A -> B, key):")
    for k in flips[:show]:
        print(f"  {k[0]:<7}{k[1][:34]:<36}q{k[2]:<5}{a[k]['pred']} -> {b[k]['pred']}"
              + (f"   key {gold[k]}" if k in gold else ""))
    if len(flips) > show:
        print(f"  ... and {len(flips) - show} more")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], *(int(x) for x in sys.argv[3:4]))
