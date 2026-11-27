"""Rescore every saved prompting result with the loop stop of fast_run/ (no GPU, no model).

Why: the ICL runs use `--fast`, which stops generation when the output starts looping
(and at 1,000 bytes). The earlier zero-shot, few-shot and segmentation runs generated to
the end of their budget, and the looping tail inflates MER and chrF++ (Jejueo 3-shot:
MER 14.0 without the stop, 1.4 with it). To put all prompting methods in one table, this
script cuts every saved raw_output where the fast runner would have stopped, parses the
cut output again and scores it. Greedy decoding is deterministic, so a fast run produces
exactly this prefix (fast_run/README.md: 68 of 70 rows identical on a T4, the other 2
were looping rows cut early).

The stop rule is the one in fast_run/: `replay_loop_stop.cut` for the two-line strategies
(all-in-one, step-by-step), and the same rule with the three-line completion check for the
segmentation runs (replay_loop_stop.cut checks the two-line completion only, which would
end a three-line answer after its Segmentation line).

Usage (from the repository root; needs pandas and sacrebleu):
    python evaluation/rescore_loop_stop.py                 # all languages and methods
    python evaluation/rescore_loop_stop.py --lang jejueo

Reads (Korean v3 files and dev smoke tests are skipped):
    results/zero_shot/<lang>/bolmo-1b_<strategy>_test.csv
    results/few_shot/<lang>/bolmo-1b_<strategy>_<K>shot_test.csv
    results/segmentation/<lang>/bolmo-1b_three_step_seg_<K>shot_test.csv
    results/icl/<lang>/bolmo-1b_<strategy>_icl<K>_test.csv

Writes results/evaluation/prompting_scores.csv: one row per file and rule
(`original` = as generated, `loop_stop` = cut by the fast_run rule), all test rows. The
`original` rows reproduce each folder's scores.csv (checked by this script). Extra columns:
    stopped_by_loop      share of rows the loop stop cuts
    stopped_by_byte_cap  share of rows the 1,000-byte cap cuts
    answer_changed       share of rows whose parsed (transliteration, gloss) changes
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("zero_shot", "fast_run", "segmentation"):
    sys.path.insert(0, str(ROOT / sub))
import score_zero_shot as sz  # noqa: E402
import replay_loop_stop as replay  # noqa: E402
import run_segmentation as seg  # noqa: E402  (also registers the "three_step" completion check)

pd = sz.pd
rz, fast = seg.rz, replay.f
LANGS = ("ainu", "tsez", "jejueo")

# folder, file pattern -> method label
PATTERNS = [
    ("zero_shot", re.compile(r"bolmo-1b_(?P<strategy>all_in_one|step_by_step)_test\.csv$"), lambda m: "zero-shot"),
    ("few_shot", re.compile(r"bolmo-1b_(?P<strategy>all_in_one|step_by_step)_(?P<k>\d+)shot_test\.csv$"),
     lambda m: f"{m['k']}-shot"),
    ("segmentation", re.compile(r"bolmo-1b_(?P<strategy>three_step_seg)_(?P<k>\d+)shot_test\.csv$"),
     lambda m: f"segmentation {m['k']}-shot"),
    ("icl", re.compile(r"bolmo-1b_(?P<strategy>all_in_one|step_by_step)_icl(?P<k>\d+)_test\.csv$"),
     lambda m: f"ICL-{m['k']}"),
]
COLS = ["language", "method", "strategy", "rule", "n", "gloss_morpheme_accuracy", "gloss_word_accuracy",
        "gloss_mer", "gloss_wer", "gloss_chrfpp", "transliteration_chrfpp", "transliteration_word_accuracy",
        "parse_failure_rate", "stopped_by_loop", "stopped_by_byte_cap", "answer_changed"]


def cut(raw: str, kind: str) -> tuple[str, str]:
    """Shortest prefix at which the fast runner would stop, and why (fast_run's rule)."""
    if kind == "all_in_one":
        return replay.cut(raw)
    b = raw.encode("utf-8")  # same loop as replay.cut, with the given completion check
    for i in range(1, len(b) + 1):
        t = b[:i].decode("utf-8", "ignore")
        if rz.is_complete(kind, t):
            return t, "complete"
        if fast._over_cap(t):
            return t, "byte_cap"
        if fast._loop(t):
            return t, "loop"
    return raw, "end_of_saved_output"


def parse(strategy: str, completion: str) -> tuple[str, str, str]:
    """(pred_transliteration, pred_gloss, parse_status), as the runner stores them."""
    if strategy == "all_in_one":
        return rz.parse_all_in_one(completion)
    if strategy == "step_by_step":
        return rz.parse_step_by_step(completion)
    _, segmentation, gloss, status = seg.parse_three_step(completion)  # Segmentation line is scored
    return segmentation, gloss, status


def files(langs):
    for folder, pat, label in PATTERNS:
        for lang in langs:
            for f in sorted((ROOT / "results" / folder / lang).glob("*_test.csv")):
                m = pat.search(f.name)
                if m:
                    yield lang, label(m), m["strategy"], f


def rescore(lang: str, strategy: str, f: Path) -> list[dict]:
    df = pd.read_csv(f, dtype=str, keep_default_na=False)
    kind = "three_step" if strategy == "three_step_seg" else "all_in_one"
    cuts = [cut(raw, kind) for raw in df.raw_output]
    new = df.copy()
    parsed = [parse(strategy, t) for t, _ in cuts]
    new["pred_transliteration"] = [p[0] for p in parsed]
    new["pred_gloss"] = [p[1] for p in parsed]
    new["parse_status"] = [p[2] for p in parsed]
    why = pd.Series([w for _, w in cuts])
    changed = ((new.pred_transliteration != df.pred_transliteration) | (new.pred_gloss != df.pred_gloss)).mean()

    out = []
    join = sz.manifest_for(lang).get("gloss_affixes_detached", False)
    for rule, d in (("original", df), ("loop_stop", new)):
        d = d.copy()
        if join:  # Jejueo: re-attach detached suffixes, as score_zero_shot.py does
            d["gold_gloss"] = d.gold_gloss.map(sz.join_detached_affixes)
            d["pred_gloss"] = d.pred_gloss.map(sz.join_detached_affixes)
        res = sz.score(d)
        res.pop("truncation_rate")  # the runners' flags mean different things; see stopped_by_* instead
        out.append({"rule": rule, "n": len(d), **res,
                    "stopped_by_loop": (why == "loop").mean() if rule == "loop_stop" else 0.0,
                    "stopped_by_byte_cap": (why == "byte_cap").mean() if rule == "loop_stop" else 0.0,
                    "answer_changed": changed if rule == "loop_stop" else 0.0})
    return out


def check_original(row: dict, f: Path) -> None:
    """The `original` scores must equal the folder's scores.csv (all rows)."""
    ref = f.parent / "scores.csv"
    if not ref.exists():
        return
    s = pd.read_csv(ref)
    s = s[(s["file"] == f.name) & (s["subset"] == "all")]
    if s.empty:
        return
    for c in ("gloss_morpheme_accuracy", "gloss_word_accuracy", "gloss_mer", "gloss_chrfpp", "transliteration_chrfpp"):
        assert abs(s.iloc[0][c] - row[c]) < 1e-3, (f.name, c, s.iloc[0][c], row[c])


def main(argv=None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=LANGS, default=None, help="default: all three")
    ap.add_argument("--out", type=Path, default=ROOT / "results/evaluation/prompting_scores.csv")
    args = ap.parse_args(argv)

    rows = []
    for lang, method, strategy, f in files([args.lang] if args.lang else LANGS):
        print(f"{lang:7s} {method:22s} {strategy:15s} {f.relative_to(ROOT)}", flush=True)
        for r in rescore(lang, strategy, f):
            r.update(language=lang, method=method, strategy=strategy, file=str(f.relative_to(ROOT)))
            if r["rule"] == "original":
                check_original(r, f)
            rows.append(r)
    out = pd.DataFrame(rows)
    out = out[COLS + [c for c in out.columns if c not in COLS]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    show = ["language", "method", "strategy", "rule", "gloss_morpheme_accuracy", "gloss_word_accuracy",
            "gloss_mer", "gloss_chrfpp", "transliteration_chrfpp", "parse_failure_rate", "stopped_by_loop"]
    print(out[show].round(4).to_string(index=False))
    print("original scores match each folder's scores.csv")
    print("Wrote", args.out.relative_to(ROOT))
    return out


if __name__ == "__main__":
    main()
