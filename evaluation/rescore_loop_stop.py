"""Score every saved test result (prompting and LoRA) in one table, under one stop rule.

No GPU, no model: it reads the saved outputs.

Why: the ICL runs use `--fast`, which stops generation when the output starts looping
(and at 1,000 bytes). The earlier zero-shot, few-shot and segmentation runs generated to
the end of their budget, and the LoRA runs stop at <eos> or a complete answer but still
loop in some rows (Tsez: 4-7%). A looping tail inflates MER and chrF++ (Jejueo 3-shot:
MER 14.0 without the stop, 1.4 with it). To put all methods in one table, this script
cuts every saved raw_output where the fast runner would have stopped, parses the cut
output again (with the parsers every runner uses) and scores it. Greedy decoding is deterministic, so a fast run produces
exactly this prefix (fast_run/README.md: 68 of 70 rows identical on a T4, the other 2
were looping rows cut early).

The stop rule is the one in fast_run/: `replay_loop_stop.cut` for the two-line strategies
(all-in-one, step-by-step), and the same rule with the three-line completion check for the
segmentation runs (replay_loop_stop.cut checks the two-line completion only, which would
end a three-line answer after its Segmentation line).

Usage (from the repository root; needs pandas and sacrebleu):
    python evaluation/rescore_loop_stop.py                 # all languages and methods
    python evaluation/rescore_loop_stop.py --lang jejueo

Reads (Korean files and dev predictions are skipped; checkpoint selection on dev is
evaluation/evaluate.py's job):
    results/zero_shot/<lang>/bolmo-1b_<strategy>_test.csv
    results/few_shot/<lang>/bolmo-1b_<strategy>_<K>shot_test.csv
    results/segmentation/<lang>/bolmo-1b_three_step_seg_<K>shot_test.csv
    results/icl/<lang>/bolmo-1b_<strategy>_icl<K>_test.csv
    results/lora/<lang>/<run>/predictions/<run>_<rule>-selected_step<N>[_<K>shot]_test.csv

Writes results/evaluation/unified_test_scores.csv: one row per file and rule
(`original` = as generated, `loop_stop` = cut by the fast_run rule), all test rows. The
script checks that the `original` rows reproduce each prompting folder's scores.csv, and
that re-parsing every saved raw_output gives the stored answer. Extra columns:
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
for sub in ("zero_shot", "fast_run", "segmentation", "icl"):
    sys.path.insert(0, str(ROOT / sub))
import score_zero_shot as sz  # noqa: E402
import replay_loop_stop as replay  # noqa: E402
import run_segmentation as seg  # noqa: E402  (also registers the "three_step" completion check)
import word_format as wf  # noqa: E402  (registers the "word" completion check of --format word)

pd = sz.pd
rz, fast = seg.rz, replay.f
LANGS = ("ainu", "tsez", "jejueo")

# glob under results/<lang>/..., file pattern -> method label
PATTERNS = [
    ("zero_shot/{lang}/*_test.csv", re.compile(r"bolmo-1b_(?P<strategy>all_in_one|step_by_step)_test\.csv$"),
     lambda m: "zero-shot"),
    ("few_shot/{lang}/*_test.csv", re.compile(r"bolmo-1b_(?P<strategy>all_in_one|step_by_step)_(?P<k>\d+)shot_test\.csv$"),
     lambda m: f"{m['k']}-shot"),
    ("segmentation/{lang}/*_test.csv", re.compile(r"bolmo-1b_(?P<strategy>three_step_seg)_(?P<k>\d+)shot_test\.csv$"),
     lambda m: f"segmentation {m['k']}-shot"),
    ("icl/{lang}/*_test.csv",
     re.compile(r"bolmo-1b_(?P<strategy>all_in_one|step_by_step)_icl(?P<k>\d+)(?P<word>_word)?_test\.csv$"),
     lambda m: f"ICL-{m['k']}" + (", word-by-word" if m["word"] else "")),
    ("lora/{lang}/*/predictions/*_test.csv",
     re.compile(r"bolmo-1b_[a-z]+_(?P<strategy>all_in_one|step_by_step)_r(?P<r>\d+)_lr(?P<lr>[\de.-]+)_s\d+"
                r"_(?P<sel>loss|chrf)-selected_step(?P<step>\d+)(?:_(?P<k>\d+)shot)?_test\.csv$"),
     lambda m: f"LoRA r{m['r']} lr{m['lr']}" + (f" + {m['k']}-shot prompt" if m["k"] else "")),
]
COLS = ["language", "method", "strategy", "selection", "checkpoint", "rule", "n", "gloss_morpheme_accuracy", "gloss_word_accuracy",
        "gloss_mer", "gloss_wer", "gloss_morpheme_bag_f1", "gloss_chrfpp", "transliteration_chrfpp",
        "transliteration_word_accuracy",
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


def answer_format(path: Path) -> str:
    """'word' for ICL runs with --format word (file name ..._icl<K>_word_test.csv), else 'line'."""
    return "word" if re.search(r"_icl\d+_word_", path.name) else "line"


def parse(strategy: str, completion: str, fmt: str = "line") -> tuple[str, str, str]:
    """(pred_transliteration, pred_gloss, parse_status), as the runner stores them."""
    if fmt == "word":
        return wf.parse_word(completion)
    if strategy == "all_in_one":
        return rz.parse_all_in_one(completion)
    if strategy == "step_by_step":
        return rz.parse_step_by_step(completion)
    _, segmentation, gloss, status = seg.parse_three_step(completion)  # Segmentation line is scored
    return segmentation, gloss, status


def files(langs):
    for glob, pat, label in PATTERNS:
        for lang in langs:
            for f in sorted((ROOT / "results").glob(glob.format(lang=lang))):
                m = pat.search(f.name)
                if m:
                    info = m.groupdict()
                    yield lang, label(m), m["strategy"], info.get("sel") or "", info.get("step") or "", f


def rescore(lang: str, strategy: str, f: Path) -> list[dict]:
    df = pd.read_csv(f, dtype=str, keep_default_na=False)
    fmt = answer_format(f)
    kind = "three_step" if strategy == "three_step_seg" else (wf.KIND if fmt == "word" else "all_in_one")
    cuts = [cut(raw, kind) for raw in df.raw_output]
    new = df.copy()
    parsed = [parse(strategy, t, fmt) for t, _ in cuts]
    new["pred_transliteration"] = [p[0] for p in parsed]
    new["pred_gloss"] = [p[1] for p in parsed]
    new["parse_status"] = [p[2] for p in parsed]
    why = pd.Series([w for _, w in cuts])
    # the runners parse with the same functions, so re-parsing the saved output must give the stored answer
    reparsed = [parse(strategy, raw, fmt) for raw in df.raw_output]
    bad = sum((p[0], p[1]) != (a, b) for p, a, b in zip(reparsed, df.pred_transliteration, df.pred_gloss))
    assert bad == 0, f"{f.name}: {bad} rows parse differently from the stored answer"
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
    ap.add_argument("--out", type=Path, default=ROOT / "results/evaluation/unified_test_scores.csv")
    args = ap.parse_args(argv)

    rows = []
    for lang, method, strategy, sel, step, f in files([args.lang] if args.lang else LANGS):
        print(f"{lang:7s} {method:22s} {strategy:15s} {f.relative_to(ROOT)}", flush=True)
        for r in rescore(lang, strategy, f):
            r.update(language=lang, method=method, strategy=strategy, selection=sel, checkpoint=step,
                     file=str(f.relative_to(ROOT)))
            if r["rule"] == "original" and not method.startswith("LoRA"):
                check_original(r, f)
            rows.append(r)
    out = pd.DataFrame(rows)
    out = out[COLS + [c for c in out.columns if c not in COLS]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    show = ["language", "method", "strategy", "selection", "rule", "gloss_morpheme_accuracy", "gloss_word_accuracy",
            "gloss_mer", "gloss_chrfpp", "transliteration_chrfpp", "parse_failure_rate", "stopped_by_loop"]
    print(out[show].round(4).to_string(index=False))
    print("checked: original scores match each prompting folder's scores.csv; re-parsing reproduces every stored answer")
    print("Wrote", args.out.relative_to(ROOT))
    return out


if __name__ == "__main__":
    main()
