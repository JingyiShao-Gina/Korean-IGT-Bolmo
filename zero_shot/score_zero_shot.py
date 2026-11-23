"""Score zero-shot prediction CSVs with the team metrics (evaluation/metrics.py, by Soyeong).

Usage (from the repository root; needs pandas and sacrebleu):
    python zero_shot/score_zero_shot.py --lang ainu
    python zero_shot/score_zero_shot.py --lang jejueo --shots 3   # few-shot results

Scores results/zero_shot/<lang>/bolmo-1b_<strategy>_test.csv (with --shots K:
results/few_shot/<lang>/bolmo-1b_<strategy>_<K>shot_test.csv) and writes scores.csv
in the same folder. Each strategy is scored on all test rows and on
the subset whose gold transcription and gloss have the same number of words (the
Ainu data is not cleaned, and only 180/750 Ainu test rows are aligned).

If the language's manifest sets "gloss_affixes_detached" (Jejueo: "eat -PST -POL -DECL"),
the gold and predicted glosses are first passed through metrics.join_detached_affixes, so
that word-level metrics compare whole words; morpheme-level metrics are unaffected.

--check-korean also re-scores the Korean all-in-one file and asserts that it matches
results/evaluation/test_scores.csv on the soyeong/evaluation branch (0.0594 morpheme
accuracy), to show that this script computes the same numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
import pandas as pd  # noqa: E402
from metrics import evaluate, join_detached_affixes, words  # noqa: E402


def manifest_for(lang: str) -> dict:
    cfg = json.loads((ROOT / "configs/zero_shot.json").read_text(encoding="utf-8"))
    return json.loads((ROOT / cfg["languages"][lang]["dataset_manifest"]).read_text(encoding="utf-8"))

STRATEGIES = ("all_in_one", "step_by_step")
COLS = ["language", "strategy", "shots", "subset", "n", "gloss_morpheme_accuracy", "gloss_word_accuracy",
        "gloss_mer", "gloss_wer", "gloss_chrfpp", "transliteration_chrfpp",
        "transliteration_word_accuracy", "parse_failure_rate", "truncation_rate"]


def score(df: pd.DataFrame) -> dict:
    res = evaluate(df.pred_gloss.tolist(), df.gold_gloss.tolist(),
                   df.pred_transliteration.tolist(), df.gold_transliteration.tolist())
    res["parse_failure_rate"] = (df.parse_status != "ok").mean()
    res["truncation_rate"] = (df.truncated.str.lower() == "true").mean()
    return res


def main(argv=None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="ainu")
    ap.add_argument("--check-korean", action="store_true")
    ap.add_argument("--shots", type=int, default=0, help="score few-shot results (0 = zero-shot)")
    args = ap.parse_args(argv)

    if args.check_korean:
        k = score(pd.read_csv(ROOT / "results/zero_shot/bolmo-1b_all_in_one_test.csv",
                              dtype=str, keep_default_na=False))
        got = round(k["gloss_morpheme_accuracy"], 4), round(k["gloss_word_accuracy"], 4)
        assert got == (0.0594, 0.0188), got
        print("Korean all-in-one check matches the evaluation branch:", got)

    cfg = json.loads((ROOT / "configs/zero_shot.json").read_text(encoding="utf-8"))
    if args.shots:
        folder = ROOT / cfg["few_shot"]["output_dir"].format(lang=args.lang)
    else:
        folder = ROOT / "results/zero_shot" / args.lang
    shot_tag = f"_{args.shots}shot" if args.shots else ""
    join_affixes = manifest_for(args.lang).get("gloss_affixes_detached", False)
    if join_affixes:
        print("Gloss affixes are detached in this dataset: re-attaching them before scoring.")
    rows = []
    for strategy in STRATEGIES:
        f = folder / f"bolmo-1b_{strategy}{shot_tag}_test.csv"
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        if join_affixes:
            df["gold_gloss"] = df.gold_gloss.map(join_detached_affixes)
            df["pred_gloss"] = df.pred_gloss.map(join_detached_affixes)
        aligned = [len(words(t)) == len(words(g)) for t, g in zip(df.gold_transliteration, df.gold_gloss)]
        for subset, d in (("all", df), ("aligned_word_count", df[aligned])):
            rows.append({"language": args.lang, "strategy": strategy, "shots": args.shots, "subset": subset,
                         "file": f.name, **score(d)})
    out = pd.DataFrame(rows)
    out = out[COLS + [c for c in out.columns if c not in COLS]]
    out.to_csv(folder / "scores.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(out[COLS].round(4).to_string(index=False))
    print("Wrote", folder / "scores.csv")
    return out


if __name__ == "__main__":
    main()
