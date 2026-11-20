"""Score zero-shot prediction CSVs with the team metrics (evaluation/metrics.py, by Soyeong).

Usage (from the repository root; needs pandas and sacrebleu):
    python zero_shot/score_zero_shot.py --lang ainu

Scores results/zero_shot/<lang>/bolmo-1b_<strategy>_test.csv and writes
results/zero_shot/<lang>/scores.csv. Each strategy is scored on all test rows and on
the subset whose gold transcription and gloss have the same number of words (the
Ainu data is not cleaned, and only 180/750 Ainu test rows are aligned).

--check-korean also re-scores the Korean all-in-one file and asserts that it matches
results/evaluation/test_scores.csv on the soyeong/evaluation branch (0.0594 morpheme
accuracy), to show that this script computes the same numbers.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
import pandas as pd  # noqa: E402
from metrics import evaluate, words  # noqa: E402

STRATEGIES = ("all_in_one", "step_by_step")
COLS = ["language", "strategy", "subset", "n", "gloss_morpheme_accuracy", "gloss_word_accuracy",
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
    args = ap.parse_args(argv)

    if args.check_korean:
        k = score(pd.read_csv(ROOT / "results/zero_shot/bolmo-1b_all_in_one_test.csv",
                              dtype=str, keep_default_na=False))
        got = round(k["gloss_morpheme_accuracy"], 4), round(k["gloss_word_accuracy"], 4)
        assert got == (0.0594, 0.0188), got
        print("Korean all-in-one check matches the evaluation branch:", got)

    folder = ROOT / "results/zero_shot" / args.lang
    rows = []
    for strategy in STRATEGIES:
        f = folder / f"bolmo-1b_{strategy}_test.csv"
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        aligned = [len(words(t)) == len(words(g)) for t, g in zip(df.gold_transliteration, df.gold_gloss)]
        for subset, d in (("all", df), ("aligned_word_count", df[aligned])):
            rows.append({"language": args.lang, "strategy": strategy, "subset": subset,
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
