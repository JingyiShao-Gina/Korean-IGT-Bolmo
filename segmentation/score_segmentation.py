"""Score the segmentation run with the same metrics as zero_shot/score_zero_shot.py.

Usage (from the repository root; needs pandas and sacrebleu):
    python segmentation/score_segmentation.py --lang ainu            # 3-shot test run
    python segmentation/score_segmentation.py --lang ainu --file results/segmentation/ainu/<file>.csv

Writes scores.csv next to the predictions: all rows, and the subset whose gold transcription
and gloss have the same number of words (as for the other runs). Extra column
segmentation_exact_rate: share of rows whose predicted Segmentation line equals the gold
transcription (whitespace normalised).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "zero_shot"))
import score_zero_shot as sz  # noqa: E402

pd, words = sz.pd, sz.words


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="ainu")
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--file", default=None, help="default: the test run of --lang and --shots")
    args = ap.parse_args(argv)

    f = Path(args.file) if args.file else (
        ROOT / "results/segmentation" / args.lang / f"bolmo-1b_three_step_seg_{args.shots}shot_test.csv")
    df = pd.read_csv(f, dtype=str, keep_default_na=False)
    if sz.manifest_for(args.lang).get("gloss_affixes_detached", False):
        df["gold_gloss"] = df.gold_gloss.map(sz.join_detached_affixes)
        df["pred_gloss"] = df.pred_gloss.map(sz.join_detached_affixes)
    aligned = [len(words(t)) == len(words(g)) for t, g in zip(df.gold_transliteration, df.gold_gloss)]
    rows = []
    for subset, d in (("all", df), ("aligned_word_count", df[aligned])):
        exact = (d.pred_transliteration.map(lambda s: " ".join(s.split()))
                 == d.gold_transliteration.map(lambda s: " ".join(s.split()))).mean()
        rows.append({"language": args.lang, "strategy": "three_step_seg", "shots": args.shots,
                     "subset": subset, "file": f.name, **sz.score(d), "segmentation_exact_rate": exact})
    out = pd.DataFrame(rows)
    out = out[sz.COLS + [c for c in out.columns if c not in sz.COLS]]
    out.to_csv(f.parent / "scores.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    print("Wrote", f.parent / "scores.csv")
    return out


if __name__ == "__main__":
    main()
