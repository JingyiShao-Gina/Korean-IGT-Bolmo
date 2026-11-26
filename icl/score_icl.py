"""Score the in-context learning runs with the same metrics as zero_shot/score_zero_shot.py.

Usage (from the repository root; needs pandas and sacrebleu):
    python icl/score_icl.py --lang ainu            # K = 21, test split
    python icl/score_icl.py --lang ainu --k 25

Scores results/icl/<lang>/bolmo-1b_<strategy>_icl<K>_test.csv for both strategies that
exist and writes scores.csv in the same folder: all rows, and the subset whose gold
transcription and gloss have the same number of words (as for the other runs).
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
    ap.add_argument("--k", type=int, default=21)
    args = ap.parse_args(argv)

    folder = ROOT / "results/icl" / args.lang
    join_affixes = sz.manifest_for(args.lang).get("gloss_affixes_detached", False)
    rows = []
    for strategy in sz.STRATEGIES:
        f = folder / f"bolmo-1b_{strategy}_icl{args.k}_test.csv"
        if not f.exists():
            print("missing, skipped:", f.name)
            continue
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        if join_affixes:
            df["gold_gloss"] = df.gold_gloss.map(sz.join_detached_affixes)
            df["pred_gloss"] = df.pred_gloss.map(sz.join_detached_affixes)
        aligned = [len(words(t)) == len(words(g)) for t, g in zip(df.gold_transliteration, df.gold_gloss)]
        for subset, d in (("all", df), ("aligned_word_count", df[aligned])):
            rows.append({"language": args.lang, "strategy": strategy, "shots": f"icl{args.k}",
                         "subset": subset, "file": f.name, **sz.score(d)})
    if not rows:
        raise SystemExit(f"no ICL test files in {folder}")
    out = pd.DataFrame(rows)
    out = out[sz.COLS + [c for c in out.columns if c not in sz.COLS]]
    out.to_csv(folder / "scores.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(out[sz.COLS].round(4).to_string(index=False))
    print("Wrote", folder / "scores.csv")
    return out


if __name__ == "__main__":
    main()
