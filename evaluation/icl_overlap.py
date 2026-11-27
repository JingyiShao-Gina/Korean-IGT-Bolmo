"""How close are the retrieved ICL examples to the test sentences? (no GPU)

ICL raises Ainu morpheme accuracy from about 1.5% (3 fixed examples) to about 24%
(21 retrieved examples). The pool never contains a dev or test text, but a spoken corpus
repeats phrases, so an example can be almost the same sentence, or have the same gloss,
and the model can copy it. This script measures that and splits the scores by it.

For every test row of results/icl/<lang>/bolmo-1b_<strategy>_icl<K>_test.csv it looks up
the 21 examples (icl_example_ids) in the train split and computes:
    best_similarity    max character similarity between the test text and an example text
                       (difflib ratio after NFKC + whitespace normalisation; 1.0 = same text)
    gold_gloss_in_examples   an example has exactly the test row's gold gloss
    pred_copies_example      the predicted gloss equals one example's gloss
and scores each similarity band separately.

Usage (from the repository root):
    python evaluation/icl_overlap.py --lang ainu
Writes results/evaluation/icl_overlap_<lang>.csv (per row) and prints the summary,
also saved as results/evaluation/icl_overlap_summary.csv (all languages run so far).
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "zero_shot"), str(ROOT / "evaluation")]
import score_zero_shot as sz  # noqa: E402
import run_zero_shot as rz  # noqa: E402

pd = sz.pd
BANDS = [(1.0, 1.01, "identical text"), (0.9, 1.0, "similarity 0.9-1"), (0.8, 0.9, "similarity 0.8-0.9"),
         (0.0, 0.8, "similarity < 0.8")]


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(s or "")).split())


def gnorm(s: str) -> str:
    return " ".join(str(s or "").split()).lower()


def train_rows(lang: str) -> dict[str, dict]:
    cfg = json.loads((ROOT / "configs/zero_shot.json").read_text(encoding="utf-8"))
    manifest = ROOT / cfg["languages"][lang]["dataset_manifest"]
    return {str(r["id"]): r for r in rz.load_rows("train", None, manifest)}


def per_row(df: pd.DataFrame, train: dict[str, dict]) -> pd.DataFrame:
    out = []
    for _, r in df.iterrows():
        ex = [train[i] for i in r.icl_example_ids.split()]
        text = norm(r.text)
        sims = [difflib.SequenceMatcher(None, text, norm(e["text"]), autojunk=False).ratio() for e in ex]
        ex_glosses = {gnorm(e["glosses"]) for e in ex}
        out.append({
            "id": r.id,
            "best_similarity": max(sims),
            "most_similar_example": ex[sims.index(max(sims))]["id"],
            "gold_gloss_in_examples": gnorm(r.gold_gloss) in ex_glosses,
            "pred_copies_example": bool(gnorm(r.pred_gloss)) and gnorm(r.pred_gloss) in ex_glosses,
        })
    return pd.DataFrame(out)


def main(argv=None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="ainu")
    ap.add_argument("--k", type=int, default=21)
    args = ap.parse_args(argv)

    train = train_rows(args.lang)
    join = sz.manifest_for(args.lang).get("gloss_affixes_detached", False)
    summary, detail = [], []
    for strategy in sz.STRATEGIES:
        f = ROOT / "results/icl" / args.lang / f"bolmo-1b_{strategy}_icl{args.k}_test.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        info = per_row(df, train)
        scored = df.copy()
        if join:
            scored["gold_gloss"] = scored.gold_gloss.map(sz.join_detached_affixes)
            scored["pred_gloss"] = scored.pred_gloss.map(sz.join_detached_affixes)
        groups = [("all", pd.Series(True, index=df.index))]
        groups += [(name, (info.best_similarity >= lo) & (info.best_similarity < hi)) for lo, hi, name in BANDS]
        groups += [("gold gloss in examples", info.gold_gloss_in_examples),
                   ("gold gloss not in examples", ~info.gold_gloss_in_examples)]
        for name, mask in groups:
            d = scored[mask.values]
            row = {"language": args.lang, "strategy": strategy, "subset": name, "n": len(d),
                   "share": len(d) / len(df)}
            if len(d):
                s = sz.score(d)
                row.update({c: s[c] for c in ("gloss_morpheme_accuracy", "gloss_word_accuracy", "gloss_mer",
                                              "gloss_chrfpp", "transliteration_chrfpp")})
                row["pred_copies_example"] = info.pred_copies_example[mask.values].mean()
            summary.append(row)
        detail.append(info.assign(language=args.lang, strategy=strategy))

    if not summary:
        raise SystemExit(f"no ICL test files for {args.lang}")
    out_dir = ROOT / "results/evaluation"
    pd.concat(detail).to_csv(out_dir / f"icl_overlap_{args.lang}.csv", index=False, float_format="%.4f")
    summ = pd.DataFrame(summary)
    path = out_dir / "icl_overlap_summary.csv"
    if path.exists():  # keep the other languages' rows
        old = pd.read_csv(path)
        summ = pd.concat([old[old.language != args.lang], summ])
    summ.to_csv(path, index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    print(summ[summ.language == args.lang].round(4).to_string(index=False))
    print("Wrote", path.relative_to(ROOT), "and", f"results/evaluation/icl_overlap_{args.lang}.csv")
    return summ


if __name__ == "__main__":
    main()
