"""Paired bootstrap confidence intervals for the comparisons we report (no GPU).

Our claims "A and B are within noise" and "A beats B" rest on one run each, so we check them
by resampling test sentences: draw the test rows with replacement, recompute corpus morpheme
accuracy (and morpheme bag F1) for both systems on the SAME draw, and take the 2.5th and
97.5th percentiles of the difference over 10,000 draws. A 95% interval that contains 0 means
the difference is not distinguishable from sentence sampling noise. (It does not cover
training-seed variance, which would need several training runs.)

Scores are computed exactly as in rescore_loop_stop.py: loop-stopped outputs, affixes
re-joined for Ainu and Jejueo, case-insensitive.

Usage (from the repository root): python evaluation/bootstrap_ci.py
Writes results/evaluation/bootstrap_ci.csv.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rescore_loop_stop as rs  # noqa: E402
from metrics import morphemes, normalize, positional_correct  # noqa: E402

pd, sz = rs.pd, rs.sz
ROOT = rs.ROOT
N_DRAWS, SEED = 10_000, 42

ZS, FS, ICL = "results/zero_shot/{l}/bolmo-1b_{s}_test.csv", "results/few_shot/{l}/bolmo-1b_{s}_3shot_test.csv", \
    "results/icl/{l}/bolmo-1b_{s}_icl21_test.csv"
LORA = "results/lora/{l}/bolmo-1b_{l}_{s}_r16_lr2e-4_s42/predictions/bolmo-1b_{l}_{s}_r16_lr2e-4_s42_{sel}_test.csv"

# (language, label, system A file, strategy A, system B file, strategy B)
COMPARISONS = [
    *[(l, f"{m}: step-by-step vs all-in-one", p.format(l=l, s="step_by_step"), "step_by_step",
       p.format(l=l, s="all_in_one"), "all_in_one")
      for l, m, p in [("ainu", "zero-shot", ZS), ("tsez", "zero-shot", ZS), ("jejueo", "zero-shot", ZS),
                      ("ainu", "3-shot", FS), ("jejueo", "3-shot", FS), ("ainu", "ICL-21", ICL)]],
    ("tsez", "LoRA: step-by-step vs all-in-one (loss-selected)",
     LORA.format(l="tsez", s="step_by_step", sel="loss-selected_step667"), "step_by_step",
     LORA.format(l="tsez", s="all_in_one", sel="loss-selected_step667"), "all_in_one"),
    ("tsez", "LoRA step-by-step: chrF-selected (555) vs loss-selected (667)",
     LORA.format(l="tsez", s="step_by_step", sel="chrf-selected_step555"), "step_by_step",
     LORA.format(l="tsez", s="step_by_step", sel="loss-selected_step667"), "step_by_step"),
    ("ainu", "3-shot vs zero-shot (all-in-one)", FS.format(l="ainu", s="all_in_one"), "all_in_one",
     ZS.format(l="ainu", s="all_in_one"), "all_in_one"),
    ("jejueo", "3-shot vs zero-shot (all-in-one)", FS.format(l="jejueo", s="all_in_one"), "all_in_one",
     ZS.format(l="jejueo", s="all_in_one"), "all_in_one"),
    ("ainu", "segmentation vs 3-shot step-by-step",
     "results/segmentation/ainu/bolmo-1b_three_step_seg_3shot_test.csv", "three_step_seg",
     FS.format(l="ainu", s="step_by_step"), "step_by_step"),
    ("ainu", "ICL-21 vs 3-shot (all-in-one)", ICL.format(l="ainu", s="all_in_one"), "all_in_one",
     FS.format(l="ainu", s="all_in_one"), "all_in_one"),
    ("tsez", "ICL-21 step-by-step vs all-in-one", ICL.format(l="tsez", s="step_by_step"), "step_by_step",
     ICL.format(l="tsez", s="all_in_one"), "all_in_one"),
    ("tsez", "ICL-21 vs zero-shot (all-in-one)", ICL.format(l="tsez", s="all_in_one"), "all_in_one",
     ZS.format(l="tsez", s="all_in_one"), "all_in_one"),
    ("tsez", "LoRA vs ICL-21 (all-in-one, same language)",
     LORA.format(l="tsez", s="all_in_one", sel="loss-selected_step667"), "all_in_one",
     ICL.format(l="tsez", s="all_in_one"), "all_in_one"),
    ("tsez", "LoRA vs zero-shot (all-in-one)",
     LORA.format(l="tsez", s="all_in_one", sel="loss-selected_step667"), "all_in_one",
     ZS.format(l="tsez", s="all_in_one"), "all_in_one"),
]


def per_sentence(lang: str, path: str, strategy: str) -> dict[str, np.ndarray]:
    """Per test row: positional-correct morphemes, gold morphemes, bag overlap, predicted morphemes."""
    df = pd.read_csv(ROOT / path, dtype=str, keep_default_na=False)
    kind = "three_step" if strategy == "three_step_seg" else "all_in_one"
    preds = [rs.parse(strategy, rs.cut(raw, kind)[0])[1] for raw in df.raw_output]
    golds = list(df.gold_gloss)
    if sz.manifest_for(lang).get("gloss_affixes_detached", False):
        preds = [sz.join_detached_affixes(p) for p in preds]
        golds = [sz.join_detached_affixes(g) for g in golds]
    out = {"correct": [], "gold": [], "overlap": [], "pred": []}
    for p_text, g_text in zip(preds, golds):
        p = [t.casefold() for t in morphemes(normalize(p_text))]
        g = [t.casefold() for t in morphemes(normalize(g_text))]
        out["correct"].append(positional_correct(p, g))
        out["gold"].append(len(g))
        out["overlap"].append(sum((Counter(p) & Counter(g)).values()))
        out["pred"].append(len(p))
    return {k: np.array(v, dtype=float) for k, v in out.items()} | {"ids": df["id"].tolist()}


def acc(s, idx):
    return s["correct"][idx].sum(axis=-1) / s["gold"][idx].sum(axis=-1)


def bag_f1(s, idx):
    o, p, g = s["overlap"][idx].sum(axis=-1), s["pred"][idx].sum(axis=-1), s["gold"][idx].sum(axis=-1)
    prec, rec = np.divide(o, p, out=np.zeros_like(o), where=p > 0), o / g
    return np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(o), where=(prec + rec) > 0)


def main() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for lang, label, fa, sa, fb, sb in COMPARISONS:
        if not ((ROOT / fa).exists() and (ROOT / fb).exists()):
            print("skipped (missing file):", label)
            continue
        a, b = per_sentence(lang, fa, sa), per_sentence(lang, fb, sb)
        assert a["ids"] == b["ids"], f"{label}: the two files do not list the same test rows in the same order"
        n = len(a["ids"])
        full = np.arange(n)
        draws = rng.integers(0, n, size=(N_DRAWS, n))
        row = {"language": lang, "comparison": label, "n": n}
        for name, fn in (("morpheme_acc", acc), ("morpheme_bag_f1", bag_f1)):
            diff = fn(a, draws) - fn(b, draws)
            lo, hi = np.percentile(diff, [2.5, 97.5])
            row.update({f"{name}_A": fn(a, full), f"{name}_B": fn(b, full),
                        f"{name}_diff": fn(a, full) - fn(b, full), f"{name}_ci_low": lo, f"{name}_ci_high": hi,
                        f"{name}_significant": not (lo <= 0 <= hi)})
        rows.append(row)
        print(f"{lang:7s} {label:62s} morph diff {100 * row['morpheme_acc_diff']:+6.2f} "
              f"[{100 * row['morpheme_acc_ci_low']:+6.2f}, {100 * row['morpheme_acc_ci_high']:+6.2f}]"
              f"{'  *' if row['morpheme_acc_significant'] else ''}")
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/evaluation/bootstrap_ci.csv", index=False, float_format="%.4f")
    print("Wrote results/evaluation/bootstrap_ci.csv  (* = 95% interval excludes 0)")
    return out


if __name__ == "__main__":
    main()
