"""Compare our accuracy with the SIGMORPHON 2023 glossing shared task's accuracy (no GPU).

The shared task's evaluation (https://github.com/sigmorphon/2023glossingST,
baseline/src/eval.py and data.py) compares token i of the prediction with token i of the
gold, case-sensitively, over the whole sentence: words = the gloss split on whitespace,
morphemes = the gloss split on whitespace and "-" (`re.split(r"\\s|-", gloss)`, empty pieces
kept, "=" not a boundary). It is re-implemented here, not copied, from that description.

Our metric (evaluation/metrics.py) is the same positional, corpus-level accuracy, with three
differences: "=" is also a morpheme boundary, empty pieces are dropped, and detached affixes
are re-joined for Ainu and Jejueo (manifest flag); by default it is case-insensitive
(case-sensitive columns are kept). On data without those quirks (Tsez) the two agree.

Usage (from the repository root): python evaluation/check_official_accuracy.py
Scores the loop-stopped all-in-one predictions of every test file rescore_loop_stop.py reads.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rescore_loop_stop as rs  # noqa: E402

pd, sz = rs.pd, rs.sz


def shared_task_accuracy(preds: list[list[str]], golds: list[list[str]]) -> float:
    """Corpus accuracy: gold token i counts as correct if prediction token i is identical (not '[UNK]')."""
    correct = total = 0
    for p, g in zip(preds, golds):
        correct += sum(1 for i, tok in enumerate(g) if i < len(p) and p[i] == tok and tok != "[UNK]")
        total += len(g)
    return correct / total if total else float("nan")


def st_morphemes(gloss: str) -> list[str]:
    return re.split(r"\s|-", gloss)


def main() -> pd.DataFrame:
    rows = []
    for lang, method, strategy, sel, step, f in rs.files(rs.LANGS):
        if strategy != "all_in_one" or sel == "chrf":
            continue
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        preds = [rs.parse(strategy, rs.cut(raw, "all_in_one")[0])[1] for raw in df.raw_output]
        golds = list(df.gold_gloss)
        ours = df.assign(pred_gloss=preds)
        if sz.manifest_for(lang).get("gloss_affixes_detached", False):
            ours["gold_gloss"] = ours.gold_gloss.map(sz.join_detached_affixes)
            ours["pred_gloss"] = ours.pred_gloss.map(sz.join_detached_affixes)
        s = sz.score(ours)
        rows.append({
            "language": lang, "method": method,
            "ours_morpheme": s["gloss_morpheme_accuracy"],
            "ours_morpheme_case_sensitive": s["gloss_morpheme_accuracy_case_sensitive"],
            "shared_task_morpheme": shared_task_accuracy([st_morphemes(p) for p in preds],
                                                         [st_morphemes(g) for g in golds]),
            "ours_word_case_sensitive": s["gloss_word_accuracy_case_sensitive"],
            "shared_task_word": shared_task_accuracy([p.split() for p in preds], [g.split() for g in golds]),
        })
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print((out.set_index(["language", "method"]) * 100).round(2).to_string())
    return out


if __name__ == "__main__":
    main()
