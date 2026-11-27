"""Hand-checked cases for evaluation/metrics.py (Plan section 4.4).

Run: python -m pytest evaluation/test_metrics.py   (or: python evaluation/test_metrics.py)
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from metrics import (edit_distance, evaluate, join_detached_affixes, morphemes, normalize,  # noqa: E402
                     positional_correct, score_level, words)

GOLD = "guest-NOM two CL that wine-ACC drink-PST-DECL"


def test_tokenisation():
    assert words(GOLD) == ["guest-NOM", "two", "CL", "that", "wine-ACC", "drink-PST-DECL"]
    assert morphemes(GOLD) == ["guest", "NOM", "two", "CL", "that", "wine", "ACC",
                               "drink", "PST", "DECL"]
    # "=" is a boundary, "." is not
    assert morphemes("go=TOP do-DECL.POL") == ["go", "TOP", "do", "DECL.POL"]
    # empty pieces from stray hyphens are dropped
    assert morphemes("-NOM a--b") == ["NOM", "a", "b"]


def test_normalisation():
    assert normalize("  a \t b\n") == "a b"
    assert normalize(None) == "" and normalize(float("nan")) == ""
    # NFD Hangul (decomposed jamo) equals NFC Hangul after normalisation
    import unicodedata
    word = "마셨다"  # Korean "masyessta" (drank), NFC
    assert normalize(unicodedata.normalize("NFD", word)) == word


def test_identity():
    r = evaluate([GOLD], [GOLD])
    assert r["gloss_word_accuracy"] == 1 and r["gloss_morpheme_accuracy"] == 1
    assert r["gloss_mer"] == 0 and r["gloss_wer"] == 0
    assert math.isclose(r["gloss_chrfpp"], 100.0)


def test_substitution():
    pred = "guest-NOM two CL that wine-ACC drink-PST-POL"  # 1 morpheme wrong
    r = evaluate([pred], [GOLD])
    assert r["gloss_morpheme_accuracy"] == 9 / 10
    assert r["gloss_word_accuracy"] == 5 / 6
    assert r["gloss_mer"] == 1 / 10


def test_missing_morpheme_shifts_positions():
    # dropping "NOM" shifts every later morpheme: positional accuracy punishes it,
    # edit-based MER counts it as a single deletion
    pred = "guest two CL that wine-ACC drink-PST-DECL"
    r = evaluate([pred], [GOLD])
    assert r["gloss_morpheme_accuracy"] == 1 / 10   # only "guest" stays aligned
    assert r["gloss_mer"] == 1 / 10
    assert r["gloss_word_accuracy"] == 5 / 6        # words keep their positions


def test_extra_and_short_predictions():
    gold, short = ["a", "b", "c"], ["a"]
    assert positional_correct(short, gold) == 1
    assert positional_correct(gold + ["x", "y"], gold) == 3   # extras ignored
    assert edit_distance(gold + ["x", "y"], gold) == 2         # but counted in MER
    r = score_level(["a b c x y"], ["a b c"], words)
    assert r["accuracy"] == 1 and r["error_rate"] == 2 / 3


def test_case_insensitive_accuracy():
    pred = "student-NOM caus Neg"
    gold = "Student-NOM CAUS NEG"
    r = evaluate([pred], [gold])
    assert r["gloss_morpheme_accuracy"] == 1 and r["gloss_mer"] == 0
    assert r["gloss_morpheme_accuracy_case_sensitive"] == 1 / 4
    assert r["gloss_word_accuracy_case_sensitive"] == 0


def test_join_detached_affixes():
    # Jejueo test rows: suffixes are separate tokens in the dataset
    assert join_detached_affixes("NEG -POL -DECL 1SG -TOP eat -PST -POL -DECL") \
        == "NEG-POL-DECL 1SG-TOP eat-PST-POL-DECL"
    # a token ending in "-" attaches to the next one
    assert join_detached_affixes("DEM.PROX- thing_NOM see -PRS") == "DEM.PROX-thing_NOM see-PRS"
    assert join_detached_affixes("go =TOP") == "go=TOP"
    # already joined glosses and empty input are unchanged
    assert join_detached_affixes("guest-NOM two CL") == "guest-NOM two CL"
    assert join_detached_affixes("") == ""
    # morphemes do not change, only word boundaries
    g = "go -IRR -POL -DECL DEM.DIST.be -CVB.but approximately six hour.CLF -CVB.CND"
    assert morphemes(join_detached_affixes(g)) == morphemes(g)
    assert len(words(join_detached_affixes(g))) == 5
    # a lone "-" (Ainu: a pause, "..." in the transcription) stays a word of its own
    assert join_detached_affixes("various bird -DIM - various bird") == "various bird-DIM - various bird"
    assert join_detached_affixes("then - then 4.A= have") == "then - then 4.A=have"
    assert join_detached_affixes("outside -COP - 4.A= have") == "outside-COP - 4.A=have"


def test_empty_and_malformed_prediction():
    r = evaluate(["", None], [GOLD, "a-NOM"])
    assert r["gloss_morpheme_accuracy"] == 0 and r["gloss_word_accuracy"] == 0
    assert r["gloss_mer"] == 1
    assert r["n"] == 2 and r["gold_morphemes"] == 12


def test_empty_reference_not_divided_by_zero():
    r = score_level(["x"], [""], words)
    assert r["empty_refs"] == 1 and math.isnan(r["accuracy"])


def test_corpus_micro_average():
    # sentence 1: 1/1 correct, sentence 2: 0/3 correct -> micro 1/4, sentence avg 1/2
    r = score_level(["a", "x y z"], ["a", "b c d"], words)
    assert r["accuracy"] == 1 / 4
    assert r["avg_sentence_accuracy"] == 1 / 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
