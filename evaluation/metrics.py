"""Gloss evaluation metrics for Korean IGT.

Primary accuracy follows Ginn et al. (2024) / the SIGMORPHON 2023 glossing
shared-task evaluator: tokens are compared *by position* (the i-th predicted
token against the i-th gold token). Missing predicted tokens count as wrong;
extra predicted tokens are ignored. Accuracy = correct / gold tokens, summed
over the corpus (micro). The per-sentence average is reported as well.

Edit-based scores (MER, WER) use Levenshtein distance over the same tokens
and do penalise extra tokens.

Accuracy and MER/WER compare tokens case-insensitively (casefold): the cleaned
data unified the common tags, but rare tags (neg/NEG, Caus/CAUS) and
sentence-initial English words (Student/student) still differ in case
(see the dataset cleaning summary). Case-sensitive accuracy is also reported
for reference. chrF++ stays case-sensitive, matching the LoRA pipeline.

Tokenisation:
  * normalise: Unicode NFC, collapse whitespace, strip
  * words:     whitespace split
  * morphemes: every word split on "-" and "=" (empty pieces dropped),
               flattened over the sentence. "." inside a label such as
               "DECL.POL" is NOT a boundary.
"""
from __future__ import annotations

import math
import re
import unicodedata

MORPHEME_BOUNDARY = re.compile(r"[-=]")

CHRF_CONFIG = {"char_order": 6, "word_order": 2, "beta": 2,
               "lowercase": False, "whitespace": False}


def normalize(text) -> str:
    """NFC + collapsed whitespace. None/NaN become the empty string."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    text = unicodedata.normalize("NFC", str(text))
    return " ".join(text.split())


def words(text) -> list[str]:
    return normalize(text).split()


def morphemes(text) -> list[str]:
    return [m for w in words(text) for m in MORPHEME_BOUNDARY.split(w) if m]


def join_detached_affixes(text) -> str:
    """Re-attach affixes that a dataset writes as separate tokens.

    Some datasets (e.g. Jejueo, from wav2gloss/DoReCo) write a gloss as
    "eat -PST -POL -DECL" instead of "eat-PST-POL-DECL". A token that starts with
    "-" or "=" is joined to the previous token, and a token that ends with "-" or
    "=" is joined to the next one. Morphemes are unchanged; only the word
    boundaries are restored, so that word accuracy compares whole words.

    A token made only of "-" / "=" is a word of its own and is never joined: Ainu
    glosses use a lone "-" for a pause ("..." in the transcription), as in
    "various bird -DIM - various bird" -> "various bird-DIM - various bird".
    """
    out: list[str] = []
    glue = False
    for tok in words(text):
        lone = tok.strip("-=") == ""
        if out and not lone and (glue or tok[0] in "-="):
            out[-1] += tok
        else:
            out.append(tok)
        glue = not lone and tok[-1] in "-="
    return " ".join(out)


def positional_correct(pred: list[str], gold: list[str]) -> int:
    """Number of gold positions whose token is predicted at the same index."""
    return sum(1 for i, g in enumerate(gold) if i < len(pred) and pred[i] == g)


def edit_distance(pred: list[str], gold: list[str]) -> int:
    """Token-level Levenshtein distance (substitution, deletion, insertion = 1)."""
    prev = list(range(len(pred) + 1))
    for i, g in enumerate(gold, 1):
        cur = [i] + [0] * len(pred)
        for j, p in enumerate(pred, 1):
            cur[j] = min(prev[j] + 1,              # gold token deleted
                         cur[j - 1] + 1,           # extra predicted token
                         prev[j - 1] + (p != g))   # match / substitution
        prev = cur
    return prev[-1]


def score_level(preds: list[str], golds: list[str], tokenize, casefold: bool = True) -> dict:
    """Corpus scores at one level (word or morpheme)."""
    correct = total = edits = 0
    sentence_acc = []
    empty_refs = 0
    for p_text, g_text in zip(preds, golds, strict=True):
        p, g = tokenize(p_text), tokenize(g_text)
        if casefold:
            p, g = [t.casefold() for t in p], [t.casefold() for t in g]
        if not g:
            empty_refs += 1
            continue
        c = positional_correct(p, g)
        correct += c
        total += len(g)
        edits += edit_distance(p, g)
        sentence_acc.append(c / len(g))
    return {
        "accuracy": correct / total if total else float("nan"),
        "avg_sentence_accuracy": (sum(sentence_acc) / len(sentence_acc)
                                  if sentence_acc else float("nan")),
        "error_rate": edits / total if total else float("nan"),
        "correct": correct,
        "edits": edits,
        "total": total,
        "empty_refs": empty_refs,
    }


def bag_scores(preds: list[str], golds: list[str], tokenize, casefold: bool = True) -> dict:
    """Order-free corpus precision / recall / F1: how many gold tokens the prediction contains,
    wherever they stand (multiset overlap per sentence, summed over the corpus).

    Positional accuracy counts a token only at its own index, so one missing or extra word
    early in a sentence makes every later token wrong (gold minus its first word scores
    about 1% morpheme accuracy). This score shows how much of the content is right
    regardless of order; report it next to accuracy, not instead of it.
    """
    from collections import Counter
    overlap = n_pred = n_gold = 0
    for p_text, g_text in zip(preds, golds, strict=True):
        p, g = tokenize(p_text), tokenize(g_text)
        if casefold:
            p, g = [t.casefold() for t in p], [t.casefold() for t in g]
        overlap += sum((Counter(p) & Counter(g)).values())
        n_pred += len(p)
        n_gold += len(g)
    precision = overlap / n_pred if n_pred else 0.0
    recall = overlap / n_gold if n_gold else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def chrf_pp(preds: list[str], golds: list[str]) -> tuple[float, str]:
    """Corpus chrF++ with the same settings as the LoRA pipeline."""
    from sacrebleu.metrics import CHRF
    metric = CHRF(**CHRF_CONFIG)
    hyps = [normalize(p) for p in preds]
    refs = [normalize(g) for g in golds]
    return metric.corpus_score(hyps, [refs]).score, str(metric.get_signature())


def evaluate(pred_gloss, gold_gloss, pred_translit=None, gold_translit=None) -> dict:
    """All metrics for one prediction set. Inputs are parallel lists of strings."""
    pred_gloss = [normalize(x) for x in pred_gloss]
    gold_gloss = [normalize(x) for x in gold_gloss]
    w = score_level(pred_gloss, gold_gloss, words)
    m = score_level(pred_gloss, gold_gloss, morphemes)
    w_cs = score_level(pred_gloss, gold_gloss, words, casefold=False)
    m_cs = score_level(pred_gloss, gold_gloss, morphemes, casefold=False)
    m_bag = bag_scores(pred_gloss, gold_gloss, morphemes)
    g_chrf, sig = chrf_pp(pred_gloss, gold_gloss)
    out = {
        "n": len(gold_gloss),
        "gloss_word_accuracy": w["accuracy"],
        "gloss_morpheme_accuracy": m["accuracy"],
        "gloss_word_accuracy_case_sensitive": w_cs["accuracy"],
        "gloss_morpheme_accuracy_case_sensitive": m_cs["accuracy"],
        "gloss_word_accuracy_sent_avg": w["avg_sentence_accuracy"],
        "gloss_morpheme_accuracy_sent_avg": m["avg_sentence_accuracy"],
        "gloss_wer": w["error_rate"],
        "gloss_mer": m["error_rate"],
        "gloss_morpheme_bag_f1": m_bag["f1"],
        "gloss_morpheme_bag_precision": m_bag["precision"],
        "gloss_morpheme_bag_recall": m_bag["recall"],
        "gloss_chrfpp": g_chrf,
        "gold_words": w["total"],
        "gold_morphemes": m["total"],
        "empty_gold_gloss": m["empty_refs"],
        "chrf_signature": sig,
    }
    if pred_translit is not None and gold_translit is not None:
        t_chrf, _ = chrf_pp(pred_translit, gold_translit)
        t_w = score_level([normalize(x) for x in pred_translit],
                          [normalize(x) for x in gold_translit], words)
        out["transliteration_chrfpp"] = t_chrf
        out["transliteration_word_accuracy"] = t_w["accuracy"]
    return out
