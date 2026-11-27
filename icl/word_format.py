"""Word-by-word answer format for ICL (`run_icl.py --format word`).

Our generative models often add or drop a word in the gloss line, so every later gloss is one
position off and positional accuracy collapses (Tsez LoRA: 7.8% morpheme accuracy, while 44%
of the gold morphemes are present). Here the model glosses each word of its transliteration
on its own line, so the glosses cannot drift away from the words:

    Transliteration: Ħusenä salamno teƛno, iduɣor oqno.
    Gloss:
    Ħusenä = Hussein-ERG
    salamno = greeting-TOP
    ...

The examples are shown in the same format, from the gold transcription and gloss. Only train
rows whose transcription and gloss have the same number of words can be shown this way, so
the example pool keeps only those (Tsez: all 1,177; Jejueo: 282 of 316; Ainu: 670 of 5,388,
which is why Ainu is not a good fit). For Jejueo, whose glosses write affixes as separate
tokens, the gloss words are counted after re-joining them (metrics.join_detached_affixes).

For scoring, the per-word glosses are joined back into one gloss line (pred_gloss), so the
metrics and every scoring script stay the same as for the one-line format.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "zero_shot"))
sys.path.insert(0, str(ROOT / "evaluation"))
import run_zero_shot as rz  # noqa: E402
from metrics import join_detached_affixes  # noqa: E402

KIND = "word"
SEP = " = "
EMPTY_GLOSS = "_"  # placeholder for a word line with no gloss, so later glosses keep their position

ANSWER_LINES = ('Answer in two lines: the first line starts with "Transliteration:",\n'
                'the second line starts with "Gloss:".')
ANSWER_WORDS = ('Answer with a line that starts with "Transliteration:", then a line "Gloss:",\n'
                'then one line per word of the transliteration, in order: the word, " = ", its gloss.')

_GLOSS_LABEL = re.compile(r"^\s*gloss\s*:\s*(.*)$", re.IGNORECASE)
_WORD_LINE = re.compile(r"^\s*(\S.*?)\s+=\s+(.*?)\s*$")


# ----------------------------------------------------------------------------- examples


def gloss_words(gloss: str, join_affixes: bool) -> list[str]:
    g = rz._one_line(gloss)
    return (join_detached_affixes(g) if join_affixes else g).split()


def aligned(row: dict, join_affixes: bool) -> bool:
    """True if the gold transcription and gloss have the same number of words."""
    return len(rz._one_line(row["transcription"]).split()) == len(gloss_words(row["glosses"], join_affixes))


def word_instructions(template: str) -> str:
    """The template with its two-line answer instruction replaced by the word-by-word one."""
    if template.count(ANSWER_LINES) != 1:
        raise ValueError("answer instruction not found in the template")
    return template.replace(ANSWER_LINES, ANSWER_WORDS)


def examples_block(examples: list[dict], language_name: str, join_affixes: bool) -> str:
    parts = []
    for e in examples:
        words = rz._one_line(e["transcription"]).split()
        glosses = gloss_words(e["glosses"], join_affixes)
        if len(words) != len(glosses):
            raise ValueError(f"example {e['id']} is not word-aligned")
        lines = "\n".join(f"{w}{SEP}{g}" for w, g in zip(words, glosses))
        parts.append(
            f"Original {language_name} script: {rz._one_line(e['text'])}\n"
            f"English translation: {rz._one_line(e['translation'])}\n"
            f"Transliteration: {' '.join(words)}\n"
            f"Gloss:\n{lines}"
        )
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------- stop + parse


def _finished_lines(completion: str) -> list[str]:
    return completion.replace("\r\n", "\n").split("\n")[:-1]


def word_complete(completion: str) -> bool:
    """Done once the gloss block has ended: one finished word line per transliteration word,
    or a finished line after the word lines that is not a word line (a blank line or a new field)."""
    finished = _finished_lines(completion)
    if not finished:
        return False
    n_words = len(finished[0].split())
    seen_label = seen_word = 0
    for line in finished[1:]:
        if not seen_label:
            if _GLOSS_LABEL.match(line):
                seen_label = 1
                if _GLOSS_LABEL.match(line).group(1).strip():  # the model wrote the gloss on the label line
                    return True
            elif line.strip():
                return True  # something else instead of the gloss block: nothing more to wait for
            continue
        if _WORD_LINE.match(line):
            seen_word += 1
            if n_words and seen_word >= n_words:
                return True
        elif seen_word or line.strip():
            return True
    return False


_is_complete = rz.is_complete


def is_complete(kind: str, completion: str) -> bool:
    if kind == KIND:
        return word_complete(completion)
    return _is_complete(kind, completion)


rz.is_complete = is_complete  # the generators (original and fast) look it up here


def parse_word(completion: str) -> tuple[str, str, str]:
    """(transliteration, gloss joined into one line, parse_status).

    parse_status: ok | gloss_on_label_line (the one-line format, kept as the gloss) |
    missing_gloss (no 'Gloss:' label) | empty_gloss (label but no word line).
    """
    lines = completion.replace("\r\n", "\n").split("\n")
    translit = lines[0].strip()
    rest = lines[1:]
    i = next((k for k, l in enumerate(rest) if l.strip()), None)
    if i is None or not _GLOSS_LABEL.match(rest[i]):
        return translit, "", "missing_gloss"
    inline = _GLOSS_LABEL.match(rest[i]).group(1).strip()
    if inline:
        return translit, inline, "gloss_on_label_line"
    glosses = []
    for line in rest[i + 1:]:
        m = _WORD_LINE.match(line)
        if not m:
            if glosses or line.strip():
                break
            continue
        glosses.append(m.group(2).strip() or EMPTY_GLOSS)
    if not glosses:
        return translit, "", "empty_gloss"
    return translit, " ".join(glosses), "ok"
