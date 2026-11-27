"""CPU checks for the word-by-word ICL format.  Run: python icl/test_word_format.py"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "evaluation")]
import word_format as wf  # noqa: E402
import run_icl  # noqa: E402,F401  (imports must work with the new option)
import rescore_loop_stop as rs  # noqa: E402

rz = wf.rz
cfg = json.loads(rz.CONFIG_PATH.read_text(encoding="utf-8"))

ex_tsez = {"id": 1, "text": "Хъуна гӀалъ.", "translation": "The house burned.",
           "transcription": "Quna ʕał.", "glosses": "house-ERG burn-PST.UNW"}
ex_jejueo = {"id": "j", "text": "아니우다 난 먹엇우다.", "translation": "No thanks, I ate already.",
             "transcription": "aniuta nan məkəsʰuta", "glosses": "NEG -POL -DECL 1SG -TOP eat -PST -POL -DECL"}
ex_bad = {"id": "b", "text": "x y", "translation": "t", "transcription": "a b c", "glosses": "A-B C"}

# --- alignment and the example block
assert wf.aligned(ex_tsez, False) and not wf.aligned(ex_bad, False)
assert not wf.aligned(ex_jejueo, False) and wf.aligned(ex_jejueo, True)   # Jejueo: after re-joining affixes
block = wf.examples_block([ex_tsez], "Tsez", False)
assert block.endswith("Transliteration: Quna ʕał.\nGloss:\nQuna = house-ERG\nʕał. = burn-PST.UNW"), block
assert "eat-PST-POL-DECL" in wf.examples_block([ex_jejueo], "Jejueo", True)
try:
    wf.examples_block([ex_bad], "Tsez", False)
    raise AssertionError("misaligned example not refused")
except ValueError:
    pass

# --- the instruction swap works on both shared templates, and the prompt ends like the examples
for key in ("all_in_one", "step_by_step"):
    t = wf.word_instructions(rz.fill_language(rz.load_prompt(cfg["prompts"][key]), "Tsez"))
    assert wf.ANSWER_WORDS in t and wf.ANSWER_LINES not in t and t.endswith("\nTransliteration:")
    p = rz.render(rz.add_examples(t, block, "Tsez"), original_script="Гьаьлоли.", translation="Hello.")
    assert p.endswith("Original Tsez script: Гьаьлоли.\nEnglish translation: Hello.\nTransliteration:"), p[-120:]
    assert p.count("\nGloss:\n") == 1

# --- parsing
out = " Ħusenä salamno teƛno,\nGloss:\nĦusenä = Hussein-ERG\nsalamno = greeting-TOP\nteƛno, = give-PFV.CVB\n\nOriginal"
assert wf.parse_word(out) == ("Ħusenä salamno teƛno,", "Hussein-ERG greeting-TOP give-PFV.CVB", "ok")
assert wf.parse_word(" a b\nGloss: x-ERG y\n") == ("a b", "x-ERG y", "gloss_on_label_line")
assert wf.parse_word(" a b\nTranslation: t\n")[2] == "missing_gloss"
assert wf.parse_word(" a b\nGloss:\n\nOriginal Tsez script: z\n")[2] == "empty_gloss"
assert wf.parse_word(" a b\nGloss:\na = \nb = y\n")[1] == "_ y"       # an empty gloss keeps its position
assert wf.parse_word(" a b\nGloss:\na = x\nnot a word line\nb = y\n")[1] == "x"  # stops at the first other line

# --- completion: one word line per transliteration word, or the block ends
assert not wf.word_complete(" a b\nGloss:\na = x\n")
assert wf.word_complete(" a b\nGloss:\na = x\nb = y\n")
assert wf.word_complete(" a b c\nGloss:\na = x\n\n")                 # blank line ends the block early
assert wf.word_complete(" a b\nGloss: x y\n")                       # one-line answer: nothing more to wait for
assert not wf.word_complete(" a b\nGloss:\n")
assert not wf.word_complete(" a b\nGloss:\n\n")                     # blank line before any word line: keep going
assert rz.is_complete("word", " a\nGloss:\na = x\n") and rz.is_complete("all_in_one", " a\nGloss: x\n")

# --- the unified scorer reads word-format files with the word parser and stop rule
assert rs.answer_format(Path("bolmo-1b_all_in_one_icl21_word_test.csv")) == "word"
assert rs.answer_format(Path("bolmo-1b_all_in_one_icl21_test.csv")) == "line"
t, why = rs.cut(out, wf.KIND)
assert why == "complete" and rs.parse("all_in_one", t, "word")[1] == "Hussein-ERG greeting-TOP give-PFV.CVB"
loop = " a b\nGloss:\n" + "a = x-ERG\n" * 40
t, why = rs.cut(loop, wf.KIND)
assert why == "complete" and t.count("a = x-ERG") == 2                # two word lines for two words, then stop

print("word-format checks passed")
