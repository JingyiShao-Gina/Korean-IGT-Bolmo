"""CPU-only checks for prompt rendering and output parsing.  Run: python zero_shot/test_parsing.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_zero_shot as rz  # noqa: E402

cfg = json.loads(rz.CONFIG_PATH.read_text(encoding="utf-8"))
T = {k: rz.load_prompt(v) for k, v in cfg["prompts"].items()}

# --- templates end at the answer label and contain no gold fields
assert T["all_in_one"].endswith("\nTransliteration:"), T["all_in_one"][-40:]
assert T["step_1_transliteration"].endswith("\nTransliteration:")
assert T["step_2_gloss"].endswith("\nGloss:")
for t in T.values():
    assert "\r" not in t
    assert "{transcription}" not in t and "{glosses}" not in t and "{gold" not in t

ex = {"text": "존이 사과를 깎은 칼", "translation": "the knife with which John peeled an apple {x}"}
p = rz.render(T["all_in_one"], original_script=ex["text"], translation=ex["translation"])
assert "Original Korean script: 존이 사과를 깎은 칼" in p and "{x}" in p  # braces in data survive
p2 = rz.render(T["step_2_gloss"], original_script=ex["text"], translation=ex["translation"],
               predicted_transliteration="Johni sakwalul kkakun khal")
assert "Predicted transliteration: Johni sakwalul kkakun khal" in p2
try:
    rz.render(T["step_2_gloss"], original_script="a", translation="b")
    raise AssertionError("missing placeholder not detected")
except ValueError:
    pass

# --- completion detection (used for stopping and truncation)
assert not rz.is_complete("single", " Johni sakwalul")
assert rz.is_complete("single", " Johni sakwalul\n")
assert rz.is_complete("single", "\n\n Johni\n")          # leading blank lines skipped
assert not rz.is_complete("all_in_one", " Johni\nGloss: John-NOM")
assert rz.is_complete("all_in_one", " Johni\nGloss: John-NOM\n")
assert not rz.is_complete("all_in_one", " Johni\n\n")

# --- parsing
assert rz.parse_single(" Johni sakwalul kkakun khal\nGloss: x\n") == "Johni sakwalul kkakun khal"
assert rz.parse_single("") == ""
assert rz.parse_all_in_one(" Johni sakwalul\nGloss: John-NOM apple-ACC\nmore") == (
    "Johni sakwalul", "John-NOM apple-ACC", "ok")
assert rz.parse_all_in_one(" Johni\n\ngloss :  John-NOM \n") == ("Johni", "John-NOM", "ok")
assert rz.parse_all_in_one(" Johni\nTranslation: John\n")[2] == "missing_gloss"
assert rz.parse_all_in_one(" Johni sakwalul")[2] == "missing_gloss"
assert rz.parse_all_in_one(" Johni\nGloss:\n")[2] == "empty_gloss"
assert rz.parse_all_in_one("\nGloss: John-NOM\n") == ("Gloss: John-NOM", "", "missing_gloss")
assert rz.parse_all_in_one(" \nX\nGloss: g\n")[0] == "X"

# --- single-call step-by-step prompt (v4) and its parser
assert T["step_by_step"].endswith("\nStep 1 - Transliteration:")
assert "Work step by step." in T["step_by_step"] and "Based on your transliteration from Step 1" in T["step_by_step"]
assert rz.parse_step_by_step(" Johni sakwalul\nStep 2 - Gloss: John-NOM apple-ACC\nStep 3") == (
    "Johni sakwalul", "John-NOM apple-ACC", "ok")
assert rz.parse_step_by_step(" Johni\nstep 2: gloss: John-NOM\n") == ("Johni", "John-NOM", "ok")
assert rz.parse_step_by_step(" Johni\nGloss: John-NOM\n")[2] == "missing_gloss"  # wrong label
assert rz.parse_step_by_step(" Johni\nStep 2 - Gloss:\n")[2] == "empty_gloss"

# --- stopping criterion sees only the generated part (Bolmo passes [last prompt byte] + new)
prompt = [5, 6, 7, 8]
assert rz.generated_part([5, 6, 7, 8, 40, 41], prompt) == [40, 41]  # standard transformers
assert rz.generated_part([8, 40, 41], prompt) == [40, 41]           # Bolmo
assert rz.generated_part([8], prompt) == []

# --- languages: Korean keeps its files; Ainu has its own prompts and manifest
L = cfg["languages"]
assert L["korean"]["prompts"]["all_in_one"] == cfg["prompts"]["all_in_one"]
for lang, word in (("korean", "Korean"), ("ainu", "Ainu")):
    for key in ("all_in_one", "step_by_step"):
        t = rz.load_prompt(L[lang]["prompts"][key])
        assert f"Language: {word}\nOriginal {word} script: {{original_script}}" in t, (lang, key)
        other = "Ainu" if word == "Korean" else "Korean"
        assert other not in t, (lang, key)
    # Ainu prompts are the Korean ones with only the language name changed
    if lang == "ainu":
        for key in ("all_in_one", "step_by_step"):
            assert rz.load_prompt(L["ainu"]["prompts"][key]) == T[key].replace("Korean", "Ainu")
m_ainu = json.loads((rz.ROOT / L["ainu"]["dataset_manifest"]).read_text(encoding="utf-8"))
assert m_ainu["split_mapping"]["validation"] == "eval" and m_ainu["split_counts"]["test"] == 750

print("all parsing checks passed")
