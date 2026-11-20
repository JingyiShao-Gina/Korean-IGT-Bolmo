"""CPU-only checks for prompt rendering and output parsing.  Run: python zero_shot/test_parsing.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_zero_shot as rz  # noqa: E402

cfg = json.loads(rz.CONFIG_PATH.read_text(encoding="utf-8"))
GENERIC = {k: rz.load_prompt(v) for k, v in cfg["prompts"].items()}
T = {k: rz.fill_language(t, "Korean") for k, t in GENERIC.items()}

# --- prompt files are language-neutral
for k, t in GENERIC.items():
    assert "{language}" in t, k
    assert "Korean" not in t and "Ainu" not in t, k
    assert "\r" not in t
    assert "{transcription}" not in t and "{glosses}" not in t and "{gold" not in t

# --- templates end at the answer label
assert T["all_in_one"].endswith("\nTransliteration:"), T["all_in_one"][-40:]
assert T["step_by_step"].endswith("\nTransliteration:")
assert T["step_1_transliteration"].endswith("\nTransliteration:")
assert T["step_2_gloss"].endswith("\nGloss:")

ex = {"text": "존이 사과를 깎은 칼", "translation": "the knife with which John peeled an apple {x}"}
p = rz.render(T["all_in_one"], original_script=ex["text"], translation=ex["translation"])
assert "Original Korean script: 존이 사과를 깎은 칼" in p and "{x}" in p  # braces in data survive
assert rz.render(GENERIC["all_in_one"], language="Korean", original_script=ex["text"], translation=ex["translation"]) == p
p2 = rz.render(T["step_2_gloss"], original_script=ex["text"], translation=ex["translation"],
               predicted_transliteration="Johni sakwalul kkakun khal")
assert "Predicted transliteration: Johni sakwalul kkakun khal" in p2
for bad in (lambda: rz.render(T["step_2_gloss"], original_script="a", translation="b"),
            lambda: rz.render(GENERIC["all_in_one"], original_script="a", translation="b")):  # {language} left
    try:
        bad()
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
assert rz.parse_all_in_one("\nGloss: John-NOM\n") == ("", "John-NOM", "empty_transliteration")
assert rz.parse_all_in_one(" \nX\nGloss: g\n")[0] == "X"

# --- single-call step-by-step prompt (v4) and its parser
assert "Work step by step." in T["step_by_step"] and "Based on your transliteration from Step 1" in T["step_by_step"]
# v5: step-by-step differs from all-in-one only in the task-description lines
aio, sbs = T["all_in_one"].split("\n"), T["step_by_step"].split("\n")
assert aio[:5] == sbs[:5] and aio[-6:] == sbs[-6:], "only the task description may differ"
assert rz.parse_step_by_step(" Johni sakwalul\nGloss: John-NOM apple-ACC\nmore") == (
    "Johni sakwalul", "John-NOM apple-ACC", "ok")
assert rz.parse_step_by_step(" Johni sakwalul\nStep 2 - Gloss: John-NOM apple-ACC\nStep 3") == (
    "Johni sakwalul", "John-NOM apple-ACC", "ok")
assert rz.parse_step_by_step(" Johni\nstep 2: gloss: John-NOM\n") == ("Johni", "John-NOM", "ok")
assert rz.parse_step_by_step(" Johni\nTranslation: John\n")[2] == "missing_gloss"  # wrong label
assert rz.parse_step_by_step(" Johni\nStep 2 - Gloss:\n")[2] == "empty_gloss"

# --- the model leaves the transliteration empty and jumps to the gloss label (Ainu smoke test)
assert rz.parse_step_by_step("\nStep 2 - Gloss:\n") == ("", "", "empty_answer")
assert rz.parse_step_by_step("\nStep 2 - Gloss: fire-ACC\n") == ("", "fire-ACC", "empty_transliteration")
assert rz.parse_step_by_step("\n ape a= omare\nStep 2 - Gloss: fire\n") == ("ape a= omare", "fire", "ok")
assert rz.is_complete("all_in_one", "\nStep 2 - Gloss:\n")
assert not rz.is_complete("all_in_one", "\nStep 2 - Gloss:")
assert rz.is_complete("all_in_one", " V1 V2\nGloss: --\n")

# --- stopping criterion sees only the generated part (Bolmo passes [last prompt byte] + new)
prompt = [5, 6, 7, 8]
assert rz.generated_part([5, 6, 7, 8, 40, 41], prompt) == [40, 41]  # standard transformers
assert rz.generated_part([8, 40, 41], prompt) == [40, 41]           # Bolmo
assert rz.generated_part([8], prompt) == []

# --- languages: every language uses the shared prompts, only the name changes
for lang, L in cfg["languages"].items():
    name = L["language_name"]
    for key in ("all_in_one", "step_by_step"):
        t = rz.fill_language(GENERIC[key], name)
        assert f"Language: {name}\nOriginal {name} script: {{original_script}}" in t, (lang, key)
    assert (rz.ROOT / L["dataset_manifest"]).exists(), lang
m_ainu = json.loads((rz.ROOT / cfg["languages"]["ainu"]["dataset_manifest"]).read_text(encoding="utf-8"))
assert m_ainu["split_mapping"]["validation"] == "eval" and m_ainu["split_counts"]["test"] == 750

# --- filled with "Korean", the prompts are exactly the ones used before (v3 runs)
ref_dir = sys.argv[1] if len(sys.argv) > 1 else None
if ref_dir:
    ref = {"all_in_one": "all_in_one",  # step_by_step changed in v5, so it is not compared
           "step_1_transliteration": "transliteration", "step_2_gloss": "gloss"}
    for key, fname in ref.items():
        old = (Path(ref_dir) / f"{fname}.txt").read_text(encoding="utf-8").rstrip()
        assert T[key] == old, key
    print("Korean prompts identical to the previous files")

print("all parsing checks passed")
