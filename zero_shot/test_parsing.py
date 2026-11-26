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

# --- few-shot: example selection and prompt insertion
train = [
    {"id": "a", "text": "아니우다 난 먹엇우다.", "translation": "No thanks, I ate already.",
     "transcription": "aniuta nan məkəsʰuta", "glosses": "NEG -POL -DECL 1SG -TOP eat -PST -POL -DECL"},
    {"id": "h", "text": "야, 알앗우다.", "translation": "Alright!", "transcription": "ja alasʰuta",
     "glosses": "yes know -PST -POL -DECL"},                                                # 2 words
    {"id": "b", "text": "[ ]", "translation": "[ ]", "transcription": "x", "glosses": "X-NOM"},    # placeholder
    {"id": "c", "text": "뭐", "translation": "what", "transcription": "mə", "glosses": "what"},     # no gloss tag
    {"id": "d", "text": "아니 아니 궨찮수다.", "translation": "No, no, it's fine.",
     "transcription": "ani ani kʷent͡ɕʰanhsʰuta", "glosses": "NEG NEG be_okay -POL -DECL"},
    {"id": "e", "text": "a test sentence", "translation": "t", "transcription": "t", "glosses": "t-NOM"},  # in test
    {"id": "f", "text": "x" * 61, "translation": "t", "transcription": "t", "glosses": "t-NOM"},      # too long
    {"id": "g", "text": "a b\nc", "translation": "t", "transcription": "t", "glosses": ""},            # empty gloss
]
picked = rz.select_examples(train, 2, seed=42, exclude_texts=["a test sentence"])
assert sorted(e["id"] for e in picked) == ["a", "d"]
assert rz.select_examples(train, 2, 42, ["a test sentence"]) == picked          # deterministic
try:
    rz.select_examples(train, 3, 42, ["a test sentence"])
    raise AssertionError("too few qualifying rows not detected")
except ValueError:
    pass
JT = {k: rz.fill_language(t, "Jejueo") for k, t in GENERIC.items()}
for key in ("all_in_one", "step_by_step"):
    fs = rz.add_examples(JT[key], rz.examples_block(picked, "Jejueo"), "Jejueo")
    p = rz.render(fs, original_script="갈거우다.", translation="I will go.")
    # the target sentence comes last, in the examples' format, right before the answer label
    assert p.endswith("Original Jejueo script: 갈거우다.\nEnglish translation: I will go.\nTransliteration:")
    assert p.count("Original Jejueo script:") == 3                              # 2 examples + target
    assert p.count("\nTransliteration:") == 3 and p.count("\nGloss:") == 2
    assert "Gloss: NEG -POL -DECL 1SG -TOP eat -PST -POL -DECL" in p
    # instructions come first, unchanged
    assert p.index("Do not explain your answer.") < p.index("Original Jejueo script:")
    fields = "Original Jejueo script: {original_script}\nEnglish translation: {translation}\n"
    assert p.startswith(JT[key].replace(fields, "").rsplit("\n\nTransliteration:", 1)[0].rstrip() + "\n\n")
assert rz.examples_block([{**train[0], "glosses": "NEG\n-POL"}], "Jejueo").endswith("Gloss: NEG -POL")

# --- --resume: a run interrupted after 3 rows and resumed equals an uninterrupted run
import tempfile  # noqa: E402


class FakeGen:
    """Deterministic stand-in for the model; optionally crashes after `fail_after` calls."""
    def __init__(self, fail_after=None):
        self.calls, self.fail_after = 0, fail_after

    def complete(self, prompt, kind):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated disconnect")
        word = prompt.split("Original Korean script: ")[1].split("\n")[0]
        return f" {word}-tr\nGloss: {word}-GL\n", 10, False


rows = [{"id": f"r{i}", "text": f"w{i}", "translation": f"t{i}", "transcription": f"w{i}",
         "glosses": f"g{i}"} for i in range(7)]
sig = {"language": "korean", "strategy": "all_in_one", "shots": 0}
full = rz.run_all_in_one(FakeGen(), rows, T["all_in_one"])
with tempfile.TemporaryDirectory() as tmp:
    ck = rz.checkpoint_path(Path(tmp) / "out.csv")
    fh = rz.open_checkpoint(ck, sig)
    try:
        rz.run_all_in_one(FakeGen(fail_after=3), rows, T["all_in_one"], lambda r: rz.append_checkpoint(fh, r))
        raise AssertionError("simulated crash did not happen")
    except RuntimeError:
        pass
    fh.close()
    with open(ck, "a", encoding="utf-8") as raw:  # a half-written last line from the crash
        raw.write('{"id": "r3", "pred_gl')
    done = rz.load_checkpoint(ck, sig)
    assert sorted(done) == ["r0", "r1", "r2"], sorted(done)
    fh = rz.open_checkpoint(ck, sig)                # resume appends on a fresh line
    todo = [r for r in rows if r["id"] not in done]
    new = rz.run_all_in_one(FakeGen(), todo, T["all_in_one"], lambda r: rz.append_checkpoint(fh, r))
    fh.close()
    by_id = {**done, **{r["id"]: r for r in new}}
    assert [by_id[r["id"]] for r in rows] == full    # identical to the uninterrupted run
    assert sorted(rz.load_checkpoint(ck, sig)) == [r["id"] for r in rows]
    try:
        rz.load_checkpoint(ck, {**sig, "shots": 3})  # other settings must not be mixed in
        raise AssertionError("signature mismatch not detected")
    except SystemExit:
        pass

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
