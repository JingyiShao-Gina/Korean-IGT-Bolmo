"""CPU checks for evaluation/rescore_loop_stop.py.  Run: python evaluation/test_rescore.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rescore_loop_stop as rs  # noqa: E402

# a finished two-line answer is cut right after the gloss line; the tail is dropped
raw = " ani nan\nGloss: NEG 1SG -TOP\nOriginal Jejueo script: ..."
t, why = rs.cut(raw, "all_in_one")
assert why == "complete" and t == " ani nan\nGloss: NEG 1SG -TOP\n", (why, t)
assert rs.parse("all_in_one", t) == ("ani nan", "NEG 1SG -TOP", "ok")

# a looping gloss is cut once the loop is detected, and the answer keeps its first words
raw = " ani\nGloss: do -CG" + " do -CG" * 200
t, why = rs.cut(raw, "all_in_one")
assert why == "loop" and len(t) < 120 and t.startswith(" ani\nGloss: do -CG"), (why, len(t))
assert rs.parse("all_in_one", t)[1].startswith("do -CG do -CG")

# a looping transliteration line never reaches the gloss: still a parse failure
t, why = rs.cut(" aɪ" * 300, "all_in_one")
assert why == "loop" and rs.parse("all_in_one", t)[2] == "missing_gloss"

# the byte cap
t, why = rs.cut("".join(chr(0xAC00 + i) for i in range(600)), "all_in_one")  # 600 distinct syllables, 3 bytes each
assert why == "byte_cap" and 1000 <= len(t.encode()) < 1003, (why, len(t.encode()))  # stops at a whole character

# segmentation answers have three lines: not cut after the Segmentation line
raw = " ekan\nSegmentation: ek =an\nGloss: come.SG =4.S\nmore"
t, why = rs.cut(raw, "three_step")
assert why == "complete" and t.endswith("Gloss: come.SG =4.S\n"), (why, t)
assert rs.parse("three_step_seg", t) == ("ek =an", "come.SG =4.S", "ok")

# the two-line rule is exactly fast_run's replay rule
for raw in (" a\nGloss: b\nc", " x" * 400, ""):
    assert rs.cut(raw, "all_in_one") == rs.replay.cut(raw)

print("rescore checks passed")
