"""CPU-only checks for the LoRA data pipeline (no model, no GPU):  python lora/test_lora.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lora_bolmo as lb  # noqa: E402

rz = lb.rz
cfg = lb.load_config()
T = lb.load_templates(cfg)
ROW = {"id": "x1", "text": "존이 사과를 깎은 칼", "translation": "the knife with which John peeled an apple",
       "transcription": "Johni sakwalul kkakun khal", "glosses": "John-NOM apple-ACC peel-ADN knife"}


def test_all_in_one_target_round_trips_through_zero_shot_parser():
    [ex] = lb.build_examples([ROW], "all_in_one", T)
    assert ex["prompt"].endswith("Transliteration:")
    translit, gloss, status = rz.parse_all_in_one(ex["target"])
    assert (translit, gloss, status) == (ROW["transcription"], ROW["glosses"], "ok")


def test_two_stage_uses_gold_transliteration_for_training():
    s1, s2 = lb.build_examples([ROW], "two_stage", T)
    assert (s1["kind"], s2["kind"]) == ("transliteration", "gloss")
    assert s1["prompt"].endswith("Transliteration:") and s2["prompt"].endswith("Gloss:")
    assert f"Predicted transliteration: {ROW['transcription']}" in s2["prompt"]
    assert ROW["glosses"] not in s1["prompt"] + s2["prompt"]
    assert rz.parse_single(s1["target"]) == ROW["transcription"]
    assert rz.parse_single(s2["target"]) == ROW["glosses"]


def test_encoding_masks_prompt_and_appends_eos():
    e = lb.encode_example("ab:", " 한", bos=1, eos=1, offset=4)
    assert e["ids"] == [1, 4 + 97, 4 + 98, 4 + 58, 4 + 32, 4 + 0xED, 4 + 0x95, 4 + 0x9C, 1]
    assert e["n_prompt"] == 4 and e["n_target"] == 5
    assert len(e["ids"]) - e["n_prompt"] == e["n_target"]


def test_eval_steps_every_half_epoch():
    total, steps = lb.eval_steps(2079, 3, 16, 0.5)
    assert total == 390 and steps == [65, 130, 195, 260, 325, 390]
    total, steps = lb.eval_steps(4158, 3, 16, 0.5)
    assert total == 780 and steps == [130, 260, 390, 520, 650, 780]


def test_micro_order_is_deterministic_and_covers_each_epoch():
    a, b = lb.micro_order(10, 3, 42), lb.micro_order(10, 3, 42)
    assert a == b and len(a) == 30
    for k in range(3):
        assert sorted(a[10 * k:10 * (k + 1)]) == list(range(10))
    assert a[:10] != a[10:20]


def test_selection_rules_and_tie_break():
    m = [{"step": "65", "epoch": "0.5", "val_loss": "1.0", "val_gloss_chrfpp": "40", "val_transliteration_chrfpp": "1"},
         {"step": "130", "epoch": "1", "val_loss": "0.8", "val_gloss_chrfpp": "45", "val_transliteration_chrfpp": "1"},
         {"step": "195", "epoch": "1.5", "val_loss": "0.8", "val_gloss_chrfpp": "45", "val_transliteration_chrfpp": "1"},
         {"step": "260", "epoch": "2", "val_loss": "0.9", "val_gloss_chrfpp": "44", "val_transliteration_chrfpp": "1"}]
    sel = lb.select_checkpoints(m)
    assert sel["loss"]["step"] == "130" and sel["chrf"]["step"] == "130" and sel["same_checkpoint"]


def test_run_settings_and_ids():
    s = lb.run_settings(cfg, "baseline", "all_in_one")
    assert s["run_id"] == "bolmo-1b_all_in_one_r16_lr2e-4_s42"
    s = lb.run_settings(cfg, "r32", "two_stage", seed=43)
    assert (s["adapter"]["r"], s["adapter"]["lora_alpha"], s["training"]["seed"]) == (32, 64, 43)
    assert lb.run_settings(cfg, "lr1e-4", "all_in_one")["run_id"].endswith("_lr1e-4_s42")


def test_chrf_counts_empty_predictions():
    recs = [{"pred_gloss": "John-NOM apple-ACC", "gold_gloss": "John-NOM apple-ACC",
             "pred_transliteration": "", "gold_transliteration": "Johni sakwalul"}]
    s = lb.chrf_scores(recs, cfg["chrf"])
    assert s["gloss_chrfpp"] == 100.0 and s["transliteration_chrfpp"] == 0.0
    assert "nc:6" in s["chrf_signature"] and "nw:2" in s["chrf_signature"]


def test_truncation_flag_at_budget_minus_one():
    # all-in-one output that loops and stops at 511 of 512 bytes, no <eos>, no finished Gloss line
    looping = "Hankwuk-ey ka-ss-ni\nGloss: Korea-LOC go-PST-Q" + "-Q" * 230
    assert lb.is_truncated(False, looping[:511], "all_in_one")
    assert lb.is_truncated(False, "Johni sakwalul kkak", "single")          # step 1/2 cut at 255 bytes
    assert not lb.is_truncated(True, "Johni sakwalul", "single")            # ended with <eos>
    assert not lb.is_truncated(False, "Johni sakwalul\n", "single")         # stopped: answer complete
    assert not lb.is_truncated(False, "Johni\nGloss: John-NOM\n", "all_in_one")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
