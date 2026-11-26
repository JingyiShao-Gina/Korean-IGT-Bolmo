"""CPU-only checks for the LoRA data pipeline (no model, no GPU, no download):  python lora/test_lora.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lora_bolmo as lb  # noqa: E402

rz = lb.rz
cfg = lb.load_config()
LANGS = ("ainu", "tsez", "jejueo")
ROW = {"id": "x1", "text": "オラウン チセ ソイ タ", "translation": "When i made it to the front of my home,",
       "transcription": "oraun cise soy ta", "glosses": "then house outside at"}
TRAIN = [{"id": f"t{i}", "text": f"ア イ ウ エ {i}", "translation": f"sentence {i}",
          "transcription": f"a i u e {i}", "glosses": f"1SG-a go-PST and NEG {i}"} for i in range(6)]


def _prompt(lang, strategy, examples=None):
    return lb.prompt_fn(lb.load_templates(lang), strategy, lb.lang_info(lang)["language_name"], examples)


def test_prompts_are_the_zero_shot_prompt_files():
    zs = lb.load_zs_config()
    for lang in LANGS:
        name = lb.lang_info(lang)["language_name"]
        T = lb.load_templates(lang)
        assert set(T) == {"all_in_one", "step_by_step"}
        for k in T:
            assert T[k] == rz.fill_language(rz.load_prompt(zs["prompts"][k]), name)
            assert "{language}" not in T[k] and f"Original {name} script:" in T[k]
    assert "Work step by step." in lb.load_templates("ainu")["step_by_step"]
    assert "Work step by step." not in lb.load_templates("ainu")["all_in_one"]


def test_target_round_trips_through_the_strategy_parser():
    for lang in LANGS:
        for strategy in lb.STRATEGIES:
            [ex] = lb.build_examples([ROW], strategy, _prompt(lang, strategy))
            assert ex["prompt"].endswith("Transliteration:") and ex["kind"] == strategy
            assert ex["target"] == " oraun cise soy ta\nGloss: then house outside at"
            assert lb.PARSERS[strategy](ex["target"]) == (ROW["transcription"], ROW["glosses"], "ok")
            # what a trained model generates (target + stop) is complete under the stop rule
            assert rz.is_complete("all_in_one", ex["target"] + "\n")


def test_answer_fields_are_one_line_and_unusable_rows_are_left_out():
    messy = {**ROW, "id": "m", "transcription": " oraun\ncise  soy ", "glosses": "then\nhouse"}
    [ex] = lb.build_examples([messy], "all_in_one", _prompt("ainu", "all_in_one"))
    assert ex["target"] == " oraun cise soy\nGloss: then house"
    bad = [{**ROW, "id": "a", "glosses": ""}, {**ROW, "id": "b", "transcription": None},
           {**ROW, "id": "c", "text": None}, {**ROW, "id": "d", "translation": "  "}]
    kept, rep = lb.prepare_examples([ROW] + bad, "step_by_step", _prompt("tsez", "step_by_step"), 10_000)
    assert [e["row_id"] for e in kept] == ["x1"] and rep["dropped_empty_field"] == ["a", "b", "c", "d"]
    # a missing input still gives a prompt (for dev/test generation), never a crash
    assert "Original Tsez script: \n" in _prompt("tsez", "all_in_one")({**ROW, "text": None})


def test_too_long_examples_are_left_out_and_reported():
    make = _prompt("jejueo", "all_in_one")
    [ex] = lb.build_examples([ROW], "all_in_one", make)
    n = lb.example_bytes(ex)
    assert n == len(lb.encode_example(ex["prompt"], ex["target"], bos=1, eos=1, offset=4)["ids"])
    kept, rep = lb.prepare_examples([ROW], "all_in_one", make, n)
    assert len(kept) == 1 and rep["dropped_too_long"] == []
    kept, rep = lb.prepare_examples([ROW], "all_in_one", make, n - 1)
    assert kept == [] and rep["dropped_too_long"] == ["x1"] and rep["max_example_bytes"] == n


def test_few_shot_prompt_equals_run_zero_shot():
    ex = rz.select_examples(TRAIN, 3, 42)
    for strategy in lb.STRATEGIES:
        T = lb.load_templates("ainu")
        ours = _prompt("ainu", strategy, ex)(ROW)
        theirs = rz.render(rz.add_examples(T[strategy], rz.examples_block(ex, "Ainu"), "Ainu"),
                           original_script=ROW["text"], translation=ROW["translation"])
        assert ours == theirs and ours.endswith(f"Original Ainu script: {ROW['text']}\n"
                                                f"English translation: {ROW['translation']}\nTransliteration:")
        assert ours.count("\nGloss: ") == 3
    assert _prompt("ainu", "all_in_one", [])(ROW) == _prompt("ainu", "all_in_one")(ROW)


def test_encoding_masks_prompt_and_appends_eos():
    e = lb.encode_example("ab:", " 한", bos=1, eos=1, offset=4)
    assert e["ids"] == [1, 4 + 97, 4 + 98, 4 + 58, 4 + 32, 4 + 0xED, 4 + 0x95, 4 + 0x9C, 1]
    assert e["n_prompt"] == 4 and e["n_target"] == 5
    assert len(e["ids"]) - e["n_prompt"] == e["n_target"]


def test_eval_steps_every_half_epoch():
    total, steps = lb.eval_steps(2079, 3, 16, 0.5)
    assert total == 390 and steps == [65, 130, 195, 260, 325, 390]
    total, steps = lb.eval_steps(704, 3, 16, 0.5)
    assert total == 132 and len(steps) == 6 and steps[-1] == 132


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


def test_run_settings_ids_and_paths():
    s = lb.run_settings(cfg, "baseline", "step_by_step", "ainu")
    assert s["run_id"] == "bolmo-1b_ainu_step_by_step_r16_lr2e-4_s42"
    s = lb.run_settings(cfg, "r32", "all_in_one", "jejueo", seed=43)
    assert (s["adapter"]["r"], s["adapter"]["lora_alpha"], s["training"]["seed"]) == (32, 64, 43)
    assert s["run_id"] == "bolmo-1b_jejueo_all_in_one_r32_lr2e-4_s43"

    class A:
        out_root, smoke = "/x", True
    assert lb.run_path(A, cfg, s) == Path("/x/jejueo/bolmo-1b_jejueo_all_in_one_r32_lr2e-4_s43_smoke")
    assert lb.prediction_name("R", "loss-selected", 390, 3, "test") == "R_loss-selected_step390_3shot_test.csv"
    assert lb.prediction_name("R", "chrf-selected", 65, 0, "test") == "R_chrf-selected_step65_test.csv"
    for bad in (("baseline", "two_stage", "ainu"), ("baseline", "all_in_one", "korean2")):
        try:
            lb.run_settings(cfg, *bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(bad)


def test_cli_rejects_shots_outside_predict():
    try:
        lb.main(["train", "--lang", "ainu", "--strategy", "all_in_one", "--shots", "3"])
    except SystemExit as e:
        assert "only for predict" in str(e)
    else:
        raise AssertionError("--shots accepted for train")


def test_chrf_counts_empty_predictions():
    recs = [{"pred_gloss": "1SG-a go-PST", "gold_gloss": "1SG-a go-PST",
             "pred_transliteration": "", "gold_transliteration": "a i u"}]
    s = lb.chrf_scores(recs, cfg["chrf"])
    assert s["gloss_chrfpp"] == 100.0 and s["transliteration_chrfpp"] == 0.0
    assert "nc:6" in s["chrf_signature"] and "nw:2" in s["chrf_signature"]


def test_truncation_flag():
    looping = "oraun cise\nGloss: then house" + "-Q" * 600
    assert lb.is_truncated(False, looping[:1023], "all_in_one")                 # no <eos>, gloss line unfinished
    assert not lb.is_truncated(True, "oraun cise\nGloss: then house", "all_in_one")   # ended with <eos>
    assert not lb.is_truncated(False, "oraun\nGloss: then\n", "all_in_one")     # stopped: answer complete


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
