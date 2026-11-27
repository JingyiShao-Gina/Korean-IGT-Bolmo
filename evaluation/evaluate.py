"""Score every zero-shot and LoRA prediction file and build the result tables.

Usage:
  python evaluation/evaluate.py --lora-dir "<path to igt-lora>"

Inputs
  results/zero_shot/[<lang>/]bolmo-1b_<strategy>_test.csv  (zero-shot; files directly in zero_shot/ are Korean.
                                                             The strategy is read from the file: the Korean v3
                                                             "step_by_step" file is two calls, so two_stage)
  <lora-dir>/<lang>/<run>/metrics.csv                       (val loss per checkpoint)
  <lora-dir>/<lang>/<run>/predictions/*_test.csv            (selected checkpoints, test; *_<K>shot_test.csv
                                                             = same adapter tested with the K-shot prompt)
  <lora-dir>/<lang>/<run>/dev_predictions/step-XXXXXX.csv   (every checkpoint, dev; optional: without
                                                             it the dev and selection tables are skipped)
  (runs directly in <lora-dir>/ are the earlier Korean runs and still work)

Outputs (in --out, default results/evaluation/)
  test_scores.csv             one row per test prediction file
  dev_checkpoint_scores.csv   one row per (run, checkpoint) on dev
  checkpoint_selection.csv    checkpoint chosen by each criterion, with its test score
  ../summary.csv              filled-in project summary table

If a language's manifest sets "gloss_affixes_detached" (Jejueo: "eat -PST -DECL"), gold and
predicted glosses are re-joined (metrics.join_detached_affixes) before scoring, as in
zero_shot/score_zero_shot.py, so the scores match results/zero_shot/<lang>/scores.csv.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import evaluate, join_detached_affixes  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
# "bolmo-1b_ainu_step_by_step_r16_lr2e-4_s42" (per language) or "bolmo-1b_two_stage_r16_lr2e-4_s42" (earlier Korean runs)
RUN_RE = re.compile(r"bolmo-1b_(?:(?P<lang>[a-z]+)_)?(?P<strategy>all_in_one|step_by_step|two_stage)"
                    r"_r(?P<r>\d+)_lr(?P<lr>[\de.-]+)_s(?P<seed>\d+)$")
# "<run>_loss-selected_step780[_3shot]_test.csv", or "<run>_step650_step650_test.csv" from `predict --step`
TEST_RE = re.compile(r"_(?:(?P<sel>loss|chrf)-selected|step\d+)_step(?P<step>\d+)(?:_(?P<shots>\d+)shot)?_test\.csv$")

# Selection criteria on dev: column, direction ("min"/"max")
CRITERIA = {
    "val_loss": ("val_loss", "min"),
    "val_gloss_chrfpp": ("gloss_chrfpp", "max"),
    "val_gloss_mer": ("gloss_mer", "min"),
    "val_gloss_morpheme_accuracy": ("gloss_morpheme_accuracy", "max"),
}
METRIC_COLS = ["gloss_morpheme_accuracy", "gloss_word_accuracy", "gloss_mer",
               "gloss_wer", "gloss_chrfpp", "transliteration_chrfpp",
               "gloss_morpheme_accuracy_sent_avg", "gloss_word_accuracy_sent_avg",
               "transliteration_word_accuracy"]


def affixes_detached(lang: str) -> bool:
    """True if the language's manifest says its glosses write affixes as separate tokens."""
    cfg = json.loads((REPO / "configs/zero_shot.json").read_text(encoding="utf-8"))
    entry = cfg["languages"].get(lang)
    if entry is None:
        return False
    manifest = json.loads((REPO / entry["dataset_manifest"]).read_text(encoding="utf-8"))
    return bool(manifest.get("gloss_affixes_detached", False))


def zero_shot_strategy(path: Path) -> tuple[str, str]:
    """(strategy, prompt_version) of a zero-shot prediction file, from its content, not its name.

    The Korean v3 file results/zero_shot/bolmo-1b_step_by_step_test.csv is named after the old
    meaning of "step-by-step" (two calls); today that setting is two_stage, and step_by_step is
    one call. A two-call file has a raw_output_step2 column, so it is labelled two_stage.
    """
    name = path.name.removeprefix("bolmo-1b_").removesuffix("_test.csv")
    columns = pd.read_csv(path, nrows=0).columns
    strategy = "two_stage" if "raw_output_step2" in columns else name
    meta = path.with_suffix(".meta.json")
    version = json.loads(meta.read_text(encoding="utf-8")).get("prompt_version", "") if meta.exists() else ""
    return strategy, version


def lora_budgets(generation: dict) -> dict:
    """Byte budget per n_new column, from the LoRA pipeline's generation config."""
    m = generation["max_new_bytes"]
    if isinstance(m, int):  # per-language runs: one call per row, one budget
        return {"n_new_tokens": m}
    return {"n_new_tokens": m["all_in_one"],
            "n_new_tokens_step1": m["transliteration"],
            "n_new_tokens_step2": m["gloss"]}


def score_file(path: Path, budgets: dict | None = None, join_affixes: bool = False) -> dict:
    """Score one prediction CSV.

    truncation_rate: for LoRA files (budgets given), a row counts as truncated when
    any generation step reached its byte budget. The LoRA pipeline stops at
    budget - 1 bytes while its own `truncated` flag needs >= budget, so that flag
    is always False; it is kept as truncation_rate_reported. LoRA generation stops
    as soon as the answer is complete, so reaching the budget means it was cut off.
    Zero-shot generation does not stop early (Bolmo ignores the stopping criteria),
    so reaching the budget says nothing there: zero-shot keeps its own flag.
    join_affixes: re-attach detached affixes in gold and predicted glosses first (Jejueo).
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if join_affixes:
        df["gold_gloss"] = df["gold_gloss"].map(join_detached_affixes)
        df["pred_gloss"] = df["pred_gloss"].map(join_detached_affixes)
    res = evaluate(df["pred_gloss"].tolist(), df["gold_gloss"].tolist(),
                   df["pred_transliteration"].tolist(), df["gold_transliteration"].tolist())
    res["parse_failure_rate"] = (df["parse_status"] != "ok").mean()
    reported = (df["truncated"].str.lower() == "true")
    res["truncation_rate_reported"] = reported.mean()
    if budgets:
        hit = pd.Series(False, index=df.index)
        for col, budget in budgets.items():
            if col in df:
                hit |= df[col].astype(int) >= budget - 1
        res["truncation_rate"] = hit.mean()
    else:
        res["truncation_rate"] = reported.mean()
    return res


def lora_runs(lora_dir: Path):
    """Runs in <lora-dir>/<lang>/<run> and (earlier Korean runs) <lora-dir>/<run>."""
    for d in sorted([*lora_dir.iterdir(), *lora_dir.glob("*/*")]):
        m = RUN_RE.match(d.name)
        if d.is_dir() and m and (d / "metrics.csv").exists():
            info = m.groupdict()
            info["lang"] = info["lang"] or "korean"
            yield d, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-dir", type=Path, required=True)
    ap.add_argument("--zero-shot-dir", type=Path, default=REPO / "results/zero_shot")
    ap.add_argument("--out", type=Path, default=REPO / "results/evaluation")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- test files
    test_rows = []
    for f in sorted([*args.zero_shot_dir.glob("bolmo-1b_*_test.csv"), *args.zero_shot_dir.glob("*/bolmo-1b_*_test.csv")]):
        strategy, version = zero_shot_strategy(f)
        lang = "korean" if f.parent == args.zero_shot_dir else f.parent.name
        test_rows.append({"system": f"Zero-shot {strategy} (prompt {version})" if version else f"Zero-shot {strategy}",
                          "language": lang, "run_id": f.stem, "prompt_version": version,
                          "strategy": strategy, "adaptation": "none", "shots": 0, "seed": "",
                          "selection": "", "step": "", "file": f.name,
                          **score_file(f, join_affixes=affixes_detached(lang))})

    dev_rows = []
    for run_dir, info in lora_runs(args.lora_dir):
        run = run_dir.name
        budgets = lora_budgets(json.loads((run_dir / "run_meta.json").read_text())["generation"])
        join = affixes_detached(info["lang"])
        for f in sorted((run_dir / "predictions").glob("*_test.csv")):
            m = TEST_RE.search(f.name)
            shots = int(m["shots"] or 0)
            test_rows.append({"system": f"LoRA {info['strategy']} lr{info['lr']} r{info['r']}"
                                        + (f" + {shots}-shot prompt" if shots else ""),
                              "language": info["lang"], "run_id": run, "strategy": info["strategy"],
                              "adaptation": "lora", "shots": shots,
                              "seed": info["seed"], "selection": m["sel"] or "manual",
                              "step": int(m["step"]),
                              "file": f.name, **score_file(f, budgets, join)})

        # ------------------------------------------------------------ dev checkpoints
        theo = pd.read_csv(run_dir / "metrics.csv")
        for f in sorted((run_dir / "dev_predictions").glob("step-*.csv")):
            step = int(f.stem.split("-")[1])
            t = theo.loc[theo["step"] == step].iloc[0]
            dev_rows.append({"run_id": run, "language": info["lang"], "strategy": info["strategy"], "lr": info["lr"],
                             "r": info["r"], "step": step, "epoch": round(t["epoch"], 2),
                             "val_loss": t["val_loss"],
                             "theo_val_gloss_chrfpp": t["val_gloss_chrfpp"],
                             **score_file(f, budgets, join)})

    test = pd.DataFrame(test_rows)
    dev = pd.DataFrame(dev_rows)

    if dev.empty:
        print("No dev_predictions/ found: skipping the dev checkpoint and selection tables "
              "(test scores are still written).")
    else:
        # sanity check: our chrF++ must reproduce the LoRA pipeline's numbers
        diff = (dev["gloss_chrfpp"] - dev["theo_val_gloss_chrfpp"]).abs().max()
        print(f"max |our dev chrF++ - metrics.csv chrF++| = {diff:.4f}")

    # ---------------------------------------------------------------- selection
    sel_rows = []
    for run, g in (dev.groupby("run_id", sort=True) if not dev.empty else []):
        g = g.sort_values("step")  # ties -> earlier step
        for crit, (col, how) in CRITERIA.items():
            best = g.loc[g[col].idxmin() if how == "min" else g[col].idxmax()]
            hit = test[(test["run_id"] == run) & (test["step"] == best["step"]) & (test["shots"] == 0)]
            row = {"run_id": run, "criterion": crit, "selected_step": int(best["step"]),
                   "selected_epoch": best["epoch"], "dev_value": best[col],
                   "test_predictions_available": not hit.empty}
            for c in ["gloss_morpheme_accuracy", "gloss_word_accuracy", "gloss_mer", "gloss_chrfpp"]:
                row[f"test_{c}"] = hit.iloc[0][c] if not hit.empty else None
            sel_rows.append(row)
    sel = pd.DataFrame(sel_rows)

    # ---------------------------------------------------------------- write
    front = ["system", "language", "run_id", "selection", "step"]
    test[front + METRIC_COLS + [c for c in test.columns if c not in front + METRIC_COLS]] \
        .to_csv(args.out / "test_scores.csv", index=False, float_format="%.4f")
    if not dev.empty:
        dev.to_csv(args.out / "dev_checkpoint_scores.csv", index=False, float_format="%.4f")
        sel.to_csv(args.out / "checkpoint_selection.csv", index=False, float_format="%.4f")

    summary = pd.DataFrame({
        "run_id": test["run_id"], "language": test["language"], "model_size": "1b", "strategy": test["strategy"],
        "adaptation": test["adaptation"], "shots": test["shots"], "seed": test["seed"],
        "selection_metric": test["selection"], "checkpoint": test["step"],
        **{c: test[c] for c in ["gloss_word_accuracy", "gloss_morpheme_accuracy", "gloss_mer",
                                "gloss_chrfpp", "transliteration_chrfpp",
                                "parse_failure_rate", "truncation_rate"]}})
    summary.to_csv(REPO / "results/summary.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 200)
    print("\nTEST\n", test[["system", "language", "selection", "step", "gloss_morpheme_accuracy",
                            "gloss_word_accuracy", "gloss_mer", "gloss_chrfpp",
                            "transliteration_chrfpp", "parse_failure_rate"]].round(4).to_string())
    if dev.empty:
        return
    print("\nDEV\n", dev[["run_id", "step", "val_loss", "gloss_chrfpp", "gloss_mer",
                          "gloss_morpheme_accuracy", "gloss_word_accuracy"]].round(4).to_string())
    print("\nSELECTION\n", sel.round(4).to_string())


if __name__ == "__main__":
    main()
