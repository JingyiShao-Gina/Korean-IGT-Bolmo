"""Score every zero-shot and LoRA prediction file and build the result tables.

Usage:
  python evaluation/evaluate.py --lora-dir "<path to korean-igt-lora>"

Inputs
  results/zero_shot/bolmo-1b_<strategy>_test.csv          (Gina, in Git)
  <lora-dir>/<run>/metrics.csv                              (val loss per checkpoint)
  <lora-dir>/<run>/predictions/*_test.csv                   (selected checkpoints, test)
  <lora-dir>/<run>/dev_predictions/step-XXXXXX.csv          (every checkpoint, dev)

Outputs (in --out, default results/evaluation/)
  test_scores.csv             one row per test prediction file
  dev_checkpoint_scores.csv   one row per (run, checkpoint) on dev
  checkpoint_selection.csv    checkpoint chosen by each criterion, with its test score
  ../summary.csv              filled-in project summary table
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import evaluate  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RUN_RE = re.compile(r"bolmo-1b_(?P<strategy>all_in_one|two_stage)_r(?P<r>\d+)_lr(?P<lr>[\de.-]+)_s(?P<seed>\d+)$")
# "<run>_loss-selected_step780_test.csv", or "<run>_step650_step650_test.csv" from `predict --step`
TEST_RE = re.compile(r"_(?:(?P<sel>loss|chrf)-selected|step\d+)_step(?P<step>\d+)_test\.csv$")

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


def score_file(path: Path) -> dict:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    res = evaluate(df["pred_gloss"].tolist(), df["gold_gloss"].tolist(),
                   df["pred_transliteration"].tolist(), df["gold_transliteration"].tolist())
    res["parse_failure_rate"] = (df["parse_status"] != "ok").mean()
    res["truncation_rate"] = (df["truncated"].str.lower() == "true").mean()
    return res


def lora_runs(lora_dir: Path):
    for d in sorted(lora_dir.iterdir()):
        m = RUN_RE.match(d.name)
        if d.is_dir() and m and (d / "metrics.csv").exists():
            yield d, m.groupdict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-dir", type=Path, required=True)
    ap.add_argument("--zero-shot-dir", type=Path, default=REPO / "results/zero_shot")
    ap.add_argument("--out", type=Path, default=REPO / "results/evaluation")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- test files
    test_rows = []
    for f in sorted(args.zero_shot_dir.glob("bolmo-1b_*_test.csv")):
        strategy = f.name.removeprefix("bolmo-1b_").removesuffix("_test.csv")
        test_rows.append({"system": f"Zero-shot {strategy}", "run_id": f.stem,
                          "strategy": strategy, "adaptation": "none", "seed": "",
                          "selection": "", "step": "", "file": f.name, **score_file(f)})

    dev_rows = []
    for run_dir, info in lora_runs(args.lora_dir):
        run = run_dir.name
        for f in sorted((run_dir / "predictions").glob("*_test.csv")):
            m = TEST_RE.search(f.name)
            test_rows.append({"system": f"LoRA {info['strategy']} lr{info['lr']} r{info['r']}",
                              "run_id": run, "strategy": info["strategy"], "adaptation": "lora",
                              "seed": info["seed"], "selection": m["sel"] or "manual",
                              "step": int(m["step"]),
                              "file": f.name, **score_file(f)})

        # ------------------------------------------------------------ dev checkpoints
        theo = pd.read_csv(run_dir / "metrics.csv")
        for f in sorted((run_dir / "dev_predictions").glob("step-*.csv")):
            step = int(f.stem.split("-")[1])
            t = theo.loc[theo["step"] == step].iloc[0]
            dev_rows.append({"run_id": run, "strategy": info["strategy"], "lr": info["lr"],
                             "r": info["r"], "step": step, "epoch": round(t["epoch"], 2),
                             "val_loss": t["val_loss"],
                             "theo_val_gloss_chrfpp": t["val_gloss_chrfpp"],
                             **score_file(f)})

    test = pd.DataFrame(test_rows)
    dev = pd.DataFrame(dev_rows)

    # sanity check: our chrF++ must reproduce the LoRA pipeline's numbers
    diff = (dev["gloss_chrfpp"] - dev["theo_val_gloss_chrfpp"]).abs().max()
    print(f"max |our dev chrF++ - metrics.csv chrF++| = {diff:.4f}")

    # ---------------------------------------------------------------- selection
    sel_rows = []
    for run, g in dev.groupby("run_id", sort=True):
        g = g.sort_values("step")  # ties -> earlier step
        for crit, (col, how) in CRITERIA.items():
            best = g.loc[g[col].idxmin() if how == "min" else g[col].idxmax()]
            hit = test[(test["run_id"] == run) & (test["step"] == best["step"])]
            row = {"run_id": run, "criterion": crit, "selected_step": int(best["step"]),
                   "selected_epoch": best["epoch"], "dev_value": best[col],
                   "test_predictions_available": not hit.empty}
            for c in ["gloss_morpheme_accuracy", "gloss_word_accuracy", "gloss_mer", "gloss_chrfpp"]:
                row[f"test_{c}"] = hit.iloc[0][c] if not hit.empty else None
            sel_rows.append(row)
    sel = pd.DataFrame(sel_rows)

    # ---------------------------------------------------------------- write
    front = ["system", "run_id", "selection", "step"]
    test[front + METRIC_COLS + [c for c in test.columns if c not in front + METRIC_COLS]] \
        .to_csv(args.out / "test_scores.csv", index=False, float_format="%.4f")
    dev.to_csv(args.out / "dev_checkpoint_scores.csv", index=False, float_format="%.4f")
    sel.to_csv(args.out / "checkpoint_selection.csv", index=False, float_format="%.4f")

    summary = pd.DataFrame({
        "run_id": test["run_id"], "model_size": "1b", "strategy": test["strategy"],
        "adaptation": test["adaptation"], "seed": test["seed"],
        "selection_metric": test["selection"], "checkpoint": test["step"],
        **{c: test[c] for c in ["gloss_word_accuracy", "gloss_morpheme_accuracy", "gloss_mer",
                                "gloss_chrfpp", "transliteration_chrfpp",
                                "parse_failure_rate", "truncation_rate"]}})
    summary.to_csv(REPO / "results/summary.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 200)
    print("\nTEST\n", test[["system", "selection", "step", "gloss_morpheme_accuracy",
                            "gloss_word_accuracy", "gloss_mer", "gloss_chrfpp",
                            "transliteration_chrfpp", "parse_failure_rate"]].round(4).to_string())
    print("\nDEV\n", dev[["run_id", "step", "val_loss", "gloss_chrfpp", "gloss_mer",
                          "gloss_morpheme_accuracy", "gloss_word_accuracy"]].round(4).to_string())
    print("\nSELECTION\n", sel.round(4).to_string())


if __name__ == "__main__":
    main()
