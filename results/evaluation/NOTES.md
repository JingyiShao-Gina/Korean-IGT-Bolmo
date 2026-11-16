# Evaluation: zero-shot vs LoRA, and checkpoint selection

Owner: Soyeong. Date: 16 Nov 2026. Bolmo 1B, test = 263 rows, dev = 262 rows.
Code: `evaluation/metrics.py`, `evaluation/evaluate.py` (tests: `evaluation/test_metrics.py`).

Reproduce:
```
pip install sacrebleu==2.6.0 pandas
python evaluation/evaluate.py --lora-dir "<path to the korean-igt-lora Drive folder>"
```

## Metric definitions

- **Morpheme / word accuracy**: Ginn et al. (2024) style. The i-th predicted token is compared with the i-th gold token. Missing tokens count as wrong, and extra tokens are ignored. Score = correct / gold tokens over the whole corpus (micro). Morphemes come from splitting each gloss word on `-` and `=`. `.` (as in `DECL.POL`) is not a boundary.
- **MER / WER**: token-level edit distance / gold tokens. Unlike accuracy, this also penalises extra tokens, so it can exceed 1.
- **chrF++**: sacreBLEU 2.6.0, `nc:6|nw:2|space:no|case:mixed`. This is the same configuration as the LoRA pipeline, and our dev chrF++ reproduces `metrics.csv` exactly on all 18 checkpoints.
- Unparsed or empty predictions count as empty strings. No rows are dropped.

Files: `test_scores.csv` (all test files), `dev_checkpoint_scores.csv` (every checkpoint on dev), `checkpoint_selection.csv`, and `../summary.csv`.

## Table 9.1: main comparison (test)

| System | Morpheme acc | Word acc | MER ↓ | Gloss chrF++ | Translit chrF++ | Parse fail |
|---|---|---|---|---|---|---|
| Zero-shot all-in-one | 5.2 | 1.3 | 0.91 | 19.4 | 1.3 | 1.1% |
| Zero-shot step-by-step | 0.4 | 0.6 | 3.88 | 1.0 | 3.5 | 0% |
| LoRA all-in-one, A (lr 2e-4), loss = chrF++ selected (ep 3) | 46.0 | 37.2 | 0.49 | 52.8 | 52.6 | 0.4% |
| **LoRA two-stage, A (lr 2e-4), loss = chrF++ selected (ep 3)** | **48.6** | **39.4** | **0.44** | **56.4** | **54.2** | 0% |
| LoRA two-stage, B (lr 1e-4), loss-selected (ep 3) | 45.9 | 35.0 | 0.45 | 53.2 | 45.6 | 0% |
| LoRA two-stage, B (lr 1e-4), chrF++-selected (ep 2.5) | 46.2 | 34.6 | 0.45 | 52.6 | 44.5 | 0% |

- **RQ2 (LoRA vs zero-shot):** morpheme accuracy rises from about 0–5% to 46–49%. Zero-shot Bolmo 1B does not produce IGT: it copies the English translation or the Korean source. See `results/zero_shot/NOTES.md`. The all-in-one gloss chrF++ of 19.4 comes from character overlap between the copied English translation and the English gloss words, not from glossing. The step-by-step MER of 3.88 comes from long Korean outputs, where every extra token counts as an insertion.
- **RQ1 (strategy):** after LoRA, two-stage beats all-in-one on every metric (+2.6 morpheme acc, +2.2 word acc, +3.6 chrF++). In zero-shot, both are near 0, so that comparison carries no information.
- **RQ3 (learning rate):** lr 2e-4 beats lr 1e-4 (two-stage). Both runs were still improving at epoch 3.

## Table 9.2: LoRA hyperparameters (validation)

| Experiment | Best val loss ↓ | Best val gloss chrF++ ↑ | Best val MER ↓ | Best val morpheme acc ↑ |
|---|---|---|---|---|
| A: baseline (r16, lr 2e-4), two-stage | 0.513 (ep 3) | 56.64 (ep 3) | 0.421 (ep 2.5) | 50.6 (ep 3) |
| B: lower LR (r16, lr 1e-4), two-stage | 0.632 (ep 3) | 52.82 (ep 2.5) | 0.470 (ep 2.5) | 46.1 (ep 2.5) |
| A: baseline, all-in-one (extra) | 0.530 (ep 3) | 52.75 (ep 3) | 0.519 (ep 2.5) | 45.4 (ep 3) |
| C: r8 / D: r32 | not run | | | |

## Checkpoint selection (tutor's "eval strategy" question)

Validation data never updates the weights. It is only used to choose which saved checkpoint to report. We compare four selection criteria on the same six checkpoints per run (every 0.5 epoch):

| Run | val loss | val chrF++ | val MER | val morpheme acc |
|---|---|---|---|---|
| A two-stage | ep 3 (780) | ep 3 (780) | **ep 2.5 (650)** | ep 3 (780) |
| B two-stage | ep 3 (780) | **ep 2.5 (650)** | **ep 2.5 (650)** | **ep 2.5 (650)** |
| A all-in-one | ep 3 (390) | ep 3 (390) | **ep 2.5 (325)** | ep 3 (390) |

Findings:
1. **Loss vs chrF++ agree in 2 of 3 runs.** They disagree only in B, where loss picks epoch 3 and the three generation metrics (chrF++, MER, morpheme acc) all pick epoch 2.5.
2. **In B, the choice barely matters on test.** The chrF++-selected checkpoint (ep 2.5) is slightly better on morpheme acc (46.2 vs 45.9) and MER, and slightly worse on word acc (34.6 vs 35.0) and chrF++ (52.6 vs 53.2). The dev chrF++ gap that drove the choice was only 0.19 points, so it falls within noise. The chrF++-selected checkpoint even scores lower on *test* chrF++.
3. **MER is the criterion that disagrees most with loss** (it differs from loss in all 3 runs), but **on test it never beats loss selection**:

   | Run | Selected by | Step | Morpheme acc | Word acc | MER ↓ | Gloss chrF++ |
   |---|---|---|---|---|---|---|
   | A two-stage | loss (= chrF++) | 780 | **48.6** | **39.4** | 0.436 | **56.4** |
   | A two-stage | MER | 650 | 46.9 | 38.2 | **0.428** | 55.7 |
   | A all-in-one | loss (= chrF++) | 390 | 46.0 | 37.2 | **0.493** | **52.8** |
   | A all-in-one | MER | 325 | 46.1 | 37.5 | 0.639 | 52.7 |
   | B two-stage | loss | 780 | 45.9 | **35.0** | 0.454 | **53.2** |
   | B two-stage | MER (= chrF++) | 650 | **46.2** | 34.6 | **0.452** | 52.6 |

   For A two-stage, the MER-selected checkpoint is worse on every metric except MER itself. For A all-in-one, it is tied on accuracy and chrF++, but its *test* MER is much worse (0.639 vs 0.493).
4. **Why MER is unreliable for selection:** MER counts every extra predicted morpheme as an error, so a handful of runaway outputs dominate it. On the all-in-one step-325 test file, 3 of 263 sentences produce 113–213 morphemes against 12–16 gold morphemes. Together they contribute about 440 of the 1623 edits. The same effect makes dev MER jump from 0.61 to 0.72 at all-in-one epoch 2 and recover afterwards. Positional morpheme accuracy ignores extra tokens, so it is much more stable, and it agreed with loss in 2 of 3 runs.
5. **Limitation:** loss and chrF++ were still improving at epoch 3 in every run. Then "pick the best" mostly means "pick the last checkpoint", and a 3-epoch budget cannot show where the criteria really diverge (e.g., overfitting). These are single-seed results; the differences between criteria (≤ 1.8 morpheme-accuracy points) are small.

**Conclusion:** for these runs, selecting by validation loss is as good as or better than selecting by a generation metric. chrF++ selection gives the same or a near-identical checkpoint. MER selection is driven by a few degenerate outputs and does not transfer to test.

MER-selected test predictions (A two-stage step 650, A all-in-one step 325) were generated on 16 Nov 2026 with `lora_bolmo.py predict --step N` (Colab T4) and are in the Drive `predictions/` folders as `*_stepN_stepN_test.csv`.
