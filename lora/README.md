# LoRA experiments (Bolmo 1B)

Owner: Theo. Agenda 17 Nov: LoRA fine-tuning of Bolmo 1B in two settings, **all-in-one** and **two-stage** (transliteration -> gloss), rank / learning-rate comparisons, and validation loss + chrF++ for every checkpoint (Soyeong compares the two selection rules). 1B only; the 7B rows in the protocol are out of scope for now.

**Status:** code ready and tested on CPU with a tiny random Bolmo. No real training run yet; the first step on Colab is the smoke test (notebook section 4).

## What each file does

| File | Purpose |
|---|---|
| `lora/lora_bolmo.py` | Runner. Commands: `train` (resumable), `eval` (fill in any checkpoint without metrics), `select`, `predict`, `export`. |
| `lora/lora_bolmo.ipynb` | Colab notebook: Drive, install, choose run, smoke test, train, loss-vs-chrF++ plot, select + test predictions, export. |
| `configs/lora.json` | Model revision, LoRA/training settings, the five runs, chrF++ settings, generation budgets. |
| `lora/test_lora.py` | CPU-only checks of data building, encoding, eval schedule, selection, chrF++ (`python lora/test_lora.py`). |
| `lora/smoke_tiny_bolmo.py` | CPU end-to-end check with a tiny random Bolmo built from the real model code (no weights download). |

## Runs

One factor changed at a time against the baseline; each run is done for both strategies (10 runs in total).

| Run (`--run`) | LoRA rank | alpha | Learning rate | Source |
|---|---|---|---|---|
| `baseline` | 16 | 32 | 2e-4 | Protocol V4 section 4 |
| `r8` | 8 | 16 | 2e-4 | rank pilot |
| `r32` | 32 | 64 | 2e-4 | rank pilot |
| `lr1e-4` | 16 | 32 | 1e-4 | LR pilot (V4 section 7A, LR-1) |
| `lr5e-4` | 16 | 32 | 5e-4 | LR pilot (V4 section 7A, LR-3) |

Fixed for all runs: dropout 0.05, 3 epochs, effective batch size 16 (1 x 16 gradient accumulation), AdamW, weight decay 0.01, linear schedule with 3% warmup, gradient clipping 1.0, seed 42, gradient checkpointing. Repeat seeds 43 / 44 (`--seed`) only if time allows.

Checkpoints are saved and evaluated **every 0.5 epoch**: all-in-one 2079 examples -> 390 optimizer steps, checkpoints at 65, 130, ..., 390; two-stage 4158 examples (2 per row) -> 780 steps, checkpoints at 130, 260, ..., 780. Six checkpoints per run.

Between checkpoints, a **quick resume save** (`resume/`, adapter + optimizer state, overwritten every 20 optimizer steps) limits what a Colab disconnect costs to at most 20 steps. It is never evaluated and is not a selection candidate. On restart, `train` first evaluates any checkpoint whose evaluation was cut off, then continues from the newest of the last checkpoint and the quick save. Tested: a run resumed this way ends with exactly the same adapter as an uninterrupted run.

## Training setup and why

- **Same prompts as zero-shot.** Inputs use the frozen v3 templates in `zero_shot/prompts/` (script + translation), so LoRA and zero-shot differ only in the weights. The target is exactly what the zero-shot parser reads back: all-in-one ` <transliteration>\nGloss: <gloss>`, two-stage ` <transliteration>` and ` <gloss>`.
- **Loss only on the answer.** Each example is `<bos> prompt answer <eos>`; prompt bytes are not trained on. The `<eos>` teaches the model to stop, which the zero-shot model never did (it kept generating, which made Gina's runs slow).
- **Two-stage = one adapter, two tasks** (Protocol V4 section 3): (script + translation -> transliteration) and (script + translation + **gold** transliteration -> gloss). Validation chrF++ and test predictions always chain the model's **predicted** transliteration into stage 2. This teacher-forcing / exposure mismatch is a known limitation to mention in the report.
- **Where LoRA goes.** Bolmo-1B is byte-level: a 1-layer xLSTM byte encoder, a 16-layer global transformer over patches, and a 4-layer xLSTM byte decoder. LoRA is on `q_proj` / `v_proj` of the 16 global layers (32 modules, regex in the config). The runner fails if nothing matches and writes the matched names and trainable-parameter count to `run_meta.json`.
- **Byte-level loss.** Bolmo predicts 520 classes: each byte twice, once as "byte" and once as "byte, then patch boundary". The training target is the next byte, in its boundary form wherever Bolmo's own boundary predictor places a boundary - the patching it uses at inference.
- **Padding for xLSTM.** In training mode the xLSTM kernels need a sequence length divisible by 64, so each sequence is right-padded. All layers are causal, so this does not change any scored position (tested).
- **Validation loss** = teacher-forced token-level mean cross-entropy over all dev answer bytes, no gradients. For two-stage it is over both tasks; `metrics.csv` also lists the transliteration and gloss parts.
- **Validation chrF++** = SacreBLEU chrF++ (char order 6, word order 2, beta 2, case kept, whitespace off) of generated glosses on all 262 dev rows; transliteration chrF++ is reported separately. Unparseable outputs count as empty strings.
- **Generation** is greedy, batched (8, halved automatically on out-of-memory), budget 512 bytes all-in-one / 256 per stage, stopping at `<eos>` or at the end of the answer line(s).
- **Precision.** float32 on T4 (no native bf16), bf16 on L4 / A100. Recorded in `run_meta.json`.

## Outputs

Each run writes to `MyDrive/korean-igt-lora/<run_id>/`, for example `bolmo-1b_all_in_one_r16_lr2e-4_s42/`:

| File | Content |
|---|---|
| `run_meta.json` | All settings, matched LoRA modules, parameter counts, dataset manifest, prompt hashes, git commit, package versions, GPU. |
| `train_log.csv` | Per optimizer step: train loss, learning rate, gradient norm, elapsed time. |
| `metrics.csv` | **Per checkpoint: `val_loss`, `val_gloss_chrfpp`, `val_transliteration_chrfpp`**, two-stage loss parts, parse-failure and truncation rates, evaluation time. |
| `dev_predictions/step-NNNNNN.csv` | Dev predictions behind each chrF++ value. |
| `checkpoints/step-NNNNNN/` | LoRA adapter + optimizer state (resume point). Not committed. |
| `resume/` | Quick resume save every 20 steps (overwritten). Not committed. |
| `selection.json` | Checkpoint selected by min validation loss and by max validation gloss chrF++. |
| `predictions/*_test.csv` | Test predictions of both selected checkpoints. Same columns as the zero-shot CSVs; two-stage uses the step-by-step columns. |

`export` copies everything except checkpoints and dev predictions into `results/lora/<run_id>/` for Git and for Soyeong's evaluation.

## Not done yet / to check on the first GPU run

- Runtime and memory on a T4 are unknown until the smoke test and the first checkpoint. If a full run takes too long, the 0.5-epoch dev generation is the expensive part; report any change of the evaluation schedule.
- The real 1B model has not been loaded locally; the CPU tests use the same model code with a tiny random configuration.
