# LoRA experiments (Bolmo 1B, low-resource languages)

One code for the whole team: each person fine-tunes Bolmo 1B on their own language with `--lang`.
Agenda 24 Nov / 01 Dec: LoRA in the **all-in-one** and **step-by-step** setups, for Ainu (ain), Tsez (ddo) and Jejueo (jje); no hyperparameter tuning for now. The earlier Korean two-stage (two-call) LoRA runs are removed (they are still in the Git history).

| Language | `--lang` | train / dev / test rows | Who |
|---|---|---|---|
| Ainu | `ainu` | 6726 / 218 / 750 | Gina |
| Tsez | `tsez` | 3558 / 445 / 445 | Theo |
| Jejueo | `jejueo` | 704 / 211 / 712 | Soyeong |

Languages come from `configs/zero_shot.json` (`languages`); adding a language there (name + dataset manifest) makes it available here too.

## Runs (per language)

| Order | `--strategy` | `--run` | Settings |
|---|---|---|---|
| 1 | `step_by_step` | `baseline` | r16, alpha 32, lr 2e-4, seed 42 |
| 2 | `all_in_one` | `baseline` | same |

`r8`, `r32` and `lr1e-4` are still in `configs/lora.json` (one factor at a time) but not planned now. Fixed for all runs: dropout 0.05, 3 epochs, effective batch size 16 (1 x 16 gradient accumulation), AdamW, weight decay 0.01, linear schedule with 3% warmup, gradient clipping 1.0, gradient checkpointing.

Rough time per run on a T4 (scales with the number of train rows; the earlier Korean run took about 2 h for 2079 rows): Jejueo about 45 min, Tsez about 3.5 h, Ainu about 6-7 h. Ainu: use an L4 / A100 if you can, otherwise let it resume over several sessions.

## How to run

Use `lora/lora_bolmo.ipynb` on Colab: set `LANG` and `STRATEGY` in section 3 and run the cells in order. Or from the repository root:

```bash
python lora/check_data.py --lang ainu --shots 3                                   # data check, no GPU
python lora/lora_bolmo.py train   --lang ainu --strategy step_by_step --smoke     # pipeline check, not a result
python lora/lora_bolmo.py train   --lang ainu --strategy step_by_step             # resumable
python lora/lora_bolmo.py select  --lang ainu --strategy step_by_step
python lora/lora_bolmo.py predict --lang ainu --strategy step_by_step --split test             # zero-shot prompt (main result)
python lora/lora_bolmo.py predict --lang ainu --strategy step_by_step --split test --shots 3   # few-shot prompt (extra)
python lora/lora_bolmo.py export  --lang ainu --strategy step_by_step
```

## Setup and why

- **Same prompts as zero-shot and few-shot.** Training and the main test predictions use the zero-shot prompt files `zero_shot/prompts/all_in_one.txt` and `zero_shot/prompts/step_by_step.txt`, with `{language}` filled in by the same function as the zero-shot runner. Both strategies are **one call** and end at `Transliteration:`; they differ only in the instructions ("Work step by step. Step 1 ... Step 2 ..."). So zero-shot, few-shot and LoRA differ only in the examples and the weights.
- **Target.** ` <transliteration>\nGloss: <gloss>` + `<eos>`, exactly what the zero-shot parsers read back (`parse_all_in_one` / `parse_step_by_step`). Loss only on the answer; prompt bytes are not trained on. The `<eos>` teaches the model to stop. Gold fields are put on one line each (some upstream rows contain line breaks).
- **Few-shot prompt at test time (`--shots K`).** The adapter is trained without examples. `predict --shots K` builds the few-shot prompt with the zero-shot runner's own functions (`select_examples`, `add_examples`; same seed and filters), so the examples are the ones of the few-shot experiment (checked: the Ainu 3-shot step-by-step prompt has the same SHA-256 as in `results/few_shot/ainu/`). This is a train/test prompt mismatch, so it is an extra comparison, not the main result. Per-row example selection (e.g. word overlap, agenda 01 Dec) can plug into `prompt_fn` once it exists in the zero-shot runner.
- **Data that is left out (never truncated).** Rows with an empty script, translation, transliteration or gloss, and examples longer than `max_seq_bytes` (2048) are left out of training and validation loss; their ids are in `run_meta.json`. Generation (chrF++, test) uses every dev/test row. `check_data.py` (26 Nov): 2 of 6726 Ainu and 2 of 3558 Tsez train rows are too long; 1 Tsez train row and 1 dev row have an empty field.
- **Where LoRA goes.** Bolmo-1B is byte-level: a 1-layer xLSTM byte encoder, a 16-layer global transformer over patches, and a 4-layer xLSTM byte decoder. LoRA is on `q_proj` / `v_proj` of the 16 global layers (32 modules, regex in the config). The runner fails if nothing matches and writes the matched names and trainable-parameter count to `run_meta.json`.
- **Byte-level loss.** Bolmo predicts 520 classes: each byte twice, once as "byte" and once as "byte, then patch boundary". The training target is the next byte, in its boundary form wherever Bolmo's own boundary predictor places a boundary - the patching it uses at inference.
- **Padding for xLSTM.** In training mode the xLSTM kernels need a sequence length divisible by 64, so each sequence is right-padded. All layers are causal, so this does not change any scored position.
- **Validation** every 0.5 epoch (six checkpoints): teacher-forced loss over the dev answers, and SacreBLEU chrF++ (char order 6, word order 2, beta 2) of the generated **gloss and transliteration** on all dev rows (agenda 24 Nov: evaluate both). Checkpoint selection: lowest validation loss and highest validation gloss chrF++ (ties -> earlier step).
- **Generation** is greedy, batched (8, halved automatically on out-of-memory), budget 1024 bytes, stopping at `<eos>` or once the gloss line is finished. The longest gold answer is 819 bytes (Ainu test).
- **Precision.** float32 on T4 (no native bf16), bf16 on L4 / A100. Recorded in `run_meta.json`.
- **Resume.** A quick resume save (`resume/`, overwritten every 20 optimizer steps) limits what a Colab disconnect costs. On restart, `train` first evaluates any checkpoint whose evaluation was cut off, then continues from the newest save.

## Outputs

Each run writes to `MyDrive/igt-lora/<lang>/<run_id>/`, for example `ainu/bolmo-1b_ainu_step_by_step_r16_lr2e-4_s42/`:

| File | Content |
|---|---|
| `run_meta.json` | All settings, language, prompt file + hash, matched LoRA modules, parameter counts, rows left out, dataset manifest, git commit, package versions, GPU. |
| `train_log.csv` | Per optimizer step: train loss, learning rate, gradient norm, elapsed time. |
| `metrics.csv` | Per checkpoint: `val_loss`, `val_gloss_chrfpp`, `val_transliteration_chrfpp`, parse-failure and truncation rates. |
| `dev_predictions/step-NNNNNN.csv` | Dev predictions behind each chrF++ value. |
| `checkpoints/step-NNNNNN/` | LoRA adapter + optimizer state (resume point). Not committed. |
| `selection.json` | Checkpoint selected by min validation loss and by max validation gloss chrF++. |
| `predictions/*_test.csv` | Test predictions of both selected checkpoints, same columns as the zero-shot CSVs; `*_3shot_test.csv` = with the 3-shot prompt. |

`export` copies everything except checkpoints and dev predictions into `results/lora/<lang>/<run_id>/`. `evaluation/evaluate.py --lora-dir <MyDrive/igt-lora>` scores all languages, zero-shot and LoRA, into one table.

## Files

| File | Purpose |
|---|---|
| `lora/lora_bolmo.py` | Runner: `train` (resumable), `eval`, `select`, `predict` (`--shots K`), `export`. |
| `lora/lora_bolmo.ipynb` | Colab notebook. |
| `lora/check_data.py` | Data check per language (rows left out, longest example / answer / few-shot prompt). |
| `lora/test_lora.py` | CPU checks of prompts, targets, parsing, few-shot prompt, run ids (`python lora/test_lora.py`). |
| `lora/smoke_tiny_bolmo.py` | Optional end-to-end check on a tiny random Bolmo (needs torch, transformers 4.57.3, xlstm, peft). |
| `configs/lora.json` | Model revision, LoRA / training settings, runs, chrF++ settings, generation budget. |

## Open points

- 20+ few-shot examples (agenda 01 Dec) give prompts of 5-7 KB (`check_data.py --shots 20`); check that Bolmo handles that length before using it with `--shots`.
