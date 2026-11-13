# Zero-shot experiments (Bolmo 1B)

Owner: Gina. Agenda 17 Nov: complete zero-shot with Bolmo 1B in two settings, **all-in-one** and **step-by-step** (transliteration → gloss). **Status: done.** Test predictions are in `results/zero_shot/`, and accuracy evaluation is next (Soyeong).

- Input: original Korean script (`text`) + English translation (`translation`).
- Output: predicted transliteration + morpheme-level gloss.
- Gold targets (evaluation only, never in prompts): `transcription` (Yale romanization, not morpheme-segmented) and `glosses`.
- Data: `Soyyeongg/Korean_igt_cleaned`, revision pinned in `data/manifest.json`; test split = 263 rows.

## What each file does

| File | Purpose |
|---|---|
| `zero_shot/prompts/all_in_one.txt` | All-in-one prompt: one call that returns `Transliteration:` and `Gloss:` lines. |
| `zero_shot/prompts/transliteration.txt` | Step-by-step, step 1: predict the transliteration. |
| `zero_shot/prompts/gloss.txt` | Step-by-step, step 2: predict the gloss from the source, the translation and the **model's own** step-1 transliteration (never gold). |
| `configs/zero_shot.json` | Run settings: model ID and revision, prompt version, which prompt file belongs to which step, input/gold field names, greedy decoding, 768-token budget. |
| `zero_shot/run_zero_shot.py` | The runner. It loads data and model, fills in the prompts, generates, parses the answers, and writes a CSV plus a `.meta.json` for each setting. CLI: `--split dev/test`, `--limit N`, `--strategy all_in_one/step_by_step/both`. |
| `zero_shot/zero_shot_bolmo.ipynb` | Colab notebook: install, prompt preview, 10-row dev smoke test, full test run, download. |
| `zero_shot/test_parsing.py` | CPU-only checks for prompt filling and answer parsing (`python zero_shot/test_parsing.py`). |
| `results/zero_shot/bolmo-1b_all_in_one_test.csv` | All-in-one predictions on the test split (263 rows). |
| `results/zero_shot/bolmo-1b_step_by_step_test.csv` | Step-by-step predictions on the test split (263 rows). |
| `results/zero_shot/*.meta.json` | Record of each run: model/dataset revision, prompt version and SHA-256, git commit, package versions, GPU, runtime, parse/truncation counts. |
| `results/zero_shot/NOTES.md` | Run notes: results at a glance, model behaviour for error analysis, problems met, notes for evaluation. |

## Prediction CSV columns

| Column | Meaning |
|---|---|
| `id`, `text`, `translation` | Dataset row ID, Korean source, English translation (model input). |
| `gold_transliteration`, `gold_gloss` | Reference answers from the dataset. |
| `pred_transliteration`, `pred_gloss` | Parsed model predictions; use these for evaluation. |
| `parse_status` | `ok`, `missing_gloss`, `empty_gloss` or `empty_transliteration`. `ok` = parseable, not correct. |
| `truncated` | The answer was incomplete when the generation budget ran out (usually a repetition loop). |
| `raw_output` (all-in-one) / `raw_output_step1`, `raw_output_step2` (step-by-step) | Unedited model output. |
| `n_new_tokens*` | Number of generated ids per call. |
| `truncated_step1`, `truncated_step2` | Step-by-step only: which step ran out of budget. |

Rows are never repaired, retried or dropped. Unusable predictions are scored as empty.

## How to rerun (Colab, T4 GPU)

```python
!pip install "transformers==4.57.3" "xlstm==2.0.4" datasets pandas accelerate
# then: Runtime -> Restart session
```
```python
import os, sys
!git clone https://github.com/JingyiShao-Gina/Korean-IGT-Bolmo.git /content/Korean-IGT-Bolmo 2>/dev/null || git -C /content/Korean-IGT-Bolmo pull
os.chdir("/content/Korean-IGT-Bolmo"); sys.path.insert(0, "zero_shot")
import run_zero_shot as rz
rz.main(["--split", "dev", "--limit", "10"])   # smoke test, not a result
rz.main(["--split", "test"])                   # full run, about 2 h on a T4
```

## Generation settings

Bolmo is a base (non-chat) model, so each prompt ends with its answer label (`Transliteration:` / `Gloss:`) and the model continues from there. Decoding is greedy (`do_sample=False`, `num_beams=1`), with the same budget for every call. Bolmo's `generate` keeps writing after the answer, so only the first answer line(s) are parsed (see `results/zero_shot/NOTES.md`).

## Prompt versions

Prompts follow Korean IGT Experimental Protocol V3, section 2.

- `v2`: the all-in-one prompt showed `Transliteration: ...` / `Gloss: ...` as a format example.
- `v3` (current, used for the test run): in the 10-row dev smoke test, Bolmo copied the literal `...` in 9/10 all-in-one outputs, so the example was replaced with a written two-line instruction. Step-by-step prompts are unchanged.

Any further prompt change needs a new version, must be tested on dev only, and requires the test run to be repeated.
