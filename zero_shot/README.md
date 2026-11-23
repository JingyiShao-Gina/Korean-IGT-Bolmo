# Zero-shot experiments (Bolmo 1B)

Owner: Gina. Bolmo 1B, no fine-tuning, two settings from the agenda:

- **All-in-one**: one call that asks for the transliteration and the gloss together.
- **Step-by-step**: one call whose prompt asks for **Step 1** (transliteration) and then **Step 2** (the gloss, based on the Step 1 transliteration). This is the tutor's definition (clarified 20 Nov).
- **Two-stage** (extra, Korean only): two separate calls, first the transliteration and then the gloss from the predicted transliteration. This is what we first ran as "step-by-step"; Theo's LoRA two-stage setup uses the same prompts.

The prompt files are **language-neutral**: they contain a `{language}` placeholder, which the runner fills from the language config (`Korean`, `Ainu`, ...). The same three prompts are used for every language. For Korean, the filled prompts are word-for-word the ones used in the earlier Korean runs.

Input: original script (`text`) + English translation (`translation`). Output: transliteration + morpheme-level gloss. Gold targets (`transcription`, `glosses`) are used for evaluation only and never appear in prompts.

## Languages

| Language | Data (manifest) | Test rows | Status |
|---|---|---|---|
| Korean | `Soyyeongg/Korean_igt_cleaned` (`data/manifest.json`, cleaned by Soyeong) | 263 | all-in-one and two-stage done (`results/zero_shot/`); single-call step-by-step to run |
| Ainu | `lowresource-igt/Ainu_igt_dataset` (`data/manifest_ainu.json`, **not cleaned**) | 750 | zero-shot only (no LoRA). **Done** (prompt v5): `results/zero_shot/ainu/bolmo-1b_{all_in_one,step_by_step}_test.csv`; scores in `results/zero_shot/ainu/scores.csv` |
| Tsez (ddo) | `lowresource-igt/Tsez_igt_dataset` (`data/manifest_tsez.json`, **not cleaned**) | 445 | zero-shot only. **Done** (prompt v5, Theo): `results/zero_shot/tsez/bolmo-1b_{all_in_one,step_by_step}_test.csv`; scores in `results/zero_shot/tsez/scores.csv` |
| Jejueo | `lowresource-igt/Jejueo_igt_dataset` (`data/manifest_jejueo.json`, **not cleaned**) | 712 | zero-shot only (Soyeong). To run. Glosses write suffixes as separate tokens (`eat -PST -POL -DECL`); scoring re-attaches them before word-level metrics. Gold transcription is IPA. |

### Ainu zero-shot scores (test, 750 rows)

Scored with `zero_shot/score_zero_shot.py --lang ainu`, which uses the team metrics in `evaluation/metrics.py` (Soyeong; positional accuracy after Ginn et al. 2024, case-insensitive, corpus micro-average). The script reproduces the Korean all-in-one numbers on the `soyeong/evaluation` branch (0.0594 / 0.0188).

| | All-in-one | Step-by-step |
|---|---|---|
| Gloss morpheme accuracy | 0.85% | 0.95% |
| Gloss word accuracy | 0.87% | 0.95% |
| Gloss MER / WER | 1.15 / 1.17 | 1.25 / 1.25 |
| Gloss chrF++ | 9.79 | 10.31 |
| Transliteration chrF++ | 5.11 | 5.77 |
| Parse failure / truncation | 8.7% / 10.8% | 5.3% / 7.3% |

Both settings fail: Bolmo 1B copies the katakana source as "transliteration" and the English translation as "gloss", and it never produces a gloss tag. The few matches are English words that happen to sit at the same position as in the gold gloss (`and` 13×, `then` 11×, `father`, `village`, …). The step-by-step vs all-in-one difference (0.95% vs 0.85%) is too small to mean anything. On the 180 rows whose gold transcription and gloss have the same word count, the results are the same (`aligned_word_count` rows in `scores.csv`). MER/WER above 1 means that the model writes more tokens than the gold gloss has, and each extra token counts as an insertion.

Ainu dev = the dataset's `eval` split (218 rows). The Ainu source is in katakana, and the gold transcription is Latin script. Known data issues are listed in `data/manifest_ainu.json`; for example, only 180/750 test rows have the same number of words in transcription and gloss, which matters for word-level accuracy.

### Tsez zero-shot scores (test, 445 rows)

Scored with `zero_shot/score_zero_shot.py --lang tsez` (same metrics as Ainu). Run on a Colab L4 GPU (Korean and Ainu used a T4); greedy decoding, so the GPU should not change the outputs. Runtime: about 36 min (all-in-one) and 32 min (step-by-step).

| | All-in-one | Step-by-step |
|---|---|---|
| Gloss morpheme accuracy | 1.56% | 1.57% |
| Gloss word accuracy | 1.43% | 1.36% |
| Gloss MER / WER | 0.95 / 1.45 | 0.95 / 1.45 |
| Gloss chrF++ | 12.68 | 12.93 |
| Transliteration chrF++ | 2.33 | 2.59 |
| Transliteration word accuracy | 0.09% | 0.13% |
| Parse failure (`missing_gloss`) / truncation | 6.5% / 6.5% | 3.6% / 3.6% |

Same failure as Ainu: Bolmo 1B copies the Cyrillic source unchanged as "transliteration" (the gold is Latin script, hence the very low transliteration chrF++) and copies the English translation as "gloss", with no gloss tags. The step-by-step vs all-in-one difference is too small to mean anything. All 445 Tsez test rows have the same word count in transcription and gloss, so the `aligned_word_count` rows in `scores.csv` equal the `all` rows. The Colab run installed `xlstm` 2.0.3 (the README pins 2.0.4).

Tsez dev = the dataset's `eval` split (445 rows). Source: GlossLM; source text is Cyrillic, gold transcription is Latin script.

## What each file does

| File | Purpose |
|---|---|
| `zero_shot/prompts/all_in_one.txt` | All-in-one prompt (shared by all languages). It ends with `Transliteration:`, and the model answers `Transliteration:` / `Gloss:`. |
| `zero_shot/prompts/step_by_step.txt` | Step-by-step prompt, one call (shared). It is identical to all-in-one except for the task description ("Work step by step. Step 1: … Step 2: Based on your transliteration from Step 1, …"). It uses the same answer labels, `Transliteration:` / `Gloss:`. |
| `zero_shot/prompts/two_stage_transliteration.txt`, `two_stage_gloss.txt` | Two-stage prompts (call 1, call 2): the earlier "step-by-step", renamed from `transliteration.txt` / `gloss.txt`. Not one of the agenda settings. |
| `configs/zero_shot.json` | Run settings: model ID and revision, prompt version, the prompt files, and, for each language, its name, data manifest and output folder. Also greedy decoding and the generation budget. To add a language, add an entry under `languages` and a manifest in `data/`. |
| `data/manifest_ainu.json` | Ainu dataset ID, pinned revision, split names and counts, data-quality notes. |
| `zero_shot/run_zero_shot.py` | The runner. It loads data and model, fills in the prompts, generates, parses the answers, and writes a CSV plus a `.meta.json` for each setting. CLI: `--lang korean/ainu`, `--split dev/test`, `--limit N`, `--strategy all_in_one/step_by_step/two_stage/both` (`both` = all-in-one + step-by-step). |
| `zero_shot/zero_shot_bolmo.ipynb` | Colab notebook: install, choose `LANG`, prompt preview, 10-row dev smoke test, full test run, download. |
| `zero_shot/score_zero_shot.py` | Scores a language's two test CSVs with `evaluation/metrics.py` and writes `results/zero_shot/<lang>/scores.csv` (`--lang ainu`; `--check-korean` re-checks the Korean numbers). |
| `evaluation/metrics.py`, `evaluation/test_metrics.py` | Team metrics by Soyeong, copied unchanged from the `soyeong/evaluation` branch. |
| `zero_shot/test_parsing.py` | CPU-only checks for prompts, parsing and the stopping logic (`python zero_shot/test_parsing.py`). |
| `results/zero_shot/bolmo-1b_all_in_one_test.csv` | Korean all-in-one test predictions (263 rows, prompt v3). |
| `results/zero_shot/bolmo-1b_step_by_step_test.csv` | Korean **two-stage** test predictions (263 rows). The file name is from before the step-by-step definition changed; it is kept because Soyeong's evaluation reads it. |
| `results/zero_shot/<lang>/` | New runs (Ainu, and Korean single-call step-by-step) are written here. |
| `results/zero_shot/*.meta.json` | Record of each run: language, model/dataset revision, prompt version, files and SHA-256, git commit, package versions, GPU, runtime, parse/truncation counts. |
| `results/zero_shot/NOTES.md` | Korean run notes: results, model behaviour for error analysis, problems met, notes for evaluation. |

## Prediction CSV columns

| Column | Meaning |
|---|---|
| `id`, `text`, `translation` | Dataset row ID, source script, English translation (model input). |
| `gold_transliteration`, `gold_gloss` | Reference answers from the dataset. |
| `pred_transliteration`, `pred_gloss` | Parsed model predictions; use these for evaluation. |
| `parse_status` | `ok`, `missing_gloss` (no gloss line), `empty_answer` (transliteration and gloss both empty), `empty_gloss` or `empty_transliteration`. `ok` = parseable, not correct. |
| `truncated` | The answer was incomplete when the generation budget ran out (usually a repetition loop). |
| `raw_output` (all-in-one, step-by-step) / `raw_output_step1`, `raw_output_step2` (two-stage) | Unedited model output. |
| `n_new_tokens*` | Number of generated ids per call. |
| `truncated_step1`, `truncated_step2` | Two-stage only: which call ran out of budget. |

Rows are never repaired, retried or dropped. Unusable predictions are scored as empty.

## How to run (Colab, T4 GPU)

Open `zero_shot/zero_shot_bolmo.ipynb`, set `LANG`, and run it from the top. Or:

```python
!pip install "transformers==4.57.3" "xlstm==2.0.4" datasets pandas accelerate
# then: Runtime -> Restart session
```
```python
import os, sys
!git clone https://github.com/JingyiShao-Gina/Korean-IGT-Bolmo.git /content/Korean-IGT-Bolmo 2>/dev/null || git -C /content/Korean-IGT-Bolmo pull
os.chdir("/content/Korean-IGT-Bolmo"); sys.path.insert(0, "zero_shot")
import run_zero_shot as rz
rz.main(["--lang", "ainu", "--split", "dev", "--limit", "10"])   # smoke test, not a result
rz.main(["--lang", "ainu", "--split", "test"])                   # full run
```

## Generation settings

Bolmo is a base (non-chat) model, so each prompt ends with its answer label and the model continues from there. Decoding is greedy (`do_sample=False`, `num_beams=1`), with the same budget for every call.

Generation stops once the expected answer line(s) are complete. The Korean v3 runs could not stop early: Bolmo's custom `generate` passes only the newly generated bytes to the stopping check, and the old check discarded them as if they were prompt, so every call ran to the end of its budget (about 2 h for the Korean test). This is fixed since v4 (`generated_part` in the runner); for example, the Ainu all-in-one smoke test generated about 169 tokens per row instead of about 600. With greedy decoding, the parsed answers do not change; only the time spent after the answer is saved.

## Prompt versions

Prompts follow Korean IGT Experimental Protocol V3, section 2, and the tutor's step-by-step definition.

- `v2`: the all-in-one prompt showed `Transliteration: ...` / `Gloss: ...` as a format example.
- `v3`: in the 10-row dev smoke test, Bolmo copied the literal `...` in 9/10 all-in-one outputs, so the example was replaced with a written two-line instruction. Used for the Korean test runs (all-in-one, two-stage).
- `v4`: adds the single-call step-by-step prompt, with answer labels `Step 1 - Transliteration:` / `Step 2 - Gloss:`, and makes all prompts language-neutral (`{language}`). The two-stage files were renamed to `two_stage_*.txt`. Filled with "Korean", the all-in-one and two-stage texts are unchanged from v3.
- `v5` (current): in the 10-row Ainu dev smoke test with v4, Bolmo returned an empty answer on 10/10 step-by-step rows (`\nStep 2 - Gloss:\n`: it skipped Step 1 and wrote only the label). v5 keeps the step instructions but uses the all-in-one labels `Transliteration:` / `Gloss:`. On the same 10 rows, all answers were then parseable. Step-by-step now differs from all-in-one **only** in the task-description lines, so a difference between the two settings can be attributed to asking for steps. The change was decided on dev only.

Any further prompt change needs a new version, must be tested on dev only, and requires the test run to be repeated.
