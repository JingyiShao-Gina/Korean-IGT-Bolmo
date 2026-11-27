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
| Ainu | `lowresource-igt/Ainu_igt_dataset` (`data/manifest_ainu.json`, **not cleaned**) | 750 | zero-shot only (no LoRA). **Done** (prompt v5): `results/zero_shot/ainu/bolmo-1b_{all_in_one,step_by_step}_test.csv`; scores in `results/zero_shot/ainu/scores.csv`. 3-shot **done**: `results/few_shot/ainu/` (scores below) |
| Tsez (ddo) | `lowresource-igt/Tsez_igt_dataset` (`data/manifest_tsez.json`, **not cleaned**) | 445 | zero-shot only. **Done** (prompt v5, Theo): `results/zero_shot/tsez/bolmo-1b_{all_in_one,step_by_step}_test.csv`; scores in `results/zero_shot/tsez/scores.csv` |
| Jejueo | `lowresource-igt/Jejueo_igt_dataset` (`data/manifest_jejueo.json`, **not cleaned**) | 712 | zero-shot only. **Done** (prompt v5, Soyeong): `results/zero_shot/jejueo/bolmo-1b_{all_in_one,step_by_step}_test.csv`; scores in `results/zero_shot/jejueo/scores.csv`. 3-shot **done**: `results/few_shot/jejueo/` (scores below). Glosses write suffixes as separate tokens (`eat -PST -POL -DECL`); scoring re-attaches them before word-level metrics. Gold transcription is IPA. |

### Ainu zero-shot scores (test, 750 rows)

Scored with `zero_shot/score_zero_shot.py --lang ainu`, which uses the team metrics in `evaluation/metrics.py` (Soyeong; positional accuracy after Ginn et al. 2024, case-insensitive, corpus micro-average). The script reproduces the Korean all-in-one numbers on the `soyeong/evaluation` branch (0.0594 / 0.0188).

| | All-in-one | Step-by-step |
|---|---|---|
| Gloss morpheme accuracy | 0.85% | 0.95% |
| Gloss word accuracy | 1.04% | 1.11% |
| Gloss MER / WER | 1.15 / 1.38 | 1.25 / 1.49 |
| Gloss chrF++ | 9.92 | 10.46 |
| Transliteration chrF++ | 5.11 | 5.77 |
| Parse failure / truncation | 8.7% / 10.8% | 5.3% / 7.3% |

Both settings fail: Bolmo 1B copies the katakana source as "transliteration" and the English translation as "gloss", and it never produces a gloss tag. The few matches are English words that happen to sit at the same position as in the gold gloss (`and` 13×, `then` 11×, `father`, `village`, …). The step-by-step vs all-in-one difference (0.95% vs 0.85%) is too small to mean anything. On the 255 rows whose gold transcription and (re-joined) gloss have the same word count, the results are the same (`aligned_word_count` rows in `scores.csv`). MER/WER above 1 means that the model writes more tokens than the gold gloss has, and each extra token counts as an insertion.

Ainu dev = the dataset's `eval` split (218 rows). The Ainu source is in katakana, and the gold transcription is Latin script. Known data issues are listed in `data/manifest_ainu.json`; for example, only 255/750 test rows have the same number of words in transcription and gloss (180 before the detached affixes are re-joined), which matters for word-level accuracy.

**Update 27 Nov (Soyeong):** like Jejueo, the Ainu glosses write many affixes and clitics as separate tokens (`APASS- do`, `4.A= COP`; 550/750 test rows). Since 27 Nov, `data/manifest_ainu.json` sets `gloss_affixes_detached`, so scoring re-joins them before word-level metrics (a lone `-`, which glosses a pause, stays a word). This raises Ainu word accuracy and changes WER and chrF++ (chrF++ uses word bigrams); morpheme accuracy and MER do not change. All Ainu tables in this repo show the re-joined scores.

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

### Jejueo zero-shot scores (test, 712 rows)

Scored with `zero_shot/score_zero_shot.py --lang jejueo` (same metrics as Ainu and Tsez). The Jejueo glosses write suffixes as separate tokens (`eat -PST -POL -DECL`), so the script first re-attaches them (`metrics.join_detached_affixes`, both gold and prediction); this changes only word-level metrics, not morpheme-level ones. Run on a Colab T4 GPU; runtime about 37 min (all-in-one) and 27 min (step-by-step).

| | All-in-one | Step-by-step |
|---|---|---|
| Gloss morpheme accuracy | 1.08% | 1.08% |
| Gloss word accuracy | 1.18% | 1.16% |
| Gloss MER / WER | 1.06 / 1.50 | 1.22 / 1.82 |
| Gloss chrF++ | 8.71 | 9.08 |
| Transliteration chrF++ | 0.30 | 0.18 |
| Transliteration word accuracy | 0.00% | 0.00% |
| Parse failure (`missing_gloss`) / truncation | 6.0% / 6.0% | 1.7% / 1.7% |

Same failure as Ainu and Tsez: Bolmo 1B copies the Hangul source unchanged as "transliteration" and the English translation as "gloss", with no gloss tags. The gold transcription is IPA (`məkəsʰuta`), so transliteration scores are close to zero. The few matches are English words at the same position as in the gold gloss (`so` 11×, `and`, `but`, `now`, `go`, …). The two settings are not meaningfully different. On the 648 rows whose gold transcription and (re-joined) gloss have the same word count, the results are the same (`aligned_word_count` rows in `scores.csv`). The Colab run installed `xlstm` 2.0.3 (the README pins 2.0.4).

Jejueo dev = the dataset's `eval` split (211 rows). In the 10-row dev smoke test, one row had the source text `[ ]` (an empty placeholder in the spoken corpus) and the step-by-step output was a repeated character; the test split has no such rows. Source: wav2gloss / DoReCo spoken corpus. Known data issues are listed in `data/manifest_jejueo.json`.

## Few-shot

`--shots K` adds K solved train examples to the all-in-one and step-by-step prompts. Everything else is the same as zero-shot: model, greedy decoding, generation budget, parsing and scoring. Zero-shot runs (`--shots 0`, the default) are unchanged.

```
python zero_shot/run_zero_shot.py --lang jejueo --split dev --limit 10 --shots 3   # smoke test
python zero_shot/run_zero_shot.py --lang jejueo --split test --shots 3             # full run
python zero_shot/score_zero_shot.py --lang jejueo --shots 3                        # scores
```

On Colab, use `zero_shot/few_shot_bolmo.ipynb` (set `LANG` and `SHOTS`).

- **Examples.** The same K examples are used for every evaluated row. They are drawn with seed 42 from train rows that meet all of these conditions (settings under `few_shot` in `configs/zero_shot.json`):
  - all four fields are filled in
  - the text is not a placeholder such as `[ ]`
  - the gloss contains a grammatical tag
  - the text has at least 3 words and at most 60 characters
  - the gloss has at most 100 characters
  - the text does not occur in the dev or test split, so the model is never shown the answer

  Because dev and test texts are both excluded, the dev smoke test and the test run use the same examples. Prompt layout: the template's header and instructions come first, then the K examples (`Original … script:`, `English translation:`, `Transliteration:`, `Gloss:`, with gold values), then the sentence to annotate in the same format, ending at `Transliteration:`. A first version put the instructions between the sentence and the answer label, and on 2 Jejueo dev rows Bolmo copied example 1 word for word. That is why the sentence now comes right before the label. Example glosses are copied from the dataset as they are, so Jejueo examples keep detached suffixes (`eat -PST -DECL`).
- **Output.** Results are written to `results/few_shot/<lang>/bolmo-1b_<strategy>_<K>shot_<split>.csv`. The `.meta.json` file records `shots`, the seed and the full examples (`few_shot_examples`). There is no separate few-shot prompt file. The runner builds the prompt from the zero-shot templates (`add_examples` in `run_zero_shot.py`). To read the Ainu 3-shot prompts in full, see `few_shot/prompts/ainu/`.
- **Interruptions (`--resume`).** With `--resume`, each finished row is appended right away to `<output>.partial.jsonl` (flushed to disk). Rerunning the same command skips the rows already there and continues, and the final CSV is the same as an uninterrupted run (greedy decoding; checked in `test_parsing.py`). The file's first line stores the run settings (language, strategy, split, shots, example ids, model revision, generation settings, prompt hashes), and a run with other settings refuses to reuse it. The meta records `resumed` and `rows_from_checkpoint`. Never start two processes on the same file. In Colab, use the **Resumable test run** section at the top of `few_shot_bolmo.ipynb`: it runs both settings as background processes and writes to Google Drive (`MyDrive/korean-igt-few-shot/<lang>/`), so a closed browser or a lost runtime does not lose the finished rows.
- **Prompt length.** With 3 shots, test prompts are about 1.2–1.6 KB on average (Jejueo 1.2, Ainu 1.3, Tsez 1.6), compared with about 0.5–0.7 KB for zero-shot. On a T4 the 10-row Jejueo dev smoke test took 145 s (all-in-one) and 160 s (step-by-step), about 3× zero-shot (45 s). Most of that is repetition loops (3/10 truncated in each setting). On the test split each setting took about 2.4–2.5 h for Ainu and about 5.1–5.2 h for Jejueo (an earlier estimate of 2 h was too low; Jejueo loops more, see below). The notebook therefore runs the two settings in separate cells (section 5: all-in-one, section 6: step-by-step), each followed by its own download.
- **Selected train ids (3 shots).**
  - Jejueo: `…jeju0060-05_0125`, `…jeju0060-05_0031`, `…jeju0063-01-02_0221`
  - Ainu: `OI990508KY.094`, `K7807151KY.122`, `K7708242UP.102`
  - Tsez: `708`, `175`, `1713`

### Ainu 3-shot scores (test, 750 rows)

Results: `results/few_shot/ainu/bolmo-1b_{all_in_one,step_by_step}_3shot_test.csv`. Scores: `results/few_shot/ainu/scores.csv`, from `zero_shot/score_zero_shot.py --lang ainu --shots 3`. The run used a Colab T4 with `--resume` as background processes; each run finished in one go (`resumed: false`). Runtime was about 2.4 h (all-in-one) and 2.5 h (step-by-step). Zero-shot is shown in brackets for comparison.

| | All-in-one | Step-by-step |
|---|---|---|
| Gloss morpheme accuracy | 1.47% (0.85%) | 1.61% (0.95%) |
| Gloss word accuracy | 1.86% (1.04%) | 1.74% (1.11%) |
| Gloss MER / WER | 2.95 / 3.71 | 3.39 / 4.42 |
| Gloss chrF++ | 9.79 (9.92) | 9.16 (10.46) |
| Transliteration chrF++ | 18.84 (5.11) | 18.40 (5.77) |
| Transliteration word accuracy | 8.15% (0.55%) | 8.61% (0.57%) |
| Parse failure (`missing_gloss`) / truncation | 6.4% / 11.1% | 6.5% / 12.4% |

The examples change the kind of output.
- **Transliteration.** Every transliteration is now in Latin script. In zero-shot, 99% of them kept the katakana. About 8% of the words match the gold exactly (for example `hine ora`), so transliteration chrF++ rises from about 5 to about 18.
- **Gloss.** Glosses now look like IGT: about half of the all-in-one glosses contain gloss tags (`4.S`, `COP`, `INFR.EV`), against 4% in zero-shot. Copies of the English translation fall from 70% to 10% (all-in-one).
- **Accuracy.** Gloss accuracy still stays below 2%, because the tags and stems rarely land on the right word. Many tags are copied from the examples (`INFR.EV COP then`).
- **Gloss chrF++.** It barely changes. With MER/WER around 3, the glosses are much longer than the gold, often because the model repeats itself until it runs out of budget.
- **Settings.** As in zero-shot, step-by-step and all-in-one are too close to tell apart.

### Jejueo 3-shot scores (test, 712 rows)

Results: `results/few_shot/jejueo/bolmo-1b_{all_in_one,step_by_step}_3shot_test.csv`. Scores: `results/few_shot/jejueo/scores.csv`, from `zero_shot/score_zero_shot.py --lang jejueo --shots 3` (detached suffixes re-attached before word-level metrics, as in zero-shot). Run on a Colab T4 from notebook sections 5 and 6 (git `a65a7f9`, not resumed); runtime about 5.1 h (all-in-one) and 5.2 h (step-by-step). Zero-shot is shown in brackets.

| | All-in-one | Step-by-step |
|---|---|---|
| Gloss morpheme accuracy | 2.43% (1.08%) | 2.18% (1.08%) |
| Gloss word accuracy | 1.38% (1.18%) | 1.21% (1.16%) |
| Gloss MER / WER | 14.02 / 7.47 (1.06 / 1.50) | 13.87 / 6.88 (1.22 / 1.82) |
| Gloss chrF++ | 2.83 (8.71) | 2.73 (9.08) |
| Transliteration chrF++ | 3.29 (0.30) | 3.20 (0.18) |
| Transliteration word accuracy | 1.70% (0.00%) | 1.76% (0.00%) |
| Parse failure (`missing_gloss`) / truncation | 24.0% / 52.8% | 23.7% / 51.5% |

The examples change the kind of output, as for Ainu, but repetition loops dominate the scores.
- **Transliteration.** It is now romanized/IPA-like in 711 of 712 rows (zero-shot copied the Hangul in most rows), so transliteration chrF++ rises from about 0.3 to about 3. It is still far from the gold IPA.
- **Gloss.** 462 of the 541 parsed all-in-one glosses contain gloss tags (`-CVB`, `-PST`, `-DECL`), against 0 in zero-shot. Copies of the English translation fall from 501 to 23. Morpheme accuracy doubles, but stays below 3%.
- **Loops.** About half of the rows hit the generation budget. In 205 all-in-one rows (197 step-by-step) the gloss repeats a pattern until the budget runs out (`do -CG do -CG do -CG …`); these rows hold 97% of all predicted gloss words. This is why MER is about 14 and gloss chrF++ falls from about 9 to about 3. In another 171 rows the transliteration line itself loops (`aɪ aɪ aɪ …`), so no `Gloss:` line is reached (`missing_gloss`). The median gloss length is 7 words, the same as the gold, so a loop stop or a lower byte cap (`fast_run/`) would mainly change MER and chrF++, not accuracy.
- **Settings.** All-in-one and step-by-step are again too close to tell apart.
- **Examples.** The 3 fixed examples are short spoken-corpus turns (`DEM.MED eat -CVB see -ADN ALREADY EXIST`, …). 13 all-in-one glosses copy one of them exactly.

## What each file does

| File | Purpose |
|---|---|
| `zero_shot/prompts/all_in_one.txt` | All-in-one prompt (shared by all languages). It ends with `Transliteration:`, and the model answers `Transliteration:` / `Gloss:`. |
| `zero_shot/prompts/step_by_step.txt` | Step-by-step prompt, one call (shared). It is identical to all-in-one except for the task description ("Work step by step. Step 1: … Step 2: Based on your transliteration from Step 1, …"). It uses the same answer labels, `Transliteration:` / `Gloss:`. |
| `few_shot/prompts/ainu/` | The full Ainu 3-shot prompts, for reading only (see `few_shot/README.md`). |
| `zero_shot/prompts/two_stage_transliteration.txt`, `two_stage_gloss.txt` | Two-stage prompts (call 1, call 2): the earlier "step-by-step", renamed from `transliteration.txt` / `gloss.txt`. Not one of the agenda settings. |
| `configs/zero_shot.json` | Run settings: model ID and revision, prompt version, the prompt files, and, for each language, its name, data manifest and output folder. Also greedy decoding and the generation budget. To add a language, add an entry under `languages` and a manifest in `data/`. |
| `data/manifest_ainu.json` | Ainu dataset ID, pinned revision, split names and counts, data-quality notes. |
| `zero_shot/run_zero_shot.py` | The runner. It loads data and model, fills in the prompts, generates, parses the answers, and writes a CSV plus a `.meta.json` for each setting. CLI: `--lang korean/ainu`, `--split dev/test`, `--limit N`, `--strategy all_in_one/step_by_step/two_stage/both` (`both` = all-in-one + step-by-step). |
| `zero_shot/zero_shot_bolmo.ipynb` | Colab notebook: install, choose `LANG`, prompt preview, 10-row dev smoke test, full test run, download. |
| `zero_shot/few_shot_bolmo.ipynb` | Colab notebook for few-shot: like the zero-shot one, plus `SHOTS`; checks out the few-shot branch, previews the prompt with its examples, downloads `results/few_shot/<lang>/`. |
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
