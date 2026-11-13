# Zero-shot Bolmo 1B: run notes and issues

Owner: Gina. Run date: 13 Nov 2026 on Colab T4, transformers 4.57.3, prompt version `v3`, model `allenai/Bolmo-1B` @ `452690a`, code @ `8f578a1`. Data: `Soyyeongg/Korean_igt_cleaned` test split (263 rows).

Accuracy has not been computed yet (Soyeong's evaluation step). Everything below comes from inspecting the prediction CSVs.

## 1. Results at a glance

| | All-in-one | Step-by-step |
|---|---|---|
| Rows | 263 | 263 |
| `parse_status` ok | 260 | 263 |
| `missing_gloss` | 3 | 0 |
| Truncated | 3 (1.1%) | 28 (10.6%; step 1: 24, step 2: 25) |
| Runtime | 44.7 min | 70.6 min |

`ok` means only that the output could be parsed. It does **not** mean that the prediction is correct.

## 2. Model behaviour (for error analysis)

Bolmo 1B does not produce IGT zero-shot. It mostly copies one of its inputs:

| | All-in-one | Step-by-step |
|---|---|---|
| Transliteration = Korean source copied verbatim | 217 | 126 |
| Transliteration contains Hangul | 263 | 181 |
| Gloss = English translation copied verbatim | 231 | 3 |
| Gloss = Korean source copied verbatim | 0 | 57 |
| Gloss contains Hangul | 8 | 228 |
| Gloss contains any `-TAG` (e.g. `-NOM`) | 0 | 0 |
| Exact match with gold (transliteration / gloss) | 0 / 0 | 0 / 0 |

- **All-in-one**: line 1 is almost always the Korean script, and line 2 (`Gloss:`) is almost always the English translation.
- **Step-by-step**: step 1 usually copies the Korean script. Occasionally it attempts a romanization, but not in the Yale convention (e.g. `Min-yong-sa-ga ...`, `nona jall hassaya`). Step 2 mostly outputs Korean text or unrelated words. Errors from step 1 propagate into step 2, as expected.
- **Repetition loops** cause every truncated row, e.g. `음은 음은 음은 ...`, `의사 소리 의사 소리 ...`, `Sang Gae Sang Gae ...`. The 3 all-in-one `missing_gloss` rows are such loops, where the model never reached a `Gloss:` line.

We expect morpheme and word accuracy close to 0 for both settings. This is the zero-shot baseline that LoRA fine-tuning (RQ2) is compared against.

## 3. Issues met during the run

1. **Colab environment.** Colab ships transformers 5.x, and Bolmo's remote code also needs `xlstm`. Fix: `pip install "transformers==4.57.3" "xlstm==2.0.4"`, then restart the session. A restart resets the working folder, so the notebook now `chdir`s into the repo before importing. The runner checks both packages before loading the model.
2. **Prompt v2 → v3 (all-in-one only).** In the dev smoke test (10 rows), v2 showed a format example `Transliteration: ... / Gloss: ...`, and Bolmo copied the literal `...` in 9 of the 10 rows. In v3 the example is replaced by a written two-line instruction. The step-by-step prompts are unchanged. The change was decided on dev only; test was run once, after the prompts were frozen.
3. **Generation does not stop after the answer.** Bolmo's custom `generate` appears to ignore the stopping criteria: 260/260 parsed all-in-one outputs continue after the two answer lines (median 601 generated ids). This does not change the predictions, because decoding is greedy and only the first answer line(s) are parsed. It does make the runs slow (about 2 h in total).
4. **Truncation flag is approximate.** `n_new_tokens` counts generated ids, and these are not exactly bytes for Bolmo (one output has 1312 ids against a 768 budget). A row is marked `truncated` when its answer is incomplete *and* at least 768 ids were generated. All flagged rows inspected so far are genuine repetition loops.
5. **Transliteration convention.** The gold transcription is Yale romanization without morpheme segmentation. The prompts (following Protocol V3) do not name Yale, so even a reasonable romanization by the model would differ from gold. Mention this as a limitation.

## 4. Notes for evaluation (Soyeong)

- Use `pred_transliteration` / `pred_gloss` against `gold_transliteration` / `gold_gloss`. Both CSVs have the same 263 IDs in the same order.
- Keep every row. `missing_gloss` and truncated rows count as wrong or empty predictions. Do not drop them.
- Report the parse-failure and truncation rates from section 1 next to the accuracies.
