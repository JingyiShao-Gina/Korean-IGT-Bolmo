# Zero-shot protocol (Bolmo 1B)

Owner: Gina. Agenda 17 Nov: complete zero-shot with Bolmo 1B in two settings, all-in-one and step-by-step. Prompts follow Korean IGT Experimental Protocol V3, section 2 (prompt version `v2`).

## Task
- Input: original Korean script (`text`) and English translation (`translation`).
- Output: predicted transliteration and morpheme-level gloss.
- Gold targets (evaluation only, never in prompts): `transcription` (Yale romanization, not morpheme-segmented) and `glosses`.
- Data: `Soyyeongg/Korean_igt_cleaned`, revision pinned in `data/manifest.json`.

## Settings
- **All-in-one** (`prompts/all_in_one.txt`): one call; the model writes `Transliteration: ...` and `Gloss: ...`.
- **Step-by-step**: step 1 (`prompts/transliteration.txt`) predicts the transliteration; step 2 (`prompts/gloss.txt`) receives the step-1 prediction, never the gold transliteration.

Bolmo is a base (non-chat) model, so prompts are plain completions that end with the answer label (`Transliteration:` / `Gloss:`). Generation is greedy (`do_sample=False`, `num_beams=1`), with the same 768-byte budget for every call. Generation stops after the expected answer line(s).

## Running
Open `zero_shot/zero_shot_bolmo.ipynb` in Colab with a GPU runtime. Run the smoke test (10 dev examples) first, check the outputs, then run the full test split. The notebook writes one CSV per setting to `results/zero_shot/`.

## Output CSV columns
`id, text, translation, gold_transliteration, gold_gloss, pred_transliteration, pred_gloss, parse_status, truncated`, plus raw model output(s). Step-by-step files keep both `raw_output_step1` and `raw_output_step2`.

- `parse_status`: `ok`, `empty_transliteration`, `missing_gloss`, or `empty_gloss`.
- `truncated`: the 768-byte budget ran out before the answer was complete.

Failed or truncated rows are kept as they are. They are not repaired or retried, and the evaluator scores an unusable prediction as empty. A run metadata JSON next to the CSVs records the model/dataset revisions, prompt hashes, package versions, GPU and runtime.

Prompt changes made after looking at outputs require a new `prompt_version`, and they must be tuned on dev only, never test.
