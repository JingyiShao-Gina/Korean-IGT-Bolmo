# Few-shot prompts (Bolmo 1B)

There is no separate few-shot prompt template. The runner (`zero_shot/run_zero_shot.py --shots K`) builds each few-shot prompt from the shared zero-shot templates in `zero_shot/prompts/`. It inserts K solved train examples before the sentence to annotate (`add_examples` and `examples_block`). How the examples are chosen, and the few-shot results, are described in `zero_shot/README.md` (section "Few-shot").

The files in this folder are **for reading only**: the runner does not use them. They show the prompts exactly as the test run built them.

## `prompts/ainu/` (3-shot, test run)

| File | Content |
|---|---|
| `ainu_3shot_all_in_one_template.txt` | All-in-one template with the three examples filled in. The sentence to annotate is still `{original_script}` / `{translation}`. |
| `ainu_3shot_step_by_step_template.txt` | The same for step-by-step. |
| `ainu_3shot_all_in_one_K7807151UP.086.txt` | The prompt for test sentence `K7807151UP.086`, exactly as sent to the model (all-in-one). |
| `ainu_3shot_step_by_step_K7807151UP.086.txt` | The same sentence, step-by-step. |

The SHA-256 of each template equals `prompt_sha256_filled` in the run's `results/few_shot/ainu/*.meta.json`, so these are the prompts the model saw. The prompt sent to the model ends at `Transliteration:` with no final newline; the files add one.

Examples (train ids, seed 42): `OI990508KY.094`, `K7807151KY.122`, `K7708242UP.102`.
