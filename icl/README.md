# In-context learning with retrieved examples

Tutor (1 Dec agenda): select relevant examples from the training set by word overlap with each test sentence; more than 20 examples; all languages.

`icl/run_icl.py` puts, for every sentence, the **K = 21** most similar train sentences into the prompt (gold transliteration and gloss), most similar last. Prompts, example format, parsing, checkpoints and output columns are those of the 3-shot runs (`zero_shot/run_zero_shot.py`, not changed); only the examples differ, per sentence instead of fixed.

- **Similarity.** Shared word types in the original script (NFKC-normalised, lower-cased, whitespace tokens, punctuation dropped). Ties: shared English translation words (stopwords removed), then the shorter example, then id.
- **Pool.** Train rows with the same filters as the fixed 3-shot examples (all fields, a gloss tag, text ≤ 60 characters, gloss ≤ 100, ≥ 3 words); no dev or test text. Ainu: 5,388 of 6,726 train rows.
- **Ainu check (offline, test split).** Every test sentence finds 21 examples that share at least one word with it (747/750 rows); the best example shares 4.7 words on average. Prompts are 6.8 KB on average, 8.3 KB at most. Bolmo 1B's 4096 positions count patches (about 2–3 bytes each), so this should fit; check `prompt_bytes` and the outputs of the longest prompts in the smoke test.
- Each output row adds `icl_example_ids` (most similar last), `icl_overlap` and `prompt_bytes`.

```
python icl/run_icl.py --lang ainu --split dev --limit 10                  # smoke test
nohup python icl/run_icl.py --lang ainu --split test --strategy all_in_one --resume > /content/icl_aio.log 2>&1 &
python icl/score_icl.py --lang ainu
```

`--fast` adds the speed-ups of `fast_run/run_fast.py` (loop stop, byte cap, bf16 on A100/L4). `--k` changes the number of examples. Runs with `--resume` survive a closed browser; if Colab recycles the runtime, run the same command again and it continues where it stopped.

Output: `results/icl/<lang>/bolmo-1b_<strategy>_icl21_test.csv` (+ `.meta.json`, `scores.csv`).

## Ainu result (test, 750 rows, T4, float32, `--fast`, 27 Nov)

| Metric | 3-shot all-in-one | 3-shot step-by-step | ICL-21 all-in-one | ICL-21 step-by-step |
|---|---|---|---|---|
| Gloss morpheme accuracy | 1.47% | 1.61% | 24.40% | 24.55% |
| Gloss word accuracy | 1.33% | 1.49% | 9.16% | 9.22% |
| Gloss chrF++ | 9.57 | 9.01 | 42.15 | 42.32 |
| Gloss MER | 2.95 | 3.39 | 0.71 | 0.70 |
| Transliteration chrF++ | 18.84 | 18.40 | 65.39 | 65.46 |
| Transliteration word accuracy | 8.15% | 8.61% | 48.16% | 48.47% |
| Parse failure / truncation | 6.4% / 11.1% | 6.5% / 12.4% | 2.4% / 4.1% | 2.7% / 4.5% |

- Retrieved examples raise every metric by a large margin; the two strategies are now almost equal. The examples share words with the test sentence, so the model can copy their transliteration and gloss; no dev or test text is in the pool.
- Loop stops: 31 (all-in-one) and 34 (step-by-step) of 750 rows; runtime 70 and 63 minutes. Generation was cut only by the loop stop (no row reached the 1,000-byte cap), so apart from those rows the outputs are what the original runner would produce (checked on the first 70 rows: 68 identical, 2 looping rows cut early).
- Files: `results/icl/ainu/` (`run_log.txt` is the Colab log).
