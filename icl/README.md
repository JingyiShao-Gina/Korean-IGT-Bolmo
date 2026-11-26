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
