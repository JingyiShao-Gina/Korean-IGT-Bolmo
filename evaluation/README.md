# Evaluation protocol

## Implementation (16 Nov 2026)

- `metrics.py`: word / morpheme accuracy, WER / MER, chrF++. `test_metrics.py`: hand-checked cases (`python evaluation/test_metrics.py`).
- `evaluate.py`: scores the zero-shot and LoRA prediction files and writes `results/evaluation/` and `results/summary.csv`. Run it with `python evaluation/evaluate.py --lora-dir "<korean-igt-lora Drive folder>"`.
- **Decision:** reported word and morpheme accuracy follow Ginn et al. (2024), using **positional** comparison (the i-th predicted token against the i-th gold token, with extra tokens ignored). This replaces the hierarchical edit-distance alignment proposed below. Edit distance is still used for WER / MER, computed over the flattened token sequence. Accuracy and MER compare tokens case-insensitively, following the dataset cleaning summary; chrF++ stays case-sensitive. Results and discussion: `results/evaluation/NOTES.md`.

## Prompting results with the loop stop (27 Nov 2026)

`rescore_loop_stop.py` puts every prompting method (zero-shot, 3-shot, segmentation, ICL) of Ainu, Tsez and Jejueo in one table, `results/evaluation/prompting_scores.csv`, under the same stop rule. No GPU; about 20 s for all files. Tests: `python evaluation/test_rescore.py`.

```
python evaluation/rescore_loop_stop.py               # all languages
python evaluation/rescore_loop_stop.py --lang jejueo
```

- **Why.** The ICL runs use `--fast` (`fast_run/`), which stops generation when the output starts looping or reaches 1,000 bytes. The earlier runs generated to the end of their budget, so a looping tail inflates MER and chrF++ there. The script cuts every saved `raw_output` where the fast runner would have stopped (`fast_run/replay_loop_stop.py`, with a three-line completion check for segmentation), parses it again and scores it. Greedy decoding makes this the output a fast run would give.
- **Rows.** Each file has an `original` row (as generated; equal to the folder's `scores.csv`, which the script checks) and a `loop_stop` row. Extra columns: `stopped_by_loop`, `stopped_by_byte_cap`, `answer_changed`. All test rows are scored; Jejueo suffixes are re-attached as in `score_zero_shot.py`.
- **What changes.** Accuracy does not change for any file: loops come after the answer's first words. MER and chrF++ change a lot where loops are frequent. Jejueo 3-shot all-in-one: MER 14.02 → 1.38, gloss chrF++ 2.83 → 7.83, transliteration chrF++ 3.29 → 9.43 (53% of rows cut by the loop stop). Ainu 3-shot: MER 2.95 → 1.12, gloss chrF++ 9.57 → 12.15. Zero-shot changes little. The ICL rows are identical under both rules, since they were already generated with the stop.
- **Use the `loop_stop` rows when comparing methods** and report `stopped_by_loop` as the loop rate; it is a model failure in its own right (Jejueo 3-shot 53%, zero-shot 2–6%).

## How close are the ICL examples to the test sentences? (27 Nov 2026)

`icl_overlap.py --lang <lang>` looks up the 21 retrieved examples of every ICL test row and measures the character similarity (difflib ratio, NFKC) between the test text and its closest example, whether an example has exactly the gold gloss, and whether the prediction copies an example's gloss. It scores each band separately. Output: `results/evaluation/icl_overlap_summary.csv`, per row `icl_overlap_<lang>.csv`.

Ainu, all-in-one (step-by-step is almost the same):

| Closest example | Rows | Morpheme acc | Gloss chrF++ | Prediction = an example's gloss |
|---|---|---|---|---|
| identical text | 0 (0%) | – | – | – |
| similarity 0.9–1 | 99 (13%) | 87.3% | 83.9 | 90% |
| similarity 0.8–0.9 | 50 (7%) | 40.4% | 62.3 | 46% |
| similarity < 0.8 | 601 (80%) | 13.7% | 34.7 | 7% |
| all | 750 | 24.4% | 42.1 | 21% |

- No test text appears in the pool, but 13% of the test sentences have a near-identical example (spoken corpora repeat phrases), and on those the model mostly copies the example's gloss. 9 test rows (1.2%) have an example with exactly the gold gloss.
- Without the near-duplicates (similarity < 0.8, 80% of rows), ICL still reaches 13.7% morpheme accuracy and 34.7 gloss chrF++, against 1.5% and 12.1 for 3 fixed examples (loop-stopped). So retrieval helps beyond copying, but much of the overall score comes from the 20% of rows with a close example: overall morpheme accuracy is 24.4%, 13.7% on the other rows. Report both numbers.

The original protocol follows.

Freeze these conventions before the main runs. They are project definitions, not a claim of exact reproduction of any published implementation. Align with the team's chosen Ginn et al. (2024) implementation only after checking its precise definitions.

## Normalization and alignment
Use Unicode NFC and collapse whitespace. Preserve case and morphological delimiters; do not strip grammatical labels or punctuation silently. A word is whitespace-delimited; morphemes split within words on hyphen or equals. Document ambiguous annotation cases before freezing the evaluator.

Compute deterministic minimum-edit-distance alignments of words, then morphemes within aligned words. Fix ties in the order match/substitute, delete, insert. Missing/extra words contribute missing/extra morphemes. Use corpus-level summed counts, not an average of sentence percentages.

- Word accuracy: 1 - (substitutions + deletions + insertions) / reference word count.
- Morpheme accuracy: 1 - (substitutions + deletions + insertions) / reference morpheme count under the hierarchical alignment above.
- Morpheme error rate (MER): the corresponding morpheme error count / reference morpheme count.
- These edit-based accuracies can be negative when insertions are numerous; do not clamp them. Report counts and denominators.
- chrF++: SacreBLEU CHRF with char_order=6, word_order=2, beta=2, lowercase=false, whitespace=false. Record package version and metric signature. Score gloss and transliteration separately.
- Exact sentence match is optional, not the primary measure.

An invalid response counts as an empty prediction for accuracy/MER; chrF++ also receives an empty string. Retain every test example and report parse-failure and truncation rates. Report empty reference cases separately; never divide by zero.

## Checkpoint comparison
For a fixed training run, select one checkpoint with minimum validation loss and another with maximum validation gloss chrF++. Break ties by earlier epoch. Generate end-to-end two-stage predictions for chrF++; do not condition generation on reference transliteration. Evaluate both selected checkpoints on the held-out test split once the analysis protocol is frozen.

## Verification before use
Test identity, substitution, insertion, deletion, empty output, malformed JSON, Unicode normalization, and hyphen/equals boundaries using hand-checked examples. Training loss gradients must never be enabled during evaluation.
