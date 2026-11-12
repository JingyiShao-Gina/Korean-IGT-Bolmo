# Evaluation protocol

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
