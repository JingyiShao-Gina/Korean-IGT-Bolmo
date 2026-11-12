# Data contract and audit

Dataset collection to investigate: https://huggingface.co/lowresource-igt/datasets

No data is redistributed in this scaffold. Confirm the exact Korean dataset, configuration, license, field meanings, and immutable revision before downloading or training.

Local JSONL records use:
- id: stable unique string
- source: Korean source text
- transliteration: reference segmented romanization
- gloss: reference English morpheme gloss
- translation: optional free translation; excluded from baseline inputs
- split: train, validation, or test

Audit source language and encoding, swapped columns, missing fields, duplicates, reference segmentation, and inconsistent gloss labels. Do not assume a column called text is a clean Korean source. Keep original values and a logged transformation mapping. Exclude or quarantine suspect records with reasons; never fabricate corrections.

Prefer official splits. Check overlap across all splits by normalized source and document/source identity; group related examples to prevent leakage. If official splits do not exist, use a deterministic grouped 80/10/10 split with seed 42, record the algorithm and IDs, and freeze split hashes. Do not move test examples based on model performance.

Keep raw/processed data local and ignored. Commit only permitted metadata and small synthetic examples. Fill in manifest.template.json after verification.
