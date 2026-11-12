# Korean IGT with Bolmo 1B/7B

Research scaffold for Korean interlinear glossed text (IGT) generation with Bolmo 1B and 7B.

## Research questions
1. How do 1B and 7B models perform without task examples?
2. Does predicting transliteration first improve gloss generation compared with generating both fields together?
3. How does LoRA adaptation change performance?
4. Does selecting checkpoints by validation loss or generation-based chrF++ produce different test results?

## Experimental plan
- Zero-shot: model size (1B/7B) × prompting strategy (all-in-one/two-stage).
- LoRA: start with 1B; repeat feasible experiments with 7B. Compare ranks 8, 16, 32 and learning rates 1e-4, 2e-4 with one-factor-at-a-time pilot runs.
- Checkpoint selection: train once per configuration, evaluate the same saved checkpoints with validation loss and chrF++, then compare the independently selected checkpoints.
- Report morpheme and word accuracy, morpheme error rate, and chrF++; report transliteration separately from glossing.
- Use a fixed test split only after prompt, hyperparameter, and checkpoint choices are frozen.

## Three-person work allocation
Names are intentionally unassigned until the team confirms them.

| Role | Primary responsibility | Deliverables |
| --- | --- | --- |
| Member 1 — Data and zero-shot | Audit Korean data, freeze splits, implement both prompts for 1B/7B | Dataset manifest, zero-shot predictions, prompt comparison |
| Member 2 — LoRA | Prepare supervised examples, implement adapter training, run rank/LR pilots | Training code, configs, adapter/checkpoint metadata |
| Member 3 — Evaluation and analysis | Implement and validate metrics, compare checkpoint selection, analyze errors | Evaluation code, comparison tables, report figures |
| All three | Cross-review one another's outputs and maintain the experiment log | Reproducible report and weekly progress notes |

## Repository layout
- `zero_shot/`: prompt templates and inference protocol.
- `lora/`: training and ablation protocol.
- `evaluation/`: metric definitions and checkpoint-selection rules.
- `data/`: schema, split policy, and source manifest template.
- `configs/`: experiment specifications in JSON.
- `results/`: empty results table and reporting convention.

## Getting started
1. Read `data/README.md`; audit the source rather than assuming every text field is valid Korean.
2. Fill in exact Hugging Face model IDs, immutable revisions, dataset ID/config/revision, and split hashes in the configs.
3. Implement inference, training, and evaluation against the documented schemas.
4. Run a small development smoke test, freeze prompts and metric conventions, then run the full experiments.

This initial commit contains project documentation, prompts, and configuration templates. It does **not** contain runnable training/inference implementations, downloaded models, data, or experimental results. Configs describe intended experiments; they are not arguments for an existing CLI.

## Reproducibility
Record Git commit, model/tokenizer revision, dataset revision and hashes, package versions, GPU, seed, prompt version, generation settings, and runtime per run. Never commit access tokens, private data, model weights, or checkpoints. Follow source data and model licenses before use or redistribution.
