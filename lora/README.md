# LoRA experiments

These are proposed starting settings, not validated optimal hyperparameters.

Train causal-language-model adapters using training examples only. Mask prompt/padding tokens with -100 so loss is computed only on target tokens. For all-in-one training, target the two-field JSON response. For two-stage training, create a transliteration example and a gloss example for each training item. Gold transliteration may condition the gloss training example; document this teacher-forcing choice and its exposure mismatch. At validation generation and test time, always chain predicted transliteration into stage 2.

Audit the actual architecture before assigning LoRA target modules. The config's candidate projection names are suggestions; resolve them against named_modules and record the exact matched names and trainable parameter count. Fail if none match.

Start with rank 16, alpha 32, dropout 0.05, learning rate 2e-4, 3 epochs, effective batch size 16, seed 42. Pilot ranks 8/32 (alpha = 2 × rank) and LR 1e-4 independently. Repeat final comparisons with seeds 42, 43, 44 if resources permit. The 7B run may need smaller micro-batches or separately documented QLoRA; do not label quantized and non-quantized runs as identical conditions.

Save and evaluate at the same epoch boundaries for both checkpoint-selection rules. Validation uses no gradients and never updates weights. Validation loss uses teacher-forced targets; chrF++ uses generated glosses. Keep the test split out of tuning and checkpoint selection.
