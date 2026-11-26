# Few-shot with morpheme segmentation (Ainu first)

Tutor (1 Dec agenda): prompt the model to first transliterate, then segment, then gloss. Try one language first; extend to the others only if it improves the results.

One call, three answer lines (prompt: `segmentation/prompts/three_step.txt`, an extension of the step-by-step prompt):

```
Transliteration: oraun cise soy ta ekan ruwe ne akusu ,
Segmentation: oraun cise soy ta ek =an ruwe ne akusu ,
Gloss: then house outside at come.SG=4.S INFR.EV COP then
```

- **Where the segmentation comes from.** The gold transcription is the segmented transliteration: clitics are split off with `=` (`a= kor`, `ek =an`; in 532 of 750 Ainu test rows). Affixes are not marked: there is no `-` in any Ainu transcription, while the gloss does split affixes (`aeyayramekotep` = `4.A- with.APPL- live -thing`). For the examples, the Step 1 line is the gold transcription with the `=` boundaries removed.
- **Same setup as the Ainu 3-shot run**, so the results compare directly with `results/few_shot/ainu/`: the same 3 train examples (seed 42), greedy decoding, `max_new_tokens` 768. Only example 3 contains a clitic; in examples 1 and 2 the Segmentation line equals the Transliteration line.
- **Scoring.** The predicted Segmentation line is scored as the transliteration (against the gold transcription), so transliteration scores stay comparable. The Step 1 line is kept in `pred_transliteration_unsegmented`. `scores.csv` adds `segmentation_exact_rate`.
- `--fast` adds the speed-ups of `fast_run/run_fast.py`. By default it is off, so generation is the same as in the earlier runs.

```
python segmentation/run_segmentation.py --lang ainu --split dev --limit 10   # smoke test
python segmentation/run_segmentation.py --lang ainu --split test --resume    # full run
python segmentation/score_segmentation.py --lang ainu                        # scores
```

Output: `results/segmentation/<lang>/bolmo-1b_three_step_seg_3shot_test.csv` (+ `.meta.json`, `scores.csv`).
