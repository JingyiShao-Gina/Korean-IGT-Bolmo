"""Check a language's data before LoRA training (no GPU, no model):

    python lora/check_data.py --lang ainu
    python lora/check_data.py                 # every language in configs/zero_shot.json
    python lora/check_data.py --lang tsez --shots 3 20

Per split: rows, rows with an empty script/translation/transliteration/gloss (left out of training), the longest training
example in bytes per strategy and how many exceed training.max_seq_bytes, and the longest gold answer
against generation.max_new_bytes. With --shots K: the longest K-shot prompt (fixed examples of
run_zero_shot.py) that `predict --shots K` would send.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lora_bolmo as lb  # noqa: E402

rz = lb.rz


def check(lang: str, shots: list[int]) -> dict:
    cfg = lb.load_config()
    max_seq, max_new = cfg["training"]["max_seq_bytes"], cfg["generation"]["max_new_bytes"]
    info = lb.lang_info(lang)
    templates = lb.load_templates(lang)
    print(f"\n=== {lang} ({info['language_name']})  max_seq_bytes={max_seq}  max_new_bytes={max_new}")
    out = {}
    for split in ("train", "dev", "test"):
        rows = lb._rows(lang, split, None)
        answers = [len(lb.answer_fields(r)[0].encode()) + len(lb.answer_fields(r)[1].encode()) + 8
                   for r in rows]  # ' ' + translit + '\nGloss: ' + gloss
        line = f"{split:5s} rows {len(rows):5d}  empty field {sum(not lb.usable(r) for r in rows):4d}  " \
               f"longest answer {max(answers)} bytes ({sum(a >= max_new for a in answers)} >= budget)"
        for strategy in lb.STRATEGIES:
            make_prompt = lb.prompt_fn(templates, strategy, info["language_name"])
            _, rep = lb.prepare_examples(rows, strategy, make_prompt, max_seq)
            line += f"  | {strategy}: longest {rep['max_example_bytes']}, too long {len(rep['dropped_too_long'])}"
            out[(split, strategy)] = rep
        print(line)
    for k in shots:
        try:
            ex = lb.few_shot_examples(lang, k)
        except ValueError as err:
            print(f"{k}-shot: {err}")
            continue
        for strategy in lb.STRATEGIES:
            make_prompt = lb.prompt_fn(templates, strategy, info["language_name"], ex)
            longest = max(len(make_prompt(r).encode()) for r in lb._rows(lang, "test", None))
            print(f"{k}-shot {strategy}: longest test prompt {longest} bytes (+ up to {max_new} generated)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", nargs="*", default=None, help=f"default: all of {lb.languages()}")
    ap.add_argument("--shots", nargs="*", type=int, default=[], help="also measure K-shot prompts")
    args = ap.parse_args(argv)
    for lang in args.lang or lb.languages():
        check(lang, args.shots)


if __name__ == "__main__":
    main()
